from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig
from .operators import FractionalMemoryOperator
from .physics import RealGridPhysicsConfig, RealGridPhysicsCore


class MemoryTransformer(nn.Module):
    """Compresses the recent state history into a learned memory vector."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input = nn.Linear(cfg.in_channels, cfg.memory_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.memory_dim,
            nhead=cfg.memory_heads,
            dim_feedforward=2 * cfg.memory_dim,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.memory_layers)
        self.norm = nn.LayerNorm(cfg.memory_dim)

    def forward(self, history: Tensor) -> Tensor:
        pooled = history.mean(dim=(-1, -2))
        x = self.input(pooled)
        x = self.encoder(x)
        return self.norm(x[:, -1])


class OrderHead(nn.Module):
    """Predicts bounded alpha/beta fields and positive kappa from X_t and H_t."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.state_net = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.hidden, cfg.hidden, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(cfg.hidden + cfg.memory_dim, 3, 1)

    def forward(self, current: Tensor, memory: Tensor) -> Dict[str, Tensor]:
        z = self.state_net(current)
        m = memory[:, :, None, None].expand(-1, -1, z.shape[-2], z.shape[-1])
        raw = self.out(torch.cat([z, m], dim=1))
        alpha = self.cfg.alpha_min + (self.cfg.alpha_max - self.cfg.alpha_min) * torch.sigmoid(raw[:, 0:1])
        beta = self.cfg.beta_min + (self.cfg.beta_max - self.cfg.beta_min) * torch.sigmoid(raw[:, 1:2])
        log_kappa = raw[:, 2:3].clamp(-8.0, 4.0)
        kappa = F.softplus(log_kappa) + 1e-5
        return {"alpha": alpha, "beta": beta, "kappa": kappa}


class FractionalClosure(nn.Module):
    """Learned fractional closure added to the explicit physics branch."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.memory_op = FractionalMemoryOperator(
            history=cfg.history,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            beta_min=cfg.beta_min,
            beta_max=cfg.beta_max,
            basis_orders=cfg.basis_orders,
            sigma=cfg.operator_sigma,
        )
        self.feature = nn.Sequential(
            nn.Conv2d(cfg.in_channels + cfg.memory_dim, cfg.hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.hidden, cfg.hidden, 3, padding=1),
            nn.GELU(),
        )
        self.force = nn.Conv2d(cfg.hidden, 2, 1)
        self.scalar_flux = nn.Conv2d(cfg.hidden, 4, 1)

    @staticmethod
    def divergence_flux(fx: Tensor, fy: Tensor, lat_deg: Tensor, lon_deg: Tensor) -> Tensor:
        from .physics import spherical_grad
        gx, _ = spherical_grad(fx, lat_deg, lon_deg)
        _, gy = spherical_grad(fy, lat_deg, lon_deg)
        return gx + gy

    def forward(self, history: Tensor, memory: Tensor, orders: Dict[str, Tensor], lat_deg: Tensor, lon_deg: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        alpha, beta, kappa = orders["alpha"], orders["beta"], orders["kappa"]
        frac = self.memory_op(history, alpha, beta)
        current = history[:, -1]
        m = memory[:, :, None, None].expand(-1, -1, current.shape[-2], current.shape[-1])
        feat = self.feature(torch.cat([current, m], dim=1))
        force_resid = self.force(feat)
        momentum = -self.cfg.fractional_scale * kappa * frac[:, :2] + self.cfg.residual_scale * force_resid
        flux_resid = self.scalar_flux(feat)
        theta_base = -self.cfg.fractional_scale * kappa * frac[:, 2:3]
        q_base = -self.cfg.fractional_scale * kappa * frac[:, 3:4]
        theta_fx = theta_base + self.cfg.residual_scale * flux_resid[:, 0:1]
        theta_fy = self.cfg.residual_scale * flux_resid[:, 1:2]
        q_fx = q_base + self.cfg.residual_scale * flux_resid[:, 2:3]
        q_fy = self.cfg.residual_scale * flux_resid[:, 3:4]
        theta_t = -self.divergence_flux(theta_fx, theta_fy, lat_deg, lon_deg)
        q_t = -self.divergence_flux(q_fx, q_fy, lat_deg, lon_deg)
        closure = torch.cat([momentum, theta_t, q_t], dim=1)
        return closure, {
            "fractional_response": frac,
            "theta_flux_x": theta_fx,
            "theta_flux_y": theta_fy,
            "q_flux_x": q_fx,
            "q_flux_y": q_fy,
            "closure_tendency": closure,
        }


class FNGFTWeatherModel(nn.Module):
    """Top-level real-data prototype: memory + dynamic orders + fractional closure + physics."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.memory = MemoryTransformer(cfg)
        self.order_head = OrderHead(cfg)
        self.closure = FractionalClosure(cfg)
        self.physics = RealGridPhysicsCore(
            RealGridPhysicsConfig(
                dt_hours=cfg.dt_hours,
                coriolis_scale=cfg.coriolis_scale,
                stratification_scale=cfg.stratification_scale,
                diffusion_scale=cfg.diffusion_scale,
            )
        )

    def step(self, history: Tensor, lat_deg: Tensor, lon_deg: Tensor) -> tuple[Tensor, Dict[str, Tensor]]:
        if history.ndim != 5:
            raise ValueError("history must have shape [batch,time,channel,lat,lon]")
        if history.shape[1] != self.cfg.history:
            raise ValueError(f"Expected history={self.cfg.history}, got {history.shape[1]}")
        memory = self.memory(history)
        orders = self.order_head(history[:, -1], memory)
        closure, diagnostics = self.closure(history, memory, orders, lat_deg, lon_deg)
        physical = self.physics(history[:, -1], lat_deg, lon_deg)
        next_state = physical + self.cfg.closure_scale * closure
        diagnostics.update(orders)
        diagnostics["memory"] = memory
        diagnostics["physical_state"] = physical
        diagnostics["forecast_state"] = next_state
        return next_state, diagnostics

    def forward(self, history: Tensor, lat_deg: Tensor, lon_deg: Tensor, steps: int = 1) -> tuple[Tensor, Dict[str, Tensor]]:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        current = history
        preds = []
        last_info: Dict[str, Tensor] = {}
        for _ in range(steps):
            pred, last_info = self.step(current, lat_deg, lon_deg)
            preds.append(pred)
            current = torch.cat([current[:, 1:], pred.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1), last_info
