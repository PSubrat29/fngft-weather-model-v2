from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

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
        # Learned positional embedding: without it self-attention ignores the order of the history.
        self.position = nn.Parameter(torch.zeros(1, cfg.history, cfg.memory_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.memory_dim,
            nhead=cfg.memory_heads,
            dim_feedforward=2 * cfg.memory_dim,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.memory_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.memory_dim)

    def forward(self, history: Tensor) -> Tensor:
        pooled = history.mean(dim=(-1, -2))
        x = self.input(pooled) + self.position[:, -pooled.shape[1] :]
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


def _index_divergence(fx: Tensor, fy: Tensor, periodic_x: bool) -> Tensor:
    """Flux divergence in grid-index units (per forecast step, standardized units)."""
    if periodic_x:
        dx = (torch.roll(fx, shifts=-1, dims=-1) - torch.roll(fx, shifts=1, dims=-1)) * 0.5
    else:
        dx = torch.cat(
            [fx[..., 1:2] - fx[..., 0:1], (fx[..., 2:] - fx[..., :-2]) * 0.5, fx[..., -1:] - fx[..., -2:-1]], dim=-1
        )
    dy = torch.cat(
        [fy[..., 1:2, :] - fy[..., 0:1, :], (fy[..., 2:, :] - fy[..., :-2, :]) * 0.5, fy[..., -1:, :] - fy[..., -2:-1, :]],
        dim=-2,
    )
    return dx + dy


class FractionalClosure(nn.Module):
    """Learned fractional closure added to the explicit physics branch.

    The closure is a per-step increment in standardized units. The learned residual output layers
    start at zero, so an untrained model equals physics + fractional damping and training grows the
    correction from there.
    """

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
            periodic_x=cfg.longitude_periodic,
        )
        self.feature = nn.Sequential(
            nn.Conv2d(cfg.in_channels + cfg.memory_dim, cfg.hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.hidden, cfg.hidden, 3, padding=1),
            nn.GELU(),
        )
        self.force = nn.Conv2d(cfg.hidden, 2, 1)
        self.scalar_flux = nn.Conv2d(cfg.hidden, 4, 1)
        for layer in (self.force, self.scalar_flux):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def divergence_flux(self, fx: Tensor, fy: Tensor) -> Tensor:
        return _index_divergence(fx, fy, self.cfg.longitude_periodic)

    def forward(self, history: Tensor, memory: Tensor, orders: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
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
        theta_t = -self.divergence_flux(theta_fx, theta_fy)
        q_t = -self.divergence_flux(q_fx, q_fy)
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
    """Top-level real-data prototype: memory + dynamic orders + fractional closure + physics.

    The model consumes and produces standardized states. The physics branch runs in physical units
    using the training normalization statistics stored in the ``state_mean``/``state_std`` buffers
    (saved with the checkpoint).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.dt_hours is None or cfg.dt_hours <= 0:
            raise ValueError("ModelConfig.dt_hours must be a positive number of hours (training infers it from the data)")
        if cfg.in_channels != 4:
            raise ValueError("ModelConfig.in_channels must be 4 (u, v, theta, q)")
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
                periodic_x=cfg.longitude_periodic,
            )
        )
        self.register_buffer("state_mean", torch.zeros(cfg.in_channels))
        self.register_buffer("state_std", torch.ones(cfg.in_channels))

    def set_normalization(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.state_mean.copy_(torch.as_tensor(mean, dtype=self.state_mean.dtype))
        self.state_std.copy_(torch.as_tensor(std, dtype=self.state_std.dtype))

    def to_physical(self, standardized: Tensor) -> Tensor:
        return standardized * self.state_std.view(1, -1, 1, 1) + self.state_mean.view(1, -1, 1, 1)

    def to_standardized(self, physical: Tensor) -> Tensor:
        return (physical - self.state_mean.view(1, -1, 1, 1)) / self.state_std.view(1, -1, 1, 1)

    def step(self, history: Tensor, lat_deg: Tensor, lon_deg: Tensor) -> tuple[Tensor, Dict[str, Tensor]]:
        if history.ndim != 5:
            raise ValueError("history must have shape [batch,time,channel,lat,lon]")
        if history.shape[1] != self.cfg.history:
            raise ValueError(f"Expected history={self.cfg.history}, got {history.shape[1]}")
        if history.shape[-2] != lat_deg.numel() or history.shape[-1] != lon_deg.numel():
            raise ValueError("lat/lon lengths do not match the history grid")
        memory = self.memory(history)
        orders = self.order_head(history[:, -1], memory)
        closure, diagnostics = self.closure(history, memory, orders)
        physical = self.to_standardized(self.physics(self.to_physical(history[:, -1]), lat_deg, lon_deg))
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


def build_model(model_config: dict, normalizer: Optional[dict] = None) -> FNGFTWeatherModel:
    """Recreate a model from the ``model_config`` dict stored in a checkpoint."""
    model = FNGFTWeatherModel(ModelConfig(**model_config))
    if normalizer is not None:
        model.set_normalization(normalizer["mean"], normalizer["std"])
    return model
