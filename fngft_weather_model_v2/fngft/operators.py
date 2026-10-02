from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class VariableOrderFractionalOperator(nn.Module):
    """Research approximation of a spatially varying fractional Laplacian on a regular grid.

    The operator is implemented with FFTs, so both spatial directions are treated as periodic.
    For real Earth data this is a deliberate prototype approximation; global operational use
    requires a sphere-aware operator or a domain transform before deployment.
    """

    def __init__(self, alpha_min: float, alpha_max: float, basis_orders: int, sigma: float = 0.85):
        super().__init__()
        basis = torch.linspace(alpha_min, alpha_max, basis_orders)
        self.register_buffer("basis_orders", basis)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.sigma = sigma

    def forward(self, x: Tensor, alpha: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x_hat = torch.fft.rfft2(x, norm="ortho")
        ky = 2.0 * math.pi * torch.fft.fftfreq(h, d=1.0 / h, device=x.device)
        kx = 2.0 * math.pi * torch.fft.rfftfreq(w, d=1.0 / w, device=x.device)
        k2 = ky[:, None] ** 2 + kx[None, :] ** 2
        basis_outputs = []
        for order in self.basis_orders:
            multiplier = k2.clamp_min(0.0).pow(order / 2.0).unsqueeze(0).unsqueeze(0)
            basis_outputs.append(torch.fft.irfft2(x_hat * multiplier, s=(h, w), norm="ortho"))
        stack = torch.stack(basis_outputs, dim=1)  # [B,K,C,H,W]
        spacing = max((self.alpha_max - self.alpha_min) / max(len(self.basis_orders) - 1, 1), 1e-5)
        width = max(self.sigma * spacing, 1e-4)
        dist2 = (alpha - self.basis_orders.view(1, -1, 1, 1)) ** 2
        weights = torch.exp(-0.5 * dist2 / (width * width))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (weights.unsqueeze(2) * stack).sum(dim=1)


class FractionalMemoryOperator(nn.Module):
    """Causal temporal power-law aggregation of fractional spatial responses."""

    def __init__(self, *, history: int, alpha_min: float, alpha_max: float, beta_min: float, beta_max: float, basis_orders: int, sigma: float):
        super().__init__()
        self.history = history
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.spatial = VariableOrderFractionalOperator(alpha_min, alpha_max, basis_orders, sigma)

    def forward(self, history: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        if history.shape[1] != self.history:
            raise ValueError(f"Expected {self.history} history steps, got {history.shape[1]}")
        responses = [self.spatial(history[:, i], alpha) for i in range(self.history)]
        resp = torch.stack(responses, dim=1)
        lags = torch.arange(self.history, 0, -1, dtype=history.dtype, device=history.device).view(1, self.history, 1, 1, 1)
        beta_eff = beta.clamp(self.beta_min, self.beta_max)
        weights = lags.pow(-beta_eff.unsqueeze(1))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (weights * resp).sum(dim=1)


def masked_causal_kernel(beta: Tensor, history: int) -> Tensor:
    """Return normalized lag weights for diagnostics."""
    lags = torch.arange(history, 0, -1, dtype=beta.dtype, device=beta.device).view(1, history)
    weights = lags.pow(-beta.reshape(-1, 1))
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
