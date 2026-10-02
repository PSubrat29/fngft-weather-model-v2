from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class VariableOrderFractionalOperator(nn.Module):
    """Research approximation of a spatially varying fractional Laplacian on a regular grid.

    The field is mirrored across the latitude edges (and the longitude edges when longitude is not
    periodic) before the FFT, so the spectral operator sees a continuous field instead of an
    artificial jump between the northern and southern boundaries. Wavenumbers are normalised to
    [0, 1] so the response is bounded and independent of grid resolution.

    A bank of fixed orders is combined with positive weights derived from the local alpha field.
    Global operational use still requires a sphere-aware operator.
    """

    def __init__(self, alpha_min: float, alpha_max: float, basis_orders: int, sigma: float = 0.85, periodic_x: bool = True):
        super().__init__()
        basis = torch.linspace(alpha_min, alpha_max, basis_orders)
        self.register_buffer("basis_orders", basis)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.sigma = sigma
        self.periodic_x = periodic_x

    def basis_weights(self, alpha: Tensor) -> Tensor:
        """Normalised Gaussian weights [B,K,H,W] of each basis order for the local alpha."""
        spacing = max((self.alpha_max - self.alpha_min) / max(len(self.basis_orders) - 1, 1), 1e-5)
        width = max(self.sigma * spacing, 1e-4)
        dist2 = (alpha - self.basis_orders.view(1, -1, 1, 1)) ** 2
        weights = torch.exp(-0.5 * dist2 / (width * width))
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: Tensor, alpha: Tensor) -> Tensor:
        _, _, h, w = x.shape
        ext = torch.cat([x, x.flip(-2)], dim=-2)
        if not self.periodic_x:
            ext = torch.cat([ext, ext.flip(-1)], dim=-1)
        eh, ew = ext.shape[-2:]
        x_hat = torch.fft.rfft2(ext, norm="ortho")
        ky = 2.0 * math.pi * torch.fft.fftfreq(eh, device=x.device, dtype=x.dtype)
        kx = 2.0 * math.pi * torch.fft.rfftfreq(ew, device=x.device, dtype=x.dtype)
        k2 = (ky[:, None] ** 2 + kx[None, :] ** 2) / (2.0 * math.pi**2)
        weights = self.basis_weights(alpha)
        out = torch.zeros_like(x)
        for i, order in enumerate(self.basis_orders):
            multiplier = k2.pow(order / 2.0)
            response = torch.fft.irfft2(x_hat * multiplier, s=(eh, ew), norm="ortho")[..., :h, :w]
            out = out + weights[:, i : i + 1] * response
        return out


class FractionalMemoryOperator(nn.Module):
    """Causal temporal power-law aggregation of fractional spatial responses."""

    def __init__(
        self,
        *,
        history: int,
        alpha_min: float,
        alpha_max: float,
        beta_min: float,
        beta_max: float,
        basis_orders: int,
        sigma: float,
        periodic_x: bool = True,
    ):
        super().__init__()
        self.history = history
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.spatial = VariableOrderFractionalOperator(alpha_min, alpha_max, basis_orders, sigma, periodic_x)

    def forward(self, history: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        if history.shape[1] != self.history:
            raise ValueError(f"Expected {self.history} history steps, got {history.shape[1]}")
        b, t, c, h, w = history.shape
        # All history steps share the same alpha field, so the spatial operator runs once on [B, T*C, H, W].
        resp = self.spatial(history.reshape(b, t * c, h, w), alpha).view(b, t, c, h, w)
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
