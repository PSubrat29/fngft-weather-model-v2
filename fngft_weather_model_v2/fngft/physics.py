from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

EARTH_RADIUS_M = 6_371_000.0
OMEGA = 7.2921159e-5


def _periodic_x_gradient(field: Tensor, dx: Tensor) -> Tensor:
    """Central longitude derivative with periodic boundary."""
    rolled_plus = torch.roll(field, shifts=-1, dims=-1)
    rolled_minus = torch.roll(field, shifts=1, dims=-1)
    return (rolled_plus - rolled_minus) / (2.0 * dx)


def _latitude_gradient(field: Tensor, lat_rad: Tensor) -> Tensor:
    """First/second-order finite difference gradient along latitude."""
    out = torch.empty_like(field)
    dphi = torch.diff(lat_rad).abs().mean().clamp_min(1e-8)
    out[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dphi)
    out[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / dphi
    out[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / dphi
    return out


def spherical_grad(field: Tensor, lat_deg: Tensor, lon_deg: Tensor) -> tuple[Tensor, Tensor]:
    """Approximate physical east/north gradients on a regular lat/lon grid."""
    lat = torch.deg2rad(lat_deg).to(field.device, field.dtype)
    lon = torch.deg2rad(lon_deg).to(field.device, field.dtype)
    dlon = torch.diff(lon).abs().mean().clamp_min(1e-8)
    dphi = torch.diff(lat).abs().mean().clamp_min(1e-8)
    coslat = lat.clamp(-math.pi / 2 + 1e-5, math.pi / 2 - 1e-5).cos().view(1, 1, -1, 1)
    dx = EARTH_RADIUS_M * coslat * dlon
    dy = EARTH_RADIUS_M * dphi
    d_dlon = _periodic_x_gradient(field, dlon)
    d_dphi = _latitude_gradient(field, lat)
    return d_dlon / (EARTH_RADIUS_M * coslat).clamp_min(1.0), d_dphi / EARTH_RADIUS_M


def advection(field: Tensor, u: Tensor, v: Tensor, lat_deg: Tensor, lon_deg: Tensor) -> Tensor:
    gx, gy = spherical_grad(field, lat_deg, lon_deg)
    return u * gx + v * gy


@dataclass
class RealGridPhysicsConfig:
    dt_hours: float
    coriolis_scale: float
    stratification_scale: float
    diffusion_scale: float


class RealGridPhysicsCore(nn.Module):
    """Reduced physics proxy for real gridded data.

    This is intentionally not a complete compressible/hydrostatic NWP dynamical core.
    It supplies explicit transport, Coriolis, stratification and diffusion so the learned
    network is trained as a closure around a known-physics branch.
    """

    def __init__(self, cfg: RealGridPhysicsConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, state: Tensor, lat_deg: Tensor, lon_deg: Tensor) -> Tensor:
        u, v, theta, q = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]
        lat = torch.deg2rad(lat_deg).to(state.device, state.dtype).view(1, 1, -1, 1)
        f = 2.0 * OMEGA * lat.sin() * self.cfg.coriolis_scale
        adv_u = advection(u, u, v, lat_deg, lon_deg)
        adv_v = advection(v, u, v, lat_deg, lon_deg)
        adv_theta = advection(theta, u, v, lat_deg, lon_deg)
        adv_q = advection(q, u, v, lat_deg, lon_deg)
        dtheta_x, dtheta_y = spherical_grad(theta, lat_deg, lon_deg)
        lap_theta = spherical_divergence_gradient(theta, lat_deg, lon_deg)
        lap_q = spherical_divergence_gradient(q, lat_deg, lon_deg)
        sec_per_step = max(self.cfg.dt_hours, 1e-6) * 3600.0
        du = -adv_u + f * v
        dv = -adv_v - f * u - self.cfg.stratification_scale * dtheta_y
        dtheta = -adv_theta + self.cfg.diffusion_scale * lap_theta
        dq = -adv_q + self.cfg.diffusion_scale * lap_q
        tendency = torch.cat([du, dv, dtheta, dq], dim=1)
        return state + sec_per_step * tendency


def spherical_divergence_gradient(field: Tensor, lat_deg: Tensor, lon_deg: Tensor) -> Tensor:
    # This is a scalar diffusion proxy. A full vector divergence operator belongs in the
    # future sphere-aware atmospheric dynamical-core implementation.
    return laplacian_proxy(field, lat_deg, lon_deg)


def laplacian_proxy(field: Tensor, lat_deg: Tensor, lon_deg: Tensor) -> Tensor:
    gx, gy = spherical_grad(field, lat_deg, lon_deg)
    gx2, _ = spherical_grad(gx, lat_deg, lon_deg)
    _, gy2 = spherical_grad(gy, lat_deg, lon_deg)
    return gx2 + gy2
