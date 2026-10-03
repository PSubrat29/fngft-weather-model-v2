from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

EARTH_RADIUS_M = 6_371_000.0
OMEGA = 7.2921159e-5
# cos(latitude) floor so metric terms stay finite on grids that include the poles.
MIN_COSLAT = 1e-3


def _spacing_rad(coord_deg: Tensor) -> Tensor:
    """Signed mean grid spacing in radians (positive for ascending coordinates)."""
    d = torch.diff(torch.deg2rad(coord_deg)).mean()
    return torch.where(d.abs() < 1e-8, torch.full_like(d, 1e-8), d)


def _x_derivative(field: Tensor, dlon: Tensor, periodic_x: bool) -> Tensor:
    """d/dlon (per radian), central differences; periodic or one-sided at the edges."""
    if periodic_x:
        return (torch.roll(field, shifts=-1, dims=-1) - torch.roll(field, shifts=1, dims=-1)) / (2.0 * dlon)
    interior = (field[..., 2:] - field[..., :-2]) / (2.0 * dlon)
    first = (field[..., 1:2] - field[..., 0:1]) / dlon
    last = (field[..., -1:] - field[..., -2:-1]) / dlon
    return torch.cat([first, interior, last], dim=-1)


def _y_derivative(field: Tensor, dphi: Tensor) -> Tensor:
    """d/dphi (per radian), central differences with one-sided edges."""
    interior = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dphi)
    first = (field[..., 1:2, :] - field[..., 0:1, :]) / dphi
    last = (field[..., -1:, :] - field[..., -2:-1, :]) / dphi
    return torch.cat([first, interior, last], dim=-2)


def spherical_grad(field: Tensor, lat_deg: Tensor, lon_deg: Tensor, periodic_x: bool = True) -> tuple[Tensor, Tensor]:
    """Physical east/north gradients (per metre) on a regular lat/lon grid."""
    lat_deg = lat_deg.to(field.device, field.dtype)
    lon_deg = lon_deg.to(field.device, field.dtype)
    dlon = _spacing_rad(lon_deg)
    dphi = _spacing_rad(lat_deg)
    coslat = torch.deg2rad(lat_deg).cos().clamp_min(MIN_COSLAT).view(1, 1, -1, 1)
    gx = _x_derivative(field, dlon, periodic_x) / (EARTH_RADIUS_M * coslat)
    gy = _y_derivative(field, dphi) / EARTH_RADIUS_M
    return gx, gy


def advection(field: Tensor, u: Tensor, v: Tensor, lat_deg: Tensor, lon_deg: Tensor, periodic_x: bool = True) -> Tensor:
    gx, gy = spherical_grad(field, lat_deg, lon_deg, periodic_x)
    return u * gx + v * gy


def laplacian_proxy(field: Tensor, lat_deg: Tensor, lon_deg: Tensor, periodic_x: bool = True) -> Tensor:
    gx, gy = spherical_grad(field, lat_deg, lon_deg, periodic_x)
    gx2, _ = spherical_grad(gx, lat_deg, lon_deg, periodic_x)
    _, gy2 = spherical_grad(gy, lat_deg, lon_deg, periodic_x)
    return gx2 + gy2


def spherical_divergence_gradient(field: Tensor, lat_deg: Tensor, lon_deg: Tensor, periodic_x: bool = True) -> Tensor:
    # Scalar diffusion proxy. A full vector divergence operator belongs in the future
    # sphere-aware atmospheric dynamical-core implementation.
    return laplacian_proxy(field, lat_deg, lon_deg, periodic_x)


def semi_lagrangian_advect(
    fields: Tensor,
    u: Tensor,
    v: Tensor,
    lat_deg: Tensor,
    lon_deg: Tensor,
    dt_seconds: float,
    periodic_x: bool = True,
) -> Tensor:
    """Advect ``fields`` [B,C,H,W] by winds ``u``/``v`` [B,1,H,W] (m/s) over ``dt_seconds``.

    Departure points are traced back along the local wind and the fields are interpolated
    bilinearly there. Unlike explicit centred differences this is stable for any Courant number,
    which matters for real grids (fine resolution, long time steps, fast jets).
    """
    b, _, h, w = fields.shape
    lat_deg = lat_deg.to(fields.device, fields.dtype)
    lon_deg = lon_deg.to(fields.device, fields.dtype)
    dphi = _spacing_rad(lat_deg)
    dlon = _spacing_rad(lon_deg)
    coslat = torch.deg2rad(lat_deg).cos().clamp_min(MIN_COSLAT).view(1, 1, h, 1)
    di = v * dt_seconds / (EARTH_RADIUS_M * dphi)
    dj = u * dt_seconds / (EARTH_RADIUS_M * coslat * dlon)
    # A row lying on a pole is a single point: longitude is undefined there, so no zonal displacement.
    # (With the cos(lat) floor the departure would be hundreds of columns and scramble the row.)
    on_pole = (0.5 * torch.pi - torch.deg2rad(lat_deg).abs()) < 0.25 * dphi.abs()
    dj = dj.masked_fill(on_pole.view(1, 1, h, 1), 0.0)
    rows = torch.arange(h, device=fields.device, dtype=fields.dtype).view(1, 1, h, 1)
    cols = torch.arange(w, device=fields.device, dtype=fields.dtype).view(1, 1, 1, w)
    ii = (rows - di).clamp(0.0, h - 1.0)
    jj = cols - dj
    if periodic_x:
        jj = torch.remainder(jj, float(w))
        source = torch.cat([fields, fields[..., :1]], dim=-1)  # wrap column for interpolation
        x_norm = 2.0 * jj / float(w) - 1.0
    else:
        jj = jj.clamp(0.0, w - 1.0)
        source = fields
        x_norm = 2.0 * jj / float(w - 1) - 1.0
    y_norm = 2.0 * ii / float(h - 1) - 1.0
    grid = torch.stack([x_norm.expand(b, 1, h, w)[:, 0], y_norm.expand(b, 1, h, w)[:, 0]], dim=-1)
    return F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)


@dataclass
class RealGridPhysicsConfig:
    dt_hours: float
    coriolis_scale: float
    stratification_scale: float
    diffusion_scale: float
    periodic_x: bool = True


class RealGridPhysicsCore(nn.Module):
    """Reduced physics proxy for real gridded data, operating in physical units.

    Input/output: state [B,4,H,W] with u, v in m/s and theta, q in their dataset units.

    One step consists of
      1. semi-Lagrangian advection of u, v, theta, q by the current wind,
      2. exact rotation of the wind by the Coriolis angle f*dt (energy conserving, stable),
      3. explicit stratification (dv -= s * dtheta/dy) and scalar diffusion terms.

    This is intentionally not a complete compressible/hydrostatic NWP dynamical core. It supplies
    explicit transport, rotation and smoothing so the learned network acts as a closure around a
    known-physics branch.
    """

    def __init__(self, cfg: RealGridPhysicsConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, state: Tensor, lat_deg: Tensor, lon_deg: Tensor) -> Tensor:
        periodic = self.cfg.periodic_x
        dt = max(float(self.cfg.dt_hours), 1e-6) * 3600.0
        u, v, theta = state[:, 0:1], state[:, 1:2], state[:, 2:3]
        advected = semi_lagrangian_advect(state, u, v, lat_deg, lon_deg, dt, periodic)
        ua, va, theta_a, q_a = advected[:, 0:1], advected[:, 1:2], advected[:, 2:3], advected[:, 3:4]
        if self.cfg.coriolis_scale != 0.0:
            lat = torch.deg2rad(lat_deg.to(state.device, state.dtype)).view(1, 1, -1, 1)
            angle = 2.0 * OMEGA * lat.sin() * self.cfg.coriolis_scale * dt
            c, s = angle.cos(), angle.sin()
            ua, va = ua * c + va * s, va * c - ua * s
        if self.cfg.stratification_scale != 0.0:
            _, dtheta_y = spherical_grad(theta, lat_deg, lon_deg, periodic)
            va = va - dt * self.cfg.stratification_scale * dtheta_y
        if self.cfg.diffusion_scale != 0.0:
            theta_a = theta_a + dt * self.cfg.diffusion_scale * laplacian_proxy(state[:, 2:3], lat_deg, lon_deg, periodic)
            q_a = q_a + dt * self.cfg.diffusion_scale * laplacian_proxy(state[:, 3:4], lat_deg, lon_deg, periodic)
        return torch.cat([ua, va, theta_a, q_a], dim=1)
