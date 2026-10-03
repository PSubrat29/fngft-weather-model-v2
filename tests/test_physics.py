import math

import torch

from fngft.model import FNGFTWeatherModel
from fngft.config import ModelConfig
from fngft.operators import VariableOrderFractionalOperator
from fngft.physics import EARTH_RADIUS_M, RealGridPhysicsConfig, RealGridPhysicsCore, semi_lagrangian_advect


def _grid(h=8, w=16):
    lat = torch.linspace(-30.0, 30.0, h)
    lon = torch.arange(w, dtype=torch.float32) * (360.0 / w)
    return lat, lon


def test_zero_wind_is_identity():
    lat, lon = _grid()
    field = torch.randn(2, 3, 8, 16)
    zero = torch.zeros(2, 1, 8, 16)
    out = semi_lagrangian_advect(field, zero, zero, lat, lon, 3600.0)
    assert torch.allclose(out, field, atol=1e-5)


def test_uniform_zonal_wind_shifts_one_cell_periodically():
    lat = torch.tensor([-0.001, 0.0, 0.001, 0.002])  # equator rows: dx = R * dlon
    lon = torch.arange(16, dtype=torch.float32) * 22.5
    field = torch.randn(1, 1, 4, 16)
    dt = 3600.0
    dx = EARTH_RADIUS_M * math.radians(22.5)
    u = torch.full((1, 1, 4, 16), dx / dt)
    out = semi_lagrangian_advect(field, u, torch.zeros_like(u), lat, lon, dt, periodic_x=True)
    assert torch.allclose(out, torch.roll(field, shifts=1, dims=-1), atol=1e-3)


def test_physics_long_rollout_stays_bounded():
    # Strong jet, coarse grid and long steps: explicit centred schemes blow up here.
    lat, lon = _grid(16, 32)
    core = RealGridPhysicsCore(RealGridPhysicsConfig(dt_hours=6.0, coriolis_scale=1.0, stratification_scale=0.1, diffusion_scale=1e-4))
    state = torch.randn(1, 4, 16, 32)
    state[:, 0] = state[:, 0] * 10 + 50.0
    state[:, 2] = state[:, 2] + 290.0
    start_max = state[:, 2:].abs().max()
    for _ in range(200):
        state = core(state, lat, lon)
    assert torch.isfinite(state).all()
    assert state[:, 2:].abs().max() <= start_max * 1.01
    speed = state[:, :2].pow(2).sum(1).sqrt().max()
    assert speed < 200.0


def test_fractional_operator_removes_constant_and_is_bounded():
    op = VariableOrderFractionalOperator(0.25, 2.0, 8)
    alpha = torch.full((1, 1, 8, 16), 1.0)
    const = torch.full((1, 2, 8, 16), 5.0)
    assert op(const, alpha).abs().max() < 1e-4
    noise = torch.randn(1, 2, 8, 16)
    assert op(noise, alpha).abs().max() < 3 * noise.abs().max()


def test_untrained_model_is_finite_on_regional_grid():
    cfg = ModelConfig(hidden=8, memory_dim=8, memory_heads=2, memory_layers=1, history=3, dt_hours=1.0, longitude_periodic=False)
    model = FNGFTWeatherModel(cfg)
    model.set_normalization([5.0, 0.0, 290.0, 0.01], [8.0, 6.0, 10.0, 0.004])
    lat = torch.linspace(5.0, 35.0, 10)
    lon = torch.linspace(65.0, 100.0, 12)
    pred, _ = model(torch.randn(2, 3, 4, 10, 12), lat, lon, steps=4)
    assert pred.shape == (2, 4, 4, 10, 12)
    assert torch.isfinite(pred).all()


def test_pole_rows_are_not_scrambled_by_zonal_wind():
    lat = torch.linspace(-90.0, 90.0, 9)
    lon = torch.arange(16, dtype=torch.float32) * 22.5
    field = torch.randn(1, 2, 9, 16)
    u = torch.full((1, 1, 9, 16), 20.0)
    out = semi_lagrangian_advect(field, u, torch.zeros_like(u), lat, lon, 6 * 3600.0)
    assert torch.allclose(out[:, :, 0], field[:, :, 0]) and torch.allclose(out[:, :, -1], field[:, :, -1])
    assert not torch.allclose(out[:, :, 4], field[:, :, 4])  # interior rows still move


def test_boundary_rows_and_humidity_floor():
    cfg = ModelConfig(hidden=8, memory_dim=8, memory_heads=2, memory_layers=1, history=3, dt_hours=6.0, longitude_periodic=False)
    model = FNGFTWeatherModel(cfg)
    model.set_normalization([5.0, 0.0, 290.0, 0.005], [8.0, 6.0, 10.0, 0.004])
    for layer in (model.closure.force, model.closure.scalar_flux):  # make the learned closure large
        torch.nn.init.normal_(layer.weight, std=5.0)
    lat = torch.linspace(5.0, 35.0, 10)
    lon = torch.linspace(65.0, 100.0, 12)
    hist = torch.randn(1, 3, 4, 10, 12)
    pred, _ = model(hist, lat, lon, steps=1)
    last = hist[:, -1]
    # Boundary rows/columns keep the last state (u, v, theta; q is additionally floored at zero).
    p, l = pred[0, 0, :3], last[0, :3]
    assert torch.allclose(p[:, 0], l[:, 0]) and torch.allclose(p[:, -1], l[:, -1])
    assert torch.allclose(p[:, :, 0], l[:, :, 0]) and torch.allclose(p[:, :, -1], l[:, :, -1])
    q_physical = pred[0, 0, 3] * 0.004 + 0.005
    assert q_physical.min() >= -1e-7
