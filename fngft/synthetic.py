"""Synthetic gridded weather data for installation checks and tests.

The fields are smooth, physically shaped (zonal jet, travelling waves, temperature decreasing
poleward, positive humidity) and stored with the same xarray layout as a real NetCDF dataset.
They are not real observations and say nothing about forecast skill on real weather.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

VARIABLES = {"u": "u10", "v": "v10", "theta": "theta2m", "q": "q2m"}


def make_synthetic_dataset(
    path: str | Path,
    *,
    n_times: int = 240,
    n_lat: int = 16,
    n_lon: int = 32,
    dt_hours: float = 6.0,
    start: str = "2026-01-01",
    seed: int = 7,
    noise: float = 0.05,
) -> Path:
    rng = np.random.default_rng(seed)
    time = np.datetime64(start, "ns") + (np.arange(n_times) * dt_hours * 3600 * 1e9).astype("timedelta64[ns]")
    lat = np.linspace(-60.0, 60.0, n_lat)
    lon = np.arange(n_lon) * (360.0 / n_lon)
    t = np.arange(n_times)[:, None, None] * dt_hours / 24.0  # days
    phi = np.deg2rad(lat)[None, :, None]
    lam = np.deg2rad(lon)[None, None, :]
    phase = 2.0 * (lam - 0.35 * t)  # wavenumber-2 pattern drifting east (~20 deg/day)
    phase2 = 3.0 * (lam + 0.15 * t)
    cos = np.cos(phi)
    shape = (n_times, n_lat, n_lon)

    def jitter(scale: float) -> np.ndarray:
        return (noise * scale * rng.standard_normal(shape)).astype(np.float32)

    u = 12.0 * cos**2 + 6.0 * np.sin(phase) * cos + 2.0 * np.cos(phase2) * cos
    v = 5.0 * np.cos(phase) * cos - 1.5 * np.sin(phase2) * cos
    theta = 300.0 - 35.0 * np.sin(phi) ** 2 + 4.0 * np.cos(phase - 0.6) * cos + 3.0 * np.sin(2 * np.pi * t / 365.0)
    q = 0.012 * cos**3 * (1.0 + 0.25 * np.sin(phase + 0.4)) + 0.0005
    data = {
        "u10": (("time", "lat", "lon"), (u + jitter(10.0)).astype(np.float32), {"units": "m s-1"}),
        "v10": (("time", "lat", "lon"), (v + jitter(5.0)).astype(np.float32), {"units": "m s-1"}),
        "theta2m": (("time", "lat", "lon"), (theta + jitter(10.0)).astype(np.float32), {"units": "K"}),
        "q2m": (("time", "lat", "lon"), np.abs(q + jitter(0.005)).astype(np.float32), {"units": "kg kg-1"}),
    }
    ds = xr.Dataset(data, coords={"time": time, "lat": lat, "lon": lon}, attrs={"title": "FNGFT-AI synthetic demo data"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path
