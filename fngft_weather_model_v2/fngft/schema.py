from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import xarray as xr

CANONICAL_CHANNELS = ("u", "v", "theta", "q")
SUPPORTED_SUFFIXES = {".nc", ".nc4", ".grib", ".grb", ".grib2", ".grb2", ".zarr"}


@dataclass(frozen=True)
class DatasetProfile:
    source: str
    time_count: int
    lat_count: int
    lon_count: int
    time_start: str
    time_end: str
    time_step_hours: float
    variables: Dict[str, str]
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def require_channels(mapping: Dict[str, str]) -> None:
    missing = [name for name in CANONICAL_CHANNELS if name not in mapping]
    if missing:
        raise ValueError(f"Missing canonical variables: {missing}")
    if len(mapping) != 4:
        raise ValueError("The current real-data prototype requires exactly four channels: u, v, theta, q")


def validate_dataset(
    ds: xr.Dataset,
    *,
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
    variables: Dict[str, str],
) -> DatasetProfile:
    require_channels(variables)
    for dim in (time_dim, lat_dim, lon_dim):
        if dim not in ds.dims:
            raise ValueError(f"Required dimension '{dim}' not found. Available: {list(ds.dims)}")
    for canonical, name in variables.items():
        if name not in ds.data_vars:
            raise ValueError(f"Variable '{name}' for '{canonical}' not found in dataset")
        da = ds[name]
        for dim in (time_dim, lat_dim, lon_dim):
            if dim not in da.dims:
                raise ValueError(f"Variable '{name}' must contain dimension '{dim}'")
    times = np.asarray(ds[time_dim].values)
    if times.size < 2:
        raise ValueError("At least two time samples are required")
    if not np.all(np.diff(times.astype('datetime64[ns]').astype('int64')) > 0):
        raise ValueError("Time coordinate must be strictly increasing")
    lat = np.asarray(ds[lat_dim].values, dtype=float)
    lon = np.asarray(ds[lon_dim].values, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("The real-data prototype requires 1-D lat/lon coordinates")
    if lat.size < 2 or lon.size < 2:
        raise ValueError("At least two latitude and longitude points are required")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("Latitude/longitude coordinates contain non-finite values")
    lat_d = np.diff(lat)
    lon_d = np.diff(lon)
    lat_step = float(np.median(np.abs(lat_d)))
    lon_step = float(np.median(np.abs(lon_d)))
    if np.any(np.abs(lat_d) < 1e-9) or np.any(np.abs(lon_d) < 1e-9):
        raise ValueError("Latitude/longitude coordinates must be strictly monotonic")
    if np.max(np.abs(np.abs(lat_d) - lat_step)) > max(1e-7, 0.05 * lat_step):
        raise ValueError("Latitude grid must be approximately regular")
    if np.max(np.abs(np.abs(lon_d) - lon_step)) > max(1e-7, 0.05 * lon_step):
        raise ValueError("Longitude grid must be approximately regular")
    diffs = np.diff(times.astype('datetime64[ns]').astype('int64')) / 3_600_000_000_000.0
    step = float(np.median(diffs))
    if np.max(np.abs(diffs - step)) > max(1e-6, 0.05 * step):
        raise ValueError("Time coordinate is not sufficiently regular for fixed-step forecasting")
    return DatasetProfile(
        source="",
        time_count=times.size,
        lat_count=lat.size,
        lon_count=lon.size,
        time_start=str(times[0]),
        time_end=str(times[-1]),
        time_step_hours=step,
        variables=dict(variables),
        lat_min=float(lat.min()),
        lat_max=float(lat.max()),
        lon_min=float(lon.min()),
        lon_max=float(lon.max()),
    )


def ensure_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        bad = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        raise ValueError(f"{name} contains {bad} non-finite values after loading")


def validate_state_array(state: np.ndarray, name: str = "state") -> None:
    if state.ndim != 4:
        raise ValueError(f"{name} must have shape [time, channel, lat, lon], got {state.shape}")
    if state.shape[1] != len(CANONICAL_CHANNELS):
        raise ValueError(f"{name} must contain exactly 4 channels")
    ensure_finite(state, name)
