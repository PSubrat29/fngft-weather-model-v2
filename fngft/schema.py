from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import xarray as xr

CANONICAL_CHANNELS = ("u", "v", "theta", "q")
SUPPORTED_SUFFIXES = {".nc", ".nc4", ".grib", ".grb", ".grib2", ".grb2", ".zarr"}
NS_PER_HOUR = 3_600_000_000_000.0


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
    lat_step: float = 0.0
    lon_step: float = 0.0
    time_gaps: int = 0
    longitude_global: bool = False
    units: Dict[str, Optional[str]] = field(default_factory=dict)


def require_channels(mapping: Dict[str, str]) -> None:
    missing = [name for name in CANONICAL_CHANNELS if name not in mapping]
    if missing:
        raise ValueError(f"Missing canonical variables: {missing}")
    if len(mapping) != 4:
        raise ValueError("The current real-data prototype requires exactly four channels: u, v, theta, q")


def time_values_ns(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times)
    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError(
            f"Time coordinate must decode to datetime64 (CF 'units' attribute such as 'hours since ...'); got dtype {times.dtype}"
        )
    return times.astype("datetime64[ns]").astype(np.int64)


def regular_time_step_hours(times: np.ndarray) -> tuple[float, int]:
    """Return (step_hours, number_of_gaps).

    The step is the most common spacing. Gaps are allowed only when they are whole multiples of
    the step (missing timestamps); windows that cross a gap are skipped later.
    """
    diffs = np.diff(time_values_ns(times)) / NS_PER_HOUR
    if diffs.size == 0:
        raise ValueError("At least two time samples are required")
    if not np.all(diffs > 0):
        raise ValueError("Time coordinate must be strictly increasing (sort the dataset by time first)")
    step = float(np.median(diffs))
    ratio = diffs / step
    off_grid = np.abs(ratio - np.round(ratio)) > 0.05
    if np.any(off_grid) or np.any(np.round(ratio) < 1):
        raise ValueError(
            f"Time coordinate is not regular: median step is {step:g} h but {int(np.count_nonzero(off_grid))} "
            "intervals are not whole multiples of it. Resample the data to a fixed step first."
        )
    gaps = int(np.count_nonzero(np.round(ratio) > 1))
    return step, gaps


def is_global_longitude(lon: np.ndarray) -> bool:
    lon = np.asarray(lon, dtype=float)
    if lon.size < 2:
        return False
    step = float(np.median(np.abs(np.diff(lon))))
    span = float(lon.max() - lon.min()) + step
    return abs(span - 360.0) <= 0.5 * step


def _check_regular_axis(values: np.ndarray, name: str) -> float:
    d = np.diff(values)
    if np.any(np.abs(d) < 1e-9) or not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError(f"{name} coordinate must be strictly monotonic")
    step = float(np.median(np.abs(d)))
    if np.max(np.abs(np.abs(d) - step)) > max(1e-7, 0.05 * step):
        raise ValueError(f"{name} grid must be approximately regular (Gaussian/irregular grids must be regridded first)")
    return step


def validate_dataset(
    ds: xr.Dataset,
    *,
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
    variables: Dict[str, str],
    level_dim: Optional[str] = None,
) -> DatasetProfile:
    require_channels(variables)
    for dim in (time_dim, lat_dim, lon_dim):
        if dim not in ds.dims:
            raise ValueError(
                f"Required dimension '{dim}' not found. Available dimensions: {list(ds.dims)}. "
                "Set data.time_dim / data.lat_dim / data.lon_dim in the config."
            )
    if level_dim and level_dim not in ds.dims:
        raise ValueError(f"data.level_dim '{level_dim}' not found. Available dimensions: {list(ds.dims)}")
    units: Dict[str, Optional[str]] = {}
    for canonical, name in variables.items():
        if name not in ds.data_vars:
            raise ValueError(
                f"Variable '{name}' for '{canonical}' not found in dataset. Available variables: {list(ds.data_vars)}"
            )
        da = ds[name]
        for dim in (time_dim, lat_dim, lon_dim):
            if dim not in da.dims:
                raise ValueError(f"Variable '{name}' must contain dimension '{dim}' (has {list(da.dims)})")
        for dim in da.dims:
            if dim in (time_dim, lat_dim, lon_dim) or dim == level_dim or da.sizes[dim] == 1:
                continue
            raise ValueError(
                f"Variable '{name}' has extra dimension '{dim}' of size {da.sizes[dim]}. "
                "Set data.level_dim and data.level_value to select one level, or reduce that dimension beforehand."
            )
        units[canonical] = da.attrs.get("units")
    times = np.asarray(ds[time_dim].values)
    if times.size < 2:
        raise ValueError("At least two time samples are required")
    step, gaps = regular_time_step_hours(times)
    lat = np.asarray(ds[lat_dim].values, dtype=float)
    lon = np.asarray(ds[lon_dim].values, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("The real-data prototype requires 1-D lat/lon coordinates")
    if lat.size < 2 or lon.size < 2:
        raise ValueError("At least two latitude and longitude points are required")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("Latitude/longitude coordinates contain non-finite values")
    lat_step = _check_regular_axis(lat, "Latitude")
    lon_step = _check_regular_axis(lon, "Longitude")
    return DatasetProfile(
        source="",
        time_count=int(times.size),
        lat_count=int(lat.size),
        lon_count=int(lon.size),
        time_start=str(times[0]),
        time_end=str(times[-1]),
        time_step_hours=step,
        variables=dict(variables),
        lat_min=float(lat.min()),
        lat_max=float(lat.max()),
        lon_min=float(lon.min()),
        lon_max=float(lon.max()),
        lat_step=lat_step,
        lon_step=lon_step,
        time_gaps=gaps,
        longitude_global=is_global_longitude(lon),
        units=units,
    )


def ensure_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        bad = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        raise ValueError(
            f"{name} contains {bad} non-finite (NaN/inf) values. "
            "Clean the data, or set data.missing_values: interpolate in the config."
        )


def validate_state_array(state: np.ndarray, name: str = "state") -> None:
    if state.ndim != 4:
        raise ValueError(f"{name} must have shape [time, channel, lat, lon], got {state.shape}")
    if state.shape[1] != len(CANONICAL_CHANNELS):
        raise ValueError(f"{name} must contain exactly 4 channels")
    ensure_finite(state, name)
