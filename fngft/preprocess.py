from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr

from .config import DataConfig, split_ranges
from .schema import CANONICAL_CHANNELS, NS_PER_HOUR, ensure_finite, is_global_longitude, time_values_ns, validate_dataset


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, state: np.ndarray) -> "Standardizer":
        if state.ndim != 4 or state.shape[1] != 4:
            raise ValueError(f"Expected [time,4,lat,lon], got {state.shape}")
        mean = state.mean(axis=(0, 2, 3), dtype=np.float64)
        std = state.std(axis=(0, 2, 3), dtype=np.float64)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, state: np.ndarray) -> np.ndarray:
        out = np.array(state, dtype=np.float32, copy=True)
        out -= self.mean[None, :, None, None]
        out /= self.std[None, :, None, None]
        ensure_finite(out, "standardized state")
        return out

    def inverse(self, state: np.ndarray) -> np.ndarray:
        return state * self.std[None, :, None, None] + self.mean[None, :, None, None]

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, raw: dict) -> "Standardizer":
        return cls(np.asarray(raw["mean"], dtype=np.float32), np.asarray(raw["std"], dtype=np.float32))


def _split_bounds(config: DataConfig) -> tuple[Optional[str], Optional[str]]:
    """Overall time span of the configured splits; an open side (null start/end) stays unbounded."""
    ranges = list(split_ranges(config).values())
    if not ranges:
        return None, None
    lo = None if any(s is None for s, _ in ranges) else min((s for s, _ in ranges), key=lambda v: np.datetime64(v, "ns"))
    hi = None if any(e is None for _, e in ranges) else max((e for _, e in ranges), key=lambda v: np.datetime64(v, "ns"))
    return lo, hi


def split_boundaries(times: np.ndarray, config: DataConfig) -> list[int]:
    """Index positions where a configured split starts or ends (used to keep gap filling inside a split)."""
    cuts = {0, len(times)}
    for start, end in split_ranges(config).values():
        try:
            lo, hi = time_slice_indices(times, start, end)
        except ValueError:
            continue
        cuts.update((lo, hi))
    return sorted(cuts)


def choose_level(ds: xr.Dataset, config: DataConfig) -> float:
    """Return the level coordinate value matching data.level_value (within 1%), or raise a clear error."""
    levels = np.asarray(ds[config.level_dim].values, dtype=float).reshape(-1)
    target = float(config.level_value)
    chosen = float(levels[np.argmin(np.abs(levels - target))])
    if abs(chosen - target) > max(1e-6, 0.01 * abs(target)):
        units = ds[config.level_dim].attrs.get("units", "unknown")
        raise ValueError(
            f"data.level_value={config.level_value:g} is not one of the {config.level_dim} values {levels.tolist()} "
            f"(units: {units}); the nearest is {chosen:g}. Check the units: pressure in Pa needs 85000 for 850 hPa."
        )
    return chosen


def _select_region(work: xr.Dataset, config: DataConfig) -> xr.Dataset:
    region = config.region or {}
    if not region:
        return work
    lat_dim, lon_dim = config.lat_dim, config.lon_dim
    lon = np.asarray(work[lon_dim].values, dtype=float)
    lon_min = region.get("lon_min")
    if lon_min is not None and lon_min < 0 and lon.max() > 180:
        # Region given in -180..180 while the data uses 0..360.
        work = work.assign_coords({lon_dim: ((work[lon_dim] + 180.0) % 360.0) - 180.0}).sortby(lon_dim)
    elif region.get("lon_max") is not None and region["lon_max"] > 180 and lon.min() < 0:
        work = work.assign_coords({lon_dim: work[lon_dim] % 360.0}).sortby(lon_dim)
    work = work.sortby(lat_dim)
    lat_sel = slice(region.get("lat_min"), region.get("lat_max"))
    lon_sel = slice(region.get("lon_min"), region.get("lon_max"))
    work = work.sel({lat_dim: lat_sel, lon_dim: lon_sel})
    if work.sizes[lat_dim] < 2 or work.sizes[lon_dim] < 2:
        raise ValueError(f"data.region {region} selects fewer than 2 latitude or longitude points")
    return work


def _fill_missing(array: np.ndarray, fill_value: float) -> np.ndarray:
    """Linear interpolation along time for each grid point, then a constant for never-valid points."""
    if np.isfinite(array).all():
        return array
    t = array.shape[0]
    frame = pd.DataFrame(array.reshape(t, -1)).replace([np.inf, -np.inf], np.nan)
    frame = frame.interpolate(method="linear", axis=0, limit_direction="both")
    out = np.array(frame.to_numpy(dtype=np.float32), copy=True).reshape(array.shape)
    out[~np.isfinite(out)] = fill_value
    return out


def select_dataset(
    ds: xr.Dataset,
    config: DataConfig,
    *,
    crop_to_splits: bool = False,
    tail_steps: Optional[int] = None,
) -> tuple[xr.Dataset, dict]:
    """Validate and apply all configured selections lazily (no data is loaded).

    Steps: keep the four mapped variables, select the level, optionally crop to the configured split
    period or keep only the last ``tail_steps`` time steps, crop the region, sort latitude/longitude
    ascending and coarsen. Returns the selected dataset and a dict with the chosen level.
    """
    validate_dataset(
        ds,
        time_dim=config.time_dim,
        lat_dim=config.lat_dim,
        lon_dim=config.lon_dim,
        variables=config.variables,
        level_dim=config.level_dim,
    )
    info: dict = {"level": None}
    work = ds[[config.variables[c] for c in CANONICAL_CHANNELS]]
    if config.level_dim:
        if config.level_value is None:
            raise ValueError("level_value is required when level_dim is set")
        info["level"] = choose_level(work, config)
        work = work.sel({config.level_dim: info["level"]})
        if config.level_dim in work.coords:
            work = work.drop_vars(config.level_dim)
    if crop_to_splits:
        lo, hi = _split_bounds(config)
        if lo is not None or hi is not None:
            work = work.sel({config.time_dim: slice(lo, hi)})
            if work.sizes[config.time_dim] < 2:
                raise ValueError(
                    f"The configured train/val/test ranges ({lo} .. {hi}) select fewer than 2 time steps. "
                    f"Dataset covers {ds[config.time_dim].values[0]} .. {ds[config.time_dim].values[-1]}."
                )
    if tail_steps is not None:
        work = work.isel({config.time_dim: slice(-int(tail_steps), None)})
    work = _select_region(work, config)
    work = work.sortby(config.lat_dim).sortby(config.lon_dim)
    factor = int(config.coarsen)
    if factor > 1:
        if work.sizes[config.lat_dim] < 2 * factor or work.sizes[config.lon_dim] < 2 * factor:
            raise ValueError(f"data.coarsen={factor} leaves fewer than 2 grid points")
        try:  # with dask, block-average chunk by chunk instead of loading the full-resolution field
            work = work.chunk({config.time_dim: 64})
        except Exception:
            pass
        work = work.coarsen({config.lat_dim: factor, config.lon_dim: factor}, boundary="trim").mean()
    return work, info


def _channel_array(work: xr.Dataset, config: DataConfig, canonical: str) -> xr.DataArray:
    da = work[config.variables[canonical]]
    extra = [d for d in da.dims if d not in (config.time_dim, config.lat_dim, config.lon_dim)]
    if extra:
        da = da.squeeze(extra, drop=True)
    return da.transpose(config.time_dim, config.lat_dim, config.lon_dim)


def _complete_span(incomplete: np.ndarray) -> tuple[int, int]:
    """[lo, hi) of time steps after dropping leading/trailing steps where a variable is entirely missing."""
    complete = np.flatnonzero(~incomplete)
    if complete.size == 0:
        raise ValueError("No time step contains all four variables")
    return int(complete[0]), int(complete[-1]) + 1


def state_from_selection(work: xr.Dataset, config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset returned by ``select_dataset`` into ``state[time, 4, lat, lon]``.

    Leading/trailing time steps in which a variable is entirely missing are dropped (files of different
    variables that start or end at different times); remaining NaNs are filled if configured.
    """
    times = np.asarray(work[config.time_dim].values)
    lat = np.asarray(work[config.lat_dim].values, dtype=np.float32)
    lon = np.asarray(work[config.lon_dim].values, dtype=np.float32)
    state = np.empty((len(times), len(CANONICAL_CHANNELS), lat.size, lon.size), dtype=np.float32)
    for c, canonical in enumerate(CANONICAL_CHANNELS):
        state[:, c] = _channel_array(work, config, canonical).values
    lo, hi = _complete_span(np.isnan(state).all(axis=(2, 3)).any(axis=1))
    if (lo, hi) != (0, len(times)):
        state, times = state[lo:hi], times[lo:hi]
    if config.missing_values == "interpolate" and not np.isfinite(state).all():
        cuts = split_boundaries(times, config)
        train_lo, train_hi = 0, len(times)
        try:
            train_lo, train_hi = time_slice_indices(times, config.train_start, config.train_end)
        except ValueError:
            pass
        for c in range(len(CANONICAL_CHANNELS)):
            if np.isfinite(state[:, c]).all():
                continue
            train_part = state[train_lo:train_hi, c]
            finite = train_part[np.isfinite(train_part)]
            fill = float(finite.mean()) if finite.size else 0.0
            # Fill inside each split separately so no split borrows values from another.
            for a, b in zip(cuts[:-1], cuts[1:]):
                state[a:b, c] = _fill_missing(state[a:b, c], fill)
    ensure_finite(state, "state")
    return state, times, lat, lon


def prepare_dataset(
    ds: xr.Dataset,
    config: DataConfig,
    *,
    crop_to_splits: bool = False,
    tail_steps: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and convert an xarray dataset to ``state[time, 4, lat, lon]`` plus coordinates.

    Latitude and longitude are returned in ascending order. With ``crop_to_splits`` only the time span
    covered by the configured train/val/test ranges is loaded (saves memory for long archives); with
    ``tail_steps`` only the most recent time steps are loaded (real-time forecasting).
    """
    work, _ = select_dataset(ds, config, crop_to_splits=crop_to_splits, tail_steps=tail_steps)
    return state_from_selection(work, config)


def incomplete_steps(work: xr.Dataset, config: DataConfig) -> np.ndarray:
    """Boolean per time step: True where at least one variable is entirely missing (reads the data)."""
    incomplete = np.zeros(work.sizes[config.time_dim], dtype=bool)
    for canonical in CANONICAL_CHANNELS:
        da = _channel_array(work, config, canonical)
        incomplete |= np.asarray(da.isnull().all(dim=[config.lat_dim, config.lon_dim]).values)
    return incomplete


def count_missing(work: xr.Dataset, config: DataConfig) -> int:
    """Number of NaN/inf values in the selected variables (reads the data)."""
    total = 0
    for canonical in CANONICAL_CHANNELS:
        da = _channel_array(work, config, canonical)
        total += int((~np.isfinite(da)).sum().values)
    return total


def area_weights(lat: np.ndarray) -> np.ndarray:
    """Grid-cell area weights per latitude row, normalised to mean 1 (non-zero on pole rows)."""
    lat = np.asarray(lat, dtype=np.float64)
    if lat.size < 2:
        return np.ones_like(lat)
    half = 0.5 * float(np.median(np.abs(np.diff(lat))))
    upper = np.deg2rad(np.clip(lat + half, -90.0, 90.0))
    lower = np.deg2rad(np.clip(lat - half, -90.0, 90.0))
    w = np.abs(np.sin(upper) - np.sin(lower))
    return w / w.mean() if w.mean() > 0 else np.ones_like(lat)


def resolve_longitude_periodic(lon: np.ndarray, config: DataConfig) -> bool:
    """Explicit config value wins; otherwise periodic only when the grid spans 360 degrees."""
    if config.longitude_periodic is not None:
        return bool(config.longitude_periodic)
    return is_global_longitude(lon)


def infer_time_step_hours(times: np.ndarray) -> float:
    diffs = np.diff(time_values_ns(times)) / NS_PER_HOUR
    return float(np.median(diffs))


def time_slice_indices(times: np.ndarray, start: Optional[str], end: Optional[str]) -> tuple[int, int]:
    time_ns = time_values_ns(times)
    lo = 0 if start is None else int(np.searchsorted(time_ns, np.datetime64(start, "ns").astype(np.int64), side="left"))
    hi = len(times) if end is None else int(np.searchsorted(time_ns, np.datetime64(end, "ns").astype(np.int64), side="right"))
    if hi <= lo:
        raise ValueError(
            f"Empty time selection for start={start}, end={end}. "
            f"Dataset covers {times[0]} .. {times[-1]}; adjust the train/val/test dates in the config."
        )
    return lo, hi


class TemporalWindowDataset(Dataset):
    """Returns standardized historical states and future targets.

    Windows that cross a missing timestamp (gap in the time axis) are skipped.
    """

    def __init__(
        self,
        state: np.ndarray,
        times: np.ndarray,
        *,
        history: int,
        horizon: int,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_windows: Optional[int] = None,
    ) -> None:
        if history < 1 or horizon < 1:
            raise ValueError("history and horizon must both be >= 1")
        self.state = state
        self.times = times
        self.history = history
        self.horizon = horizon
        lo, hi = time_slice_indices(times, start, end)
        # Window start s must satisfy:
        #   s + history >= lo               (first forecast target is inside split)
        #   s + history + horizon - 1 < hi  (last forecast target is inside split)
        #   s + history + horizon <= len(times)
        # Note: hi is an exclusive upper bound from searchsorted(..., side="right").
        span = history + horizon
        min_start = max(0, lo - history)
        max_start = min(len(times) - span, hi - span)
        starts = np.arange(min_start, max_start + 1, dtype=np.int64)
        if starts.size and len(times) > 1:
            time_ns = time_values_ns(times)
            step = np.median(np.diff(time_ns))
            elapsed = time_ns[starts + span - 1] - time_ns[starts]
            starts = starts[np.abs(elapsed - (span - 1) * step) <= 0.05 * step]
        if max_windows is not None and starts.size > int(max_windows):
            pick = np.unique(np.round(np.linspace(0, starts.size - 1, max(1, int(max_windows)))).astype(np.int64))
            starts = starts[pick]
        if starts.size == 0:
            raise ValueError(
                f"No temporal windows available for the split {start} .. {end}: need {span} consecutive time steps "
                f"(history={history} + horizon={horizon}). Dataset covers {times[0]} .. {times[-1]}."
            )
        self.starts = [int(s) for s in starts]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[index]
        h = self.state[s : s + self.history]
        f = self.state[s + self.history : s + self.history + self.horizon]
        return torch.from_numpy(np.ascontiguousarray(h)), torch.from_numpy(np.ascontiguousarray(f))


def build_train_normalizer(
    state: np.ndarray,
    times: np.ndarray,
    config: DataConfig,
) -> Standardizer:
    lo, hi = time_slice_indices(times, config.train_start, config.train_end)
    return Standardizer.fit(state[lo:hi])
