from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr

from .config import DataConfig
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
        out = (state - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        ensure_finite(out, "standardized state")
        return out.astype(np.float32)

    def inverse(self, state: np.ndarray) -> np.ndarray:
        return state * self.std[None, :, None, None] + self.mean[None, :, None, None]

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, raw: dict) -> "Standardizer":
        return cls(np.asarray(raw["mean"], dtype=np.float32), np.asarray(raw["std"], dtype=np.float32))


def _split_bounds(config: DataConfig) -> tuple[Optional[str], Optional[str]]:
    starts = [s for s in (config.train_start, config.val_start, config.test_start) if s]
    ends = [e for e in (config.train_end, config.val_end, config.test_end) if e]
    lo = min(starts, key=lambda s: np.datetime64(s, "ns")) if starts else None
    hi = max(ends, key=lambda e: np.datetime64(e, "ns")) if ends else None
    return lo, hi


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


def prepare_dataset(
    ds: xr.Dataset,
    config: DataConfig,
    *,
    crop_to_splits: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and convert an xarray dataset to ``state[time, 4, lat, lon]`` plus coordinates.

    Latitude and longitude are returned in ascending order. With ``crop_to_splits`` only the time span
    covered by the configured train/val/test ranges is loaded (saves memory for long archives).
    """
    validate_dataset(
        ds,
        time_dim=config.time_dim,
        lat_dim=config.lat_dim,
        lon_dim=config.lon_dim,
        variables=config.variables,
        level_dim=config.level_dim,
    )
    work = ds[[config.variables[c] for c in CANONICAL_CHANNELS]]
    if config.level_dim:
        if config.level_value is None:
            raise ValueError("level_value is required when level_dim is set")
        work = work.sel({config.level_dim: config.level_value}, method="nearest")
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
    work = _select_region(work, config)
    work = work.sortby(config.lat_dim).sortby(config.lon_dim)
    factor = int(config.coarsen)
    if factor > 1:
        if work.sizes[config.lat_dim] < 2 * factor or work.sizes[config.lon_dim] < 2 * factor:
            raise ValueError(f"data.coarsen={factor} leaves fewer than 2 grid points")
        work = work.coarsen({config.lat_dim: factor, config.lon_dim: factor}, boundary="trim").mean()

    times = np.asarray(work[config.time_dim].values)
    if config.missing_values == "interpolate":
        train_lo, train_hi = 0, len(times)
        if config.train_start or config.train_end:
            try:
                train_lo, train_hi = time_slice_indices(times, config.train_start, config.train_end)
            except ValueError:
                pass
    arrays = []
    for canonical in CANONICAL_CHANNELS:
        da = work[config.variables[canonical]]
        extra = [d for d in da.dims if d not in (config.time_dim, config.lat_dim, config.lon_dim)]
        if extra:
            da = da.squeeze(extra, drop=True)
        da = da.transpose(config.time_dim, config.lat_dim, config.lon_dim)
        values = np.asarray(da.values, dtype=np.float32)
        if config.missing_values == "interpolate":
            train_part = values[train_lo:train_hi]
            finite = train_part[np.isfinite(train_part)]
            fill = float(finite.mean()) if finite.size else 0.0
            values = _fill_missing(values, fill)
        arrays.append(values)
    state = np.stack(arrays, axis=1)
    ensure_finite(state, "state")
    lat = np.asarray(work[config.lat_dim].values, dtype=np.float32)
    lon = np.asarray(work[config.lon_dim].values, dtype=np.float32)
    return state, times, lat, lon


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
