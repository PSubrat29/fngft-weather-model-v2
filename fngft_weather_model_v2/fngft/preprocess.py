from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from .config import DataConfig
from .schema import CANONICAL_CHANNELS, ensure_finite, validate_dataset


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


def prepare_dataset(ds: xr.Dataset, config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validate_dataset(
        ds,
        time_dim=config.time_dim,
        lat_dim=config.lat_dim,
        lon_dim=config.lon_dim,
        variables=config.variables,
    )
    # Normalize coordinate orientation before numerical gradients.
    work = ds.sortby(config.lat_dim).sortby(config.lon_dim)
    if config.level_dim and config.level_dim in work.dims:
        if config.level_value is None:
            raise ValueError("level_value is required when level_dim is set")
        work = work.sel({config.level_dim: config.level_value}, method="nearest")
    arrays = []
    for canonical in CANONICAL_CHANNELS:
        da = work[config.variables[canonical]].transpose(config.time_dim, config.lat_dim, config.lon_dim)
        arrays.append(np.asarray(da.load().values, dtype=np.float32))
    state = np.stack(arrays, axis=1)
    ensure_finite(state, "state")
    times = np.asarray(work[config.time_dim].values)
    lat = np.asarray(work[config.lat_dim].values, dtype=np.float32)
    lon = np.asarray(work[config.lon_dim].values, dtype=np.float32)
    return state, times, lat, lon


def time_slice_indices(times: np.ndarray, start: Optional[str], end: Optional[str]) -> tuple[int, int]:
    time_ns = times.astype("datetime64[ns]").astype(np.int64)
    lo = 0 if start is None else int(np.searchsorted(time_ns, np.datetime64(start, "ns").astype(np.int64), side="left"))
    hi = len(times) if end is None else int(np.searchsorted(time_ns, np.datetime64(end, "ns").astype(np.int64), side="right"))
    if hi <= lo:
        raise ValueError(f"Empty time selection for start={start}, end={end}")
    return lo, hi


class TemporalWindowDataset(Dataset):
    """Returns standardized historical states and future targets."""

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
        #   s + history >= lo          (first forecast target is inside split)
        #   s + history + horizon - 1 < hi  (last forecast target is inside split)
        #   s + history + horizon <= len(times)
        # Note: hi is an exclusive upper bound from searchsorted(..., side="right").
        min_start = max(0, lo - history)
        max_start = min(len(times) - history - horizon, hi - history - horizon)
        starts = list(range(min_start, max_start + 1))
        if max_windows is not None:
            starts = starts[: max(0, int(max_windows))]
        if not starts:
            raise ValueError("No temporal windows available for the requested split")
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[index]
        h = self.state[s : s + self.history]
        f = self.state[s + self.history : s + self.history + self.horizon]
        return torch.from_numpy(h), torch.from_numpy(f)


def build_train_normalizer(
    state: np.ndarray,
    times: np.ndarray,
    config: DataConfig,
) -> Standardizer:
    lo, hi = time_slice_indices(times, config.train_start, config.train_end)
    return Standardizer.fit(state[lo:hi])
