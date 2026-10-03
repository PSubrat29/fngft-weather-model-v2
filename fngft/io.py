from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

from .config import DataConfig
from .schema import DatasetProfile, SUPPORTED_SUFFIXES, is_global_longitude, regular_time_step_hours

GRIB_SUFFIXES = {".grib", ".grb", ".grib2", ".grb2"}
# Number of most recent time steps checked by `inspect --latest`.
LATEST_INSPECT_STEPS = 24


def _is_remote(source: str) -> bool:
    return "://" in source


def _is_zarr(path: str) -> bool:
    return path.rstrip("/\\").lower().endswith(".zarr")


def _supported_files(directory: Path) -> List[Path]:
    return sorted(x for x in directory.iterdir() if x.suffix.lower() in SUPPORTED_SUFFIXES and not x.name.startswith("."))


def resolve_sources(source: str) -> List[str]:
    """Expand a file, Zarr store, directory of files, or glob pattern into a list of sources.

    An existing path always wins over glob interpretation, so folder names containing '[' work.
    """
    if not source:
        raise ValueError("data.source is empty. Set it to your dataset file, directory, glob pattern or Zarr store.")
    if _is_remote(source):
        return [source]
    p = Path(source)
    if _is_zarr(source) and p.exists():
        return [source]
    if p.is_dir():
        files = _supported_files(p)
        if not files:
            raise FileNotFoundError(f"No supported weather data files ({sorted(SUPPORTED_SUFFIXES)}) found in {p}")
        return [str(x) for x in files]
    if p.exists():
        return [source]
    if any(ch in source for ch in "*?["):
        head, tail = os.path.split(source)
        pattern = os.path.join(glob.escape(head), tail) if head and Path(head).is_dir() else source
        matches = sorted(m for m in glob.glob(pattern) if Path(m).suffix.lower() in SUPPORTED_SUFFIXES or _is_zarr(m))
        if not matches:
            raise FileNotFoundError(f"No supported files match data.source pattern: {source}")
        return matches
    raise FileNotFoundError(f"data.source does not exist: {source}")


def _detect_format(source: str, configured: str) -> str:
    fmt = configured.lower()
    if fmt != "auto":
        return fmt
    if _is_zarr(source):
        return "zarr"
    if Path(source).suffix.lower() in GRIB_SUFFIXES:
        return "grib"
    return "netcdf"


def open_weather_dataset(config: DataConfig) -> xr.Dataset:
    """Open the configured dataset lazily.

    ``data.source`` may be a single file, a Zarr store (local or remote URL), a directory or a glob
    pattern. Several files are combined into one dataset by their coordinates (time-split archives,
    one-variable-per-file archives, or both). Real-time forecasting takes the most recent time steps
    of this combined dataset, so the file layout and file modification times do not matter.
    """
    sources = resolve_sources(config.source)
    fmt = _detect_format(sources[0], config.format)
    try:
        if fmt == "zarr":
            if len(sources) != 1:
                raise ValueError("Only one Zarr store can be opened at a time")
            return xr.open_zarr(sources[0], consolidated=None)
        engine = config.engine
        if fmt == "grib":
            engine = engine or "cfgrib"
        elif fmt != "netcdf":
            raise ValueError(f"Unsupported data format: {config.format}")
        if len(sources) == 1:
            return xr.open_dataset(sources[0], engine=engine)
        return xr.open_mfdataset(
            sources,
            engine=engine,
            combine="by_coords",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="outer",
        )
    except (ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        if fmt == "grib":
            raise RuntimeError(
                "GRIB loading requires an installed compatible xarray GRIB backend (normally cfgrib + ecCodes)."
            ) from exc
        raise RuntimeError(f"Could not open {sources[0]!r} as {fmt}: {exc}") from exc


def inspect_dataset(config: DataConfig, *, latest: bool = False) -> DatasetProfile:
    """Validate the dataset with every configured selection applied and return its profile.

    The profile describes the data the model will actually see (after level, region and coarsen
    selection). With ``latest=True`` only the most recent time steps are checked, which is what
    ``forecast-latest`` uses. Leading/trailing time steps where a variable is entirely missing (files
    of different variables ending at different times) are skipped, as training and forecasting do,
    and reported as ``incomplete_edge_steps``. Remaining missing values are counted; with
    ``missing_values: error`` any NaN fails.
    """
    from .preprocess import count_missing, incomplete_steps, resolve_longitude_periodic, select_dataset
    from .schema import CANONICAL_CHANNELS

    with open_weather_dataset(config) as ds:
        lookback = LATEST_INSPECT_STEPS + 48 if latest else None
        work, info = select_dataset(ds, config, tail_steps=lookback)
        incomplete = incomplete_steps(work, config)
        complete = np.flatnonzero(~incomplete)
        if complete.size == 0:
            raise ValueError("No time step contains all four variables")
        lo, hi = int(complete[0]), int(complete[-1]) + 1
        if latest:
            lo = max(lo, hi - LATEST_INSPECT_STEPS)
        dropped = int(incomplete[:lo].sum() + incomplete[hi:].sum()) if not latest else int(incomplete[hi:].sum())
        work = work.isel({config.time_dim: slice(lo, hi)})
        times = np.asarray(work[config.time_dim].values)
        lat = np.asarray(work[config.lat_dim].values, dtype=float)
        lon = np.asarray(work[config.lon_dim].values, dtype=float)
        step, gaps = regular_time_step_hours(times)
        missing = count_missing(work, config)
        units = {c: work[config.variables[c]].attrs.get("units") for c in CANONICAL_CHANNELS}
    if missing and config.missing_values == "error":
        raise ValueError(
            f"The selected data contains {missing} non-finite (NaN/inf) values. "
            "Clean the data, or set data.missing_values: interpolate in the config."
        )
    return DatasetProfile(
        source=config.source,
        time_count=int(times.size),
        lat_count=int(lat.size),
        lon_count=int(lon.size),
        time_start=str(times[0]),
        time_end=str(times[-1]),
        time_step_hours=step,
        variables=dict(config.variables),
        lat_min=float(lat.min()),
        lat_max=float(lat.max()),
        lon_min=float(lon.min()),
        lon_max=float(lon.max()),
        lat_step=float(np.median(np.abs(np.diff(lat)))),
        lon_step=float(np.median(np.abs(np.diff(lon)))),
        time_gaps=gaps,
        longitude_global=is_global_longitude(lon),
        units=units,
        selected_level=info["level"],
        longitude_periodic=resolve_longitude_periodic(lon, config),
        missing_value_count=missing,
        incomplete_edge_steps=dropped,
    )
