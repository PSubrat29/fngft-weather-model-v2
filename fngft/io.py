from __future__ import annotations

import glob
from pathlib import Path
from typing import List

import xarray as xr

from .config import DataConfig
from .schema import DatasetProfile, SUPPORTED_SUFFIXES, validate_dataset

GRIB_SUFFIXES = {".grib", ".grb", ".grib2", ".grb2"}


def _is_remote(source: str) -> bool:
    return "://" in source


def _is_zarr(path: str) -> bool:
    return path.rstrip("/\\").lower().endswith(".zarr")


def _supported_files(directory: Path) -> List[Path]:
    return sorted(x for x in directory.iterdir() if x.suffix.lower() in SUPPORTED_SUFFIXES and not x.name.startswith("."))


def discover_latest_source(path: str | Path) -> str:
    """Return the newest supported file in a directory, or the input path itself."""
    p = Path(path)
    if _is_remote(str(path)) or not p.is_dir() or _is_zarr(str(p)):
        return str(path)
    candidates = _supported_files(p)
    if not candidates:
        raise FileNotFoundError(f"No supported weather data files ({sorted(SUPPORTED_SUFFIXES)}) found in {p}")
    return str(max(candidates, key=lambda item: item.stat().st_mtime))


def resolve_sources(source: str) -> List[str]:
    """Expand a file, Zarr store, directory of files, or glob pattern into a list of sources."""
    if not source:
        raise ValueError("data.source is empty. Set it to your dataset file, directory, glob pattern or Zarr store.")
    if _is_remote(source) or _is_zarr(source):
        return [source]
    if any(ch in source for ch in "*?["):
        matches = sorted(glob.glob(source))
        if not matches:
            raise FileNotFoundError(f"No files match data.source pattern: {source}")
        return matches
    p = Path(source)
    if p.is_dir():
        files = _supported_files(p)
        if not files:
            raise FileNotFoundError(f"No supported weather data files ({sorted(SUPPORTED_SUFFIXES)}) found in {p}")
        return [str(x) for x in files]
    if not p.exists():
        raise FileNotFoundError(f"data.source does not exist: {source}")
    return [source]


def _detect_format(source: str, configured: str) -> str:
    fmt = configured.lower()
    if fmt != "auto":
        return fmt
    if _is_zarr(source):
        return "zarr"
    if Path(source).suffix.lower() in GRIB_SUFFIXES:
        return "grib"
    return "netcdf"


def open_weather_dataset(config: DataConfig, *, latest: bool = False) -> xr.Dataset:
    """Open the configured dataset.

    ``data.source`` may be a single file, a Zarr store (local or remote URL), a directory or a glob
    pattern. A directory/glob with several files is opened as one dataset combined along time.
    With ``latest=True`` only the newest file of a directory is opened (real-time mode).
    """
    sources = [discover_latest_source(config.source)] if latest else resolve_sources(config.source)
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
        return xr.open_mfdataset(sources, engine=engine, combine="by_coords", data_vars="minimal", coords="minimal", compat="override")
    except (ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        if fmt == "grib":
            raise RuntimeError(
                "GRIB loading requires an installed compatible xarray GRIB backend (normally cfgrib + ecCodes)."
            ) from exc
        raise RuntimeError(f"Could not open {sources[0]!r} as {fmt}: {exc}") from exc


def inspect_dataset(config: DataConfig, *, latest: bool = False) -> DatasetProfile:
    with open_weather_dataset(config, latest=latest) as ds:
        profile = validate_dataset(
            ds,
            time_dim=config.time_dim,
            lat_dim=config.lat_dim,
            lon_dim=config.lon_dim,
            variables=config.variables,
            level_dim=config.level_dim,
        )
    source = discover_latest_source(config.source) if latest else config.source
    return DatasetProfile(**{**profile.__dict__, "source": source})
