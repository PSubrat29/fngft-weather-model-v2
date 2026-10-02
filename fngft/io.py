from __future__ import annotations

from pathlib import Path
from typing import Optional

import xarray as xr

from .config import DataConfig
from .schema import DatasetProfile, SUPPORTED_SUFFIXES, validate_dataset


def discover_latest_source(path: str | Path) -> str:
    """Return the newest supported file in a directory, or the input path itself."""
    p = Path(path)
    if not p.is_dir():
        return str(p)
    candidates = [x for x in p.iterdir() if x.suffix.lower() in SUPPORTED_SUFFIXES]
    if not candidates:
        raise FileNotFoundError(f"No supported weather data files found in {p}")
    return str(max(candidates, key=lambda item: item.stat().st_mtime))


def open_weather_dataset(config: DataConfig, *, latest: bool = False) -> xr.Dataset:
    source = discover_latest_source(config.source) if latest else config.source
    if not source:
        raise ValueError("data.source is empty")
    fmt = config.format.lower()
    if fmt == "auto":
        suffix = Path(source).suffix.lower()
        if suffix == ".zarr" or Path(source).name.endswith(".zarr"):
            fmt = "zarr"
        elif suffix in {".grib", ".grb", ".grib2", ".grb2"}:
            fmt = "grib"
        else:
            fmt = "netcdf"
    if fmt == "netcdf":
        return xr.open_dataset(source, engine=config.engine)
    if fmt == "zarr":
        return xr.open_zarr(source, consolidated=None)
    if fmt == "grib":
        engine = config.engine or "cfgrib"
        try:
            return xr.open_dataset(source, engine=engine)
        except Exception as exc:
            raise RuntimeError(
                "GRIB loading requires an installed compatible xarray GRIB backend (normally cfgrib + ecCodes)."
            ) from exc
    raise ValueError(f"Unsupported data format: {config.format}")


def inspect_dataset(config: DataConfig, *, latest: bool = False) -> DatasetProfile:
    with open_weather_dataset(config, latest=latest) as ds:
        profile = validate_dataset(
            ds,
            time_dim=config.time_dim,
            lat_dim=config.lat_dim,
            lon_dim=config.lon_dim,
            variables=config.variables,
        )
    return DatasetProfile(**{**profile.__dict__, "source": discover_latest_source(config.source) if latest else config.source})
