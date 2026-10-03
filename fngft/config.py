from __future__ import annotations

import datetime
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

DATA_FORMATS = {"auto", "netcdf", "zarr", "grib"}
MISSING_VALUE_POLICIES = {"error", "interpolate"}
REGION_KEYS = {"lat_min", "lat_max", "lon_min", "lon_max"}


@dataclass
class DataConfig:
    source: str = ""
    format: str = "auto"  # auto | netcdf | zarr | grib
    engine: Optional[str] = None
    time_dim: str = "time"
    lat_dim: str = "lat"
    lon_dim: str = "lon"
    variables: Dict[str, str] = field(
        default_factory=lambda: {"u": "u", "v": "v", "theta": "theta", "q": "q"}
    )
    level_dim: Optional[str] = None
    level_value: Optional[float] = None
    time_step_hours: Optional[float] = None
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    val_start: Optional[str] = None
    val_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    # None = detect from the grid (periodic only when longitude covers the full 360 degrees).
    longitude_periodic: Optional[bool] = None
    # Optional spatial subset: {lat_min, lat_max, lon_min, lon_max} in degrees.
    region: Optional[Dict[str, float]] = None
    # Integer block-averaging factor applied to lat and lon (1 = keep native resolution).
    coarsen: int = 1
    # error | interpolate (linear in time, then training-period mean for points that are never valid)
    missing_values: str = "error"


@dataclass
class ModelConfig:
    in_channels: int = 4
    hidden: int = 48
    memory_dim: int = 64
    memory_heads: int = 4
    memory_layers: int = 2
    history: int = 8
    alpha_min: float = 0.25
    alpha_max: float = 2.0
    beta_min: float = 0.10
    beta_max: float = 0.99
    basis_orders: int = 8
    operator_sigma: float = 0.85
    # None = inferred from the dataset time step during training and stored in the checkpoint.
    dt_hours: Optional[float] = None
    coriolis_scale: float = 0.0
    stratification_scale: float = 0.10
    diffusion_scale: float = 1e-4
    closure_scale: float = 1.0
    fractional_scale: float = 0.01
    residual_scale: float = 1.0
    # Resolved from the data grid during training and stored in the checkpoint.
    longitude_periodic: bool = True
    # Number of boundary rows (and boundary columns of non-periodic grids) where the learned closure is
    # switched off. Convolutions see padding there, and on real data the closure fed back unstably from
    # these rows in multi-day forecasts. 0 disables.
    closure_boundary_rows: int = 1
    # What the masked boundary rows do each step: "persistence" keeps the last state (a fixed boundary
    # condition), "physics" applies the physics branch only.
    boundary_mode: str = "persistence"
    # Physical lower bound for q applied after every step (0 for specific/relative humidity);
    # null disables it, e.g. when q is a dewpoint in degC.
    q_min: Optional[float] = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    rollout_steps: int = 2
    num_workers: int = 0
    max_train_windows: Optional[int] = None
    max_val_windows: Optional[int] = None
    seed: int = 42
    device: str = "auto"
    checkpoint: str = "artifacts/fngft_real.pt"
    lr_schedule: str = "cosine"  # cosine | constant
    early_stopping_patience: Optional[int] = None


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _build(cls, raw: Optional[dict], section: str):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config section '{section}' must be a mapping, got {type(raw).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in config section '{section}': {unknown}. Allowed keys: {sorted(allowed)}")
    return cls(**raw)


SPLITS = ("train", "val", "test")
REQUIRED = {
    "data": ("source", "format", "time_dim", "lat_dim", "lon_dim", "variables", "coarsen", "missing_values"),
    "model": ("in_channels", "hidden", "memory_dim", "memory_heads", "memory_layers", "history", "basis_orders"),
    "training": ("batch_size", "epochs", "learning_rate", "checkpoint", "rollout_steps", "lr_schedule"),
}


_DATE_ONLY = re.compile(r"^\d{4}(-\d{2}){0,2}$")


def _normalize_time(value, key: str) -> Optional[str]:
    """Return a timezone-naive UTC ISO string (YAML may give str, date or datetime, with or without offset).

    A date without a time of day used as an *end* means the end of that period, so
    ``train_end: 2019-12-31`` includes the whole of 31 December (and ``2019-12`` all of December).
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"data.{key}={value!r} is not a valid date/time")
    if isinstance(value, int):  # YAML reads an unquoted year such as 2019 as an integer
        value = str(value)
    is_date_only = (isinstance(value, datetime.date) and not isinstance(value, datetime.datetime)) or (
        isinstance(value, str) and _DATE_ONLY.match(value.strip()) is not None
    )
    try:
        if is_date_only and key.endswith("_end"):
            ts = pd.Period(str(value).strip()).end_time
        else:
            ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"data.{key}={value!r} is not a valid date/time (use e.g. 2020-01-01 or 2020-01-01T06:00)") from exc
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.isoformat()


def split_ranges(d: "DataConfig") -> Dict[str, tuple]:
    """Configured splits as {name: (start, end)}; a split is configured when its start or end is set."""
    out = {}
    for name in SPLITS:
        start, end = getattr(d, f"{name}_start"), getattr(d, f"{name}_end")
        if start is not None or end is not None:
            out[name] = (start, end)
    return out


def _is_int(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and float(value).is_integer()


def validate_config(config: AppConfig) -> None:
    d, m, t = config.data, config.model, config.training
    for section, keys in REQUIRED.items():
        obj = getattr(config, section)
        for key in keys:
            if getattr(obj, key) is None:
                raise ValueError(f"{section}.{key} must not be null")
    if d.format.lower() not in DATA_FORMATS:
        raise ValueError(f"data.format must be one of {sorted(DATA_FORMATS)}, got '{d.format}'")
    if d.missing_values not in MISSING_VALUE_POLICIES:
        raise ValueError(f"data.missing_values must be one of {sorted(MISSING_VALUE_POLICIES)}, got '{d.missing_values}'")
    if not _is_int(d.coarsen) or int(d.coarsen) < 1:
        raise ValueError(f"data.coarsen must be an integer >= 1, got {d.coarsen!r}")
    d.coarsen = int(d.coarsen)
    if d.region is not None:
        if not isinstance(d.region, dict):
            raise ValueError("data.region must be a mapping like {lat_min: 5, lat_max: 40, lon_min: 65, lon_max: 100}")
        unknown = set(d.region) - REGION_KEYS
        if unknown:
            raise ValueError(f"data.region accepts only {sorted(REGION_KEYS)}, got unknown {sorted(unknown)}")
        for key, value in d.region.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"data.region.{key} must be a number (or null for no bound), got {value!r}")
        d.region = {k: v for k, v in d.region.items() if v is not None} or None
    if d.region is not None:
        if "lat_min" in d.region and "lat_max" in d.region and d.region["lat_min"] >= d.region["lat_max"]:
            raise ValueError("data.region.lat_min must be smaller than lat_max")
        if "lon_min" in d.region and "lon_max" in d.region and d.region["lon_min"] >= d.region["lon_max"]:
            raise ValueError("data.region.lon_min must be smaller than lon_max")
    if d.level_dim and d.level_value is None:
        raise ValueError("data.level_value is required when data.level_dim is set")
    if d.level_value is not None and not d.level_dim:
        raise ValueError("data.level_value is set but data.level_dim is null; set level_dim to the level dimension name")
    if d.level_value is not None and (isinstance(d.level_value, bool) or not isinstance(d.level_value, (int, float))):
        raise ValueError(f"data.level_value must be a number, got {d.level_value!r}")
    for name in SPLITS:
        for side in ("start", "end"):
            key = f"{name}_{side}"
            setattr(d, key, _normalize_time(getattr(d, key), key))
    ranges = split_ranges(d)
    if "train" not in ranges:
        raise ValueError("data.train_start and/or data.train_end must be set (the training period)")
    bounds = {}
    for name, (start, end) in ranges.items():
        lo = pd.Timestamp(start) if start else pd.Timestamp.min
        hi = pd.Timestamp(end) if end else pd.Timestamp.max
        if lo > hi:
            raise ValueError(f"data.{name}_start ({start}) is after data.{name}_end ({end})")
        bounds[name] = (lo, hi)
    names = list(bounds)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if bounds[a][0] <= bounds[b][1] and bounds[b][0] <= bounds[a][1]:
                raise ValueError(
                    f"data split '{a}' {ranges[a]} overlaps '{b}' {ranges[b]}; splits must not share any time "
                    "(otherwise validation/test data leak into training)"
                )
    if m.in_channels != 4:
        raise ValueError("model.in_channels must be 4 (u, v, theta, q)")
    if m.memory_dim % m.memory_heads != 0:
        raise ValueError("model.memory_dim must be divisible by model.memory_heads")
    if m.history < 1:
        raise ValueError("model.history must be >= 1")
    if not (0 < m.alpha_min < m.alpha_max):
        raise ValueError("model.alpha_min/alpha_max must satisfy 0 < alpha_min < alpha_max")
    if not (0 < m.beta_min < m.beta_max):
        raise ValueError("model.beta_min/beta_max must satisfy 0 < beta_min < beta_max")
    if m.closure_boundary_rows is None or not _is_int(m.closure_boundary_rows) or int(m.closure_boundary_rows) < 0:
        raise ValueError(f"model.closure_boundary_rows must be an integer >= 0, got {m.closure_boundary_rows!r}")
    m.closure_boundary_rows = int(m.closure_boundary_rows)
    if m.q_min is not None and (isinstance(m.q_min, bool) or not isinstance(m.q_min, (int, float))):
        raise ValueError(f"model.q_min must be a number or null, got {m.q_min!r}")
    if m.boundary_mode not in {"persistence", "physics"}:
        raise ValueError("model.boundary_mode must be 'persistence' or 'physics'")
    if m.dt_hours is not None and m.dt_hours <= 0:
        raise ValueError("model.dt_hours must be positive (or null to infer it from the data)")
    for key in ("epochs", "batch_size", "rollout_steps"):
        value = getattr(t, key)
        if not _is_int(value) or int(value) < 1:
            raise ValueError(f"training.{key} must be an integer >= 1, got {value!r}")
        setattr(t, key, int(value))
    if t.lr_schedule not in {"cosine", "constant"}:
        raise ValueError("training.lr_schedule must be 'cosine' or 'constant'")


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML configuration into strongly typed dataclasses."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping with data/model/training sections")
    unknown = sorted(set(raw) - {"data", "model", "training"})
    if unknown:
        raise ValueError(f"Unknown top-level config section(s): {unknown}. Allowed: data, model, training")
    config = AppConfig(
        data=_build(DataConfig, raw.get("data"), "data"),
        model=_build(ModelConfig, raw.get("model"), "model"),
        training=_build(TrainingConfig, raw.get("training"), "training"),
    )
    validate_config(config)
    return config


def save_config(config: AppConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
