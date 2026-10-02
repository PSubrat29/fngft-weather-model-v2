from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, Optional

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


@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    rollout_steps: int = 1
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


def validate_config(config: AppConfig) -> None:
    d, m, t = config.data, config.model, config.training
    if d.format.lower() not in DATA_FORMATS:
        raise ValueError(f"data.format must be one of {sorted(DATA_FORMATS)}, got '{d.format}'")
    if d.missing_values not in MISSING_VALUE_POLICIES:
        raise ValueError(f"data.missing_values must be one of {sorted(MISSING_VALUE_POLICIES)}, got '{d.missing_values}'")
    if int(d.coarsen) < 1:
        raise ValueError("data.coarsen must be an integer >= 1")
    if d.region is not None:
        unknown = set(d.region) - REGION_KEYS
        if unknown:
            raise ValueError(f"data.region accepts only {sorted(REGION_KEYS)}, got unknown {sorted(unknown)}")
        if "lat_min" in d.region and "lat_max" in d.region and d.region["lat_min"] >= d.region["lat_max"]:
            raise ValueError("data.region.lat_min must be smaller than lat_max")
        if "lon_min" in d.region and "lon_max" in d.region and d.region["lon_min"] >= d.region["lon_max"]:
            raise ValueError("data.region.lon_min must be smaller than lon_max")
    if d.level_dim and d.level_value is None:
        raise ValueError("data.level_value is required when data.level_dim is set")
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
    if m.dt_hours is not None and m.dt_hours <= 0:
        raise ValueError("model.dt_hours must be positive (or null to infer it from the data)")
    if t.rollout_steps < 1:
        raise ValueError("training.rollout_steps must be >= 1")
    if t.lr_schedule not in {"cosine", "constant"}:
        raise ValueError("training.lr_schedule must be 'cosine' or 'constant'")


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML configuration into strongly typed dataclasses."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
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
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
