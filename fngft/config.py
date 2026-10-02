from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml


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
    longitude_periodic: bool = True


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
    dt_hours: float = 1.0
    coriolis_scale: float = 1.0
    stratification_scale: float = 0.10
    diffusion_scale: float = 1e-4
    closure_scale: float = 0.05
    fractional_scale: float = 0.01
    residual_scale: float = 0.02


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


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML configuration into strongly typed dataclasses."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return AppConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )


def save_config(config: AppConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
