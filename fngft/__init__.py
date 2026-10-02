"""FNGFT-AI real-data research prototype package."""

from .config import AppConfig, DataConfig, ModelConfig, TrainingConfig, load_config
from .model import FNGFTWeatherModel

__all__ = [
    "AppConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "FNGFTWeatherModel",
    "load_config",
]
