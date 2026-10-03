"""FNGFT-AI real-data research prototype package."""

__version__ = "0.3.1"

from .config import AppConfig, DataConfig, ModelConfig, TrainingConfig, load_config
from .model import FNGFTWeatherModel

__all__ = [
    "__version__",
    "AppConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "FNGFTWeatherModel",
    "load_config",
]
