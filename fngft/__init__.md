# `fngft/__init__.py`

## Purpose

Package entry point for FNGFT-AI.

## What it does

It exposes the public API and the package version:

- `__version__`
- configuration dataclasses and the YAML loader
- top-level `FNGFTWeatherModel`

```python
from fngft import FNGFTWeatherModel, ModelConfig, load_config, __version__
```

## What it does not do

It does not load datasets, train models, or run forecasts by itself. `python -m fngft` runs the CLI
(`fngft/__main__.py` → `fngft/cli.py`).
