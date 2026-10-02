# `fngft/__init__.py`

## Purpose

Package entry point for FNGFT-AI.

## What it does

It exposes the public API:

- configuration dataclasses
- YAML configuration loader
- top-level `FNGFTWeatherModel`

## What it does not do

It does not load datasets, train models, or run forecasts by itself.

## Why it exists

It provides a stable import surface such as:

```python
from fngft import FNGFTWeatherModel, ModelConfig
```
