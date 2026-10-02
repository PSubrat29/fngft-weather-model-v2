# `fngft/config.py`

## Purpose

Defines the complete configuration contract for the real-data prototype.

## Three configuration groups

### `DataConfig`

Describes the incoming dataset:

- source path or object-store URI
- format (`netcdf`, `zarr`, `grib`)
- xarray dimension names
- mapping from source variable names to canonical channels
- optional vertical-level selection
- train/validation/test time ranges
- expected periodic longitude behavior

The canonical state is exactly:

`u, v, theta, q`

### `ModelConfig`

Defines the neural and numerical architecture:

- history length
- hidden dimensions
- Transformer size
- alpha/beta bounds
- fractional basis count
- physics and closure scales
- model time step

### `TrainingConfig`

Defines the optimization process:

- batch size
- epochs
- learning rate
- gradient clipping
- rollout length
- checkpoint path

## Why this file matters

The old prototype allowed architecture settings to drift between the source code and the checkpoint. This version stores the model configuration inside the checkpoint and loads it back exactly.
