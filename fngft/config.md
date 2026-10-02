# `fngft/config.py`

## Purpose

Defines the complete configuration contract and validates YAML files before any work starts.

## Validation

`load_config` rejects unknown keys (typos such as `sourse:` are reported with the list of allowed
keys) and inconsistent values (alpha/beta bounds, `memory_dim` not divisible by `memory_heads`,
`level_dim` without `level_value`, unknown `format`, …).

## `DataConfig`

- `source`: file, directory of files, glob pattern, or Zarr store (local path or URL)
- `format` / `engine`: `auto | netcdf | zarr | grib` and an optional xarray engine
- `time_dim`, `lat_dim`, `lon_dim`: dimension names in the files
- `variables`: mapping from the canonical channels `u, v, theta, q` to file variable names
- `level_dim`, `level_value`: select one vertical level
- `time_step_hours`: optional assertion of the data step (null = inferred)
- `train_* / val_* / test_*`: chronological split boundaries (inclusive)
- `longitude_periodic`: `null` = automatic (periodic only for a full 360° longitude circle)
- `region`: optional `{lat_min, lat_max, lon_min, lon_max}` crop (accepts -180..180 or 0..360)
- `coarsen`: integer block-average factor for lat/lon
- `missing_values`: `error` (default) or `interpolate`

## `ModelConfig`

Network sizes, history length, alpha/beta bounds, fractional basis, physics scales and closure scales.

- `dt_hours: null` means the training job infers the step from the data and stores it in the checkpoint.
- `longitude_periodic` is resolved from the data during training and stored in the checkpoint.
- Defaults `coriolis_scale: 0`, `closure_scale: 1`, `residual_scale: 1` come from real ERA5 tests; see
  `physics.md` and `model.md`.

## `TrainingConfig`

Batch size, epochs, learning rate, weight decay, gradient clipping, rollout length, window limits,
seed, device, checkpoint path, `lr_schedule` (`cosine | constant`) and `early_stopping_patience`.

## Why this file matters

The checkpoint stores the exact model configuration it was trained with, so evaluation, real-time
forecasting and the API rebuild the identical architecture.
