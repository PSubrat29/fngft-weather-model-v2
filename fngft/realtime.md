# `fngft/realtime.py`

## Purpose

Runs one forecast cycle using the newest weather file in the configured source directory (or the
configured file itself).

## Workflow

```text
source directory → newest NetCDF/Zarr/GRIB file → validate + extract state
      → last N frames (must be contiguous) → checkpoint normalizer
      → FNGFT-AI rollout → inverse normalization → NetCDF / NPZ + summary
```

Checks: the variable mapping equals the checkpoint's, the data step equals the checkpoint's
`dt_hours`, and the history window has no missing timestamps. Forecast valid times use the
checkpoint's `dt_hours`.

## Why it is one-shot

A one-shot function is easier to test and replay than a permanently running watcher. A scheduler,
workflow engine or cloud job invokes it each time a new dataset arrives.

## Output

`--output something.nc` writes NetCDF with variables `u, v, theta, q` (time, lat, lon; physical units,
`units` attributes copied from the training data) and `alpha, beta, kappa` (lat, lon). Any other
suffix writes `.npz` with `forecast[step, channel, lat, lon]`, `channels`, `time`, `lat`, `lon`,
`alpha`, `beta`, `kappa`.

`run_forecast` (forecast from an in-memory physical history) is shared with the HTTP API.
