# `fngft/realtime.py`

## Purpose

Runs one forecast cycle using the newest weather file available in the configured source directory.

## Workflow

```text
source directory
      ↓
newest NetCDF/Zarr/GRIB file
      ↓
validate + extract state
      ↓
last N history frames
      ↓
training normalizer from checkpoint
      ↓
FNGFT-AI rollout
      ↓
inverse normalization
      ↓
forecast NPZ + diagnostics
```

## Why it is one-shot

A one-shot function is easier to test and replay than a permanently running watcher. A scheduler, workflow engine or cloud job can invoke it each time a new dataset arrives.

## Output

The optional NPZ file contains:

- forecast state
- forecast times
- lat/lon
- alpha map
- beta map
- kappa map
