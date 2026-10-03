# `fngft/realtime.py`

## Purpose

Runs one forecast cycle from the most recent data in the configured source.

## Workflow

```text
data.source (file, folder, glob, Zarr) → combined lazily
      → last history+48 time steps selected (level/region/coarsen applied)
      → trailing steps that miss a variable are dropped
      → last N frames (must be contiguous) → checkpoint normalizer
      → FNGFT-AI rollout → inverse normalization → plausibility check → NetCDF / NPZ + summary
```

Checks: the variable mapping equals the checkpoint's, the data step equals the checkpoint's
`dt_hours`, and the history window has no missing timestamps. Forecast valid times use the
checkpoint's `dt_hours`. Only the most recent time steps are read, so memory use does not grow with
the archive length.

## Plausibility warnings

Every forecast step is checked. Winds above 150 m/s are physically implausible. Any variable more than
half its training range below the training minimum or above the training maximum (both stored in the
checkpoint) is flagged as outside the training data — this catches slow drift such as humidity
creeping upward, without flagging genuine extremes like a cyclone that lies inside that margin.
(Checkpoints from 0.3.0 have no stored range; for them |standardized value| > 10 is used.) The summary returns `warnings` and `first_unphysical_step`; the CLI
prints them to stderr, the NetCDF output stores them in the `warnings` attribute, and the API and
dashboard show them.

## Output

`--output something.nc` writes NetCDF with variables `u, v, theta, q` (time, lat, lon; physical units,
`units` attributes copied from the training data) and `alpha, beta, kappa` (lat, lon). Any other
path writes `.npz` (the suffix is appended when missing and the real path is reported) with
`forecast[step, channel, lat, lon]`, `channels`, `time`, `lat`, `lon`, `alpha`, `beta`, `kappa`.

`run_forecast` (forecast from an in-memory physical history) is shared with the HTTP API.
