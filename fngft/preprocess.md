# `fngft/preprocess.py`

## Purpose

Converts a validated xarray weather dataset into the tensor representation expected by the model.

## Pipeline

```text
xarray Dataset (lazy)
    ↓ select_dataset:
    ↓   keep only the four mapped variables
    ↓   select the level (data.level_value must match a level within 1%, else a clear error)
    ↓   optional crop to the configured split period (training/evaluation) or last N steps (real time)
    ↓   optional region crop (lon given as -180..180 or 0..360)
    ↓   sort latitude and longitude ascending
    ↓   optional coarsening (block mean, chunked with dask)
    ↓ state_from_selection:
    ↓   squeeze size-1 extra dimensions, transpose to [time, lat, lon]
    ↓   optional missing-value filling
[time, 4, lat, lon] float32
    ↓ standardize with training-period statistics
temporal windows
```

`inspect` runs the same `select_dataset`, so a passing inspect means training sees valid data.

## Standardization

For each variable `z = (x - mean_train) / std_train`. Statistics come only from the configured
training period. The statistics are saved in the checkpoint and inside the model, which uses them to
run its physics branch in physical units.

## Missing values

`missing_values: error` stops with a clear message. `interpolate` fills gaps linearly in time per grid
point, separately inside each split (so validation/test values never fill training gaps); points that
are never valid (for example a land mask) get the training-period mean.

## Weights

`area_weights(lat)` returns grid-cell area weights (non-zero on pole rows), used by evaluation and by
the training log.

## `TemporalWindowDataset`

If `history=8` and `horizon=3`, each sample contains:

```text
history: [t-7 ... t]
future:  [t+1, t+2, t+3]
```

Windows that cross a missing timestamp are skipped. `max_windows` subsamples evenly across the split
(not just the first windows).

## Split-window rule

A split boundary applies to the forecast target, while the history may reach backward into earlier
data. The model may know the recent past when predicting the first target inside a validation/test
interval, but never future target data.
