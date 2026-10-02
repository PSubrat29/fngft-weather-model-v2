# `fngft/preprocess.py`

## Purpose

Converts a validated xarray weather dataset into the tensor representation expected by the model.

## Pipeline

```text
xarray Dataset
    ↓ keep only the four mapped variables
    ↓ optional vertical level selection (level_dim / level_value)
    ↓ optional crop to the configured split period (training/evaluation only)
    ↓ optional region crop (lon given as -180..180 or 0..360)
    ↓ sort latitude and longitude ascending
    ↓ optional coarsening (block mean)
    ↓ squeeze size-1 extra dimensions, transpose to [time, lat, lon]
    ↓ optional missing-value filling
[time, 4, lat, lon] float32
    ↓ standardize with training-period statistics
temporal windows
```

## Standardization

For each variable `z = (x - mean_train) / std_train`. Statistics come only from the configured
training period. The statistics are saved in the checkpoint and inside the model, which uses them to
run its physics branch in physical units.

## Missing values

`missing_values: error` stops with a clear message. `interpolate` fills gaps linearly in time per grid
point; points that are never valid (for example a land mask) get the training-period mean.

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
