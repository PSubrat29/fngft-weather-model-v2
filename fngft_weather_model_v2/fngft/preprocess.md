# `fngft/preprocess.py`

## Purpose

Converts a validated xarray weather dataset into the tensor representation expected by the model.

## Pipeline

```text
xarray Dataset
    ↓
optional vertical level selection
    ↓
transpose to [time, lat, lon]
    ↓
stack canonical channels
    ↓
[time, 4, lat, lon]
    ↓
standardize using training-period statistics
    ↓
temporal windows
```

## Standardization

For each variable:

`z = (x - mean_train) / std_train`

Statistics are calculated only from the configured training period. Validation and test periods never influence those statistics.

## `TemporalWindowDataset`

If `history=8` and `horizon=3`, each sample contains:

```text
history: [t-7 ... t]
future:  [t+1, t+2, t+3]
```

This prevents random row shuffling from destroying temporal order and supports multi-step rollout training.

## Split-window rule

A split boundary applies to the forecast target, while the history may reach backward into earlier data. This is intentional: the model is allowed to know the recent past when predicting the first target inside a validation/test interval, but it is not allowed to use future target data.
