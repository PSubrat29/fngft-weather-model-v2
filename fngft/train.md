# `fngft/train.py`

## Purpose

Trains the model against a real gridded dataset.

## Workflow

1. Fix random seeds.
2. Open the dataset (file, folder, glob or Zarr) and extract `[time,4,lat,lon]` for the period spanned
   by the configured splits.
3. Infer the time step; check it against `data.time_step_hours` / `model.dt_hours` when they are set.
4. Decide longitude periodicity (config value or automatic).
5. Fit normalization statistics on the training period only.
6. Build history/future windows (gaps skipped).
7. Compute the **persistence baseline** on the validation split and print it.
8. Train with AdamW (+ cosine learning-rate schedule), gradient clipping and an autoregressive rollout
   of `rollout_steps`.
9. Validate each epoch, save the best checkpoint, stop early after `early_stopping_patience` epochs
   without improvement. A non-finite loss stops training with a clear message.

Each epoch prints `val_mse` next to `persistence_val_mse`. The model is only useful when `val_mse`
is clearly below the persistence value.

## Checkpoint contents

- model weights (including normalization buffers)
- exact model configuration, including the inferred `dt_hours` and `longitude_periodic`
- normalization statistics, data configuration and variable units
- grid coordinates
- training configuration, per-epoch history and best epoch
- provenance: creation time, package/torch versions, git revision, data period, device

`<checkpoint>.history.json` is written next to the checkpoint.

## Leakage control

The normalizer is fitted only on the training interval. Validation and test intervals never
influence normalization statistics or model selection on the test split.
