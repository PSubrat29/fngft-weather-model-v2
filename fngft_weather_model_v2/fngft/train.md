# `fngft/train.py`

## Purpose

Trains the model against a real gridded dataset.

## Exact workflow

1. Load YAML configuration.
2. Fix random seeds.
3. Open xarray dataset.
4. Validate and extract `[time,4,lat,lon]`.
5. Compute normalization statistics from the training period only.
6. Create temporal history/future windows.
7. Build `FNGFTWeatherModel`.
8. Run autoregressive rollout for the configured horizon.
9. Calculate composite real-data loss.
10. Backpropagate and clip gradients.
11. Evaluate on validation data, if configured.
12. Save the best checkpoint.

## Checkpoint contents

Every checkpoint stores:

- model weights
- exact model configuration
- normalization statistics
- data mapping
- grid coordinates
- training configuration

This eliminates ambiguity between code defaults and the model that was actually trained.

## Leakage control

The normalizer is fitted only on the training interval. The validation and test intervals are kept out of normalization statistics.
