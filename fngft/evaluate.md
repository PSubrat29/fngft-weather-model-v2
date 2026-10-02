# `fngft/evaluate.py`

## Purpose

Evaluates a trained real-data checkpoint without changing the model.

## Metrics

Current metrics include:

- RMSE in standardized space
- MAE in standardized space
- anomaly correlation
- mean learned alpha
- mean learned beta
- mean learned kappa

## Interpretation

Forecast metrics answer:

`Does the model predict the held-out weather data?`

Alpha/beta/kappa diagnostics answer:

`What fractional regime is the model using while it predicts?`

For physical-theory validation, these must later be combined with SGS consistency, conservation, calibration, extreme-event skill, long-rollout stability and cross-resolution experiments.
