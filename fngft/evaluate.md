# `fngft/evaluate.py`

## Purpose

Scores a trained checkpoint on a held-out split without changing it.

```bash
python -m fngft evaluate --config CONFIG --checkpoint CKPT --split test --steps 4 --output report.json
```

## Metrics

All errors are latitude-weighted (cos(lat)), accumulated over every grid point and window.

Per lead time (`steps` autoregressive steps) and per variable:

- `rmse`, `mae` in **physical units** (e.g. m/s, K)
- `rmse_persistence`: error of "tomorrow = today"
- `rmse_climatology`: error of the training-period mean
- `skill_vs_persistence = 1 − rmse / rmse_persistence` (> 0 means better than persistence)
- `acc`: anomaly correlation against the training-period climatology

Overall (lead 1, standardized units): `rmse_standardized`, `rmse_persistence_standardized`,
`mae_standardized`, `anomaly_correlation`, and mean alpha/beta/kappa.

## Interpretation

Forecast metrics answer: *does the model predict held-out weather better than simple baselines?*
Alpha/beta/kappa diagnostics answer: *which fractional regime is the model using?*

For physical-theory validation these must later be combined with SGS consistency, conservation,
calibration, extreme-event skill, long-rollout stability and cross-resolution experiments.

The config's variable mapping must equal the mapping the checkpoint was trained with.
