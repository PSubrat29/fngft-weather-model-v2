# `fngft/evaluate.py`

## Purpose

Scores a trained checkpoint on a held-out split without changing it.

```bash
python -m fngft evaluate --config CONFIG --checkpoint CKPT --split test --steps 4 --output report.json
```

## Metrics

All errors are weighted by grid-cell area (rows at the poles keep their small but non-zero area) and
accumulated over every grid point and window. Training logs use the same weights.

Per lead time (`steps` autoregressive steps) and per variable:

- `rmse`, `mae` in **physical units** (e.g. m/s, K)
- `rmse_persistence`: error of "the next state equals the last observed state"
- `rmse_climatology`: error of forecasting the **training-period mean of every grid point**
- `skill_vs_persistence = 1 − rmse / rmse_persistence` (> 0 means better than persistence)
- `acc`: anomaly correlation, with anomalies taken against that per-grid-point climatology

Overall (lead 1, standardized units): `rmse_standardized`, `rmse_persistence_standardized`,
`mae_standardized`, `anomaly_correlation`, and mean alpha/beta/kappa.

The climatology is a time mean over the training period, not a seasonal (day-of-year) climatology,
so for multi-month test periods it is a conservative baseline. It is stored in the checkpoint at
training time, so a held-out file that contains only the test period can be evaluated on its own
(0.3.0 checkpoints compute it from the training period in the data source).

## Interpretation

Forecast metrics answer: *does the model predict held-out weather better than simple baselines?*
Alpha/beta/kappa diagnostics answer: *which fractional regime is the model using?*

For physical-theory validation these must later be combined with SGS consistency, conservation,
calibration, extreme-event skill, long-rollout stability and cross-resolution experiments.

The config's variable mapping must equal the mapping the checkpoint was trained with.
