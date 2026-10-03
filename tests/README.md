# Test suite

Run with `pytest -q` (38 tests, about 10 seconds on a CPU).

| File | What it checks |
|---|---|
| `test_pipeline.py` | NetCDF → validation → standardization → windows → model rollout; alpha/beta bounds |
| `test_data.py` | descending latitude, level selection and level-value check (hPa vs Pa), extra dimensions and `expver`, region/coarsen, missing values (no leakage across splits), time gaps, irregular time, multi-file, glob and per-variable sources, paths with `[`, inspect of the selected data, open-ended and overlapping splits, time-zone dates, UTF-8 configs, config typos, shipped config templates |
| `test_physics.py` | semi-Lagrangian advection (identity, exact one-cell shift, pole rows), 200-step stability with a 50 m/s jet, fractional operator bounds, regional (non-periodic) grids, boundary rows and humidity floor |
| `test_end_to_end.py` | train → checkpoint contents → evaluate with baselines (per-grid-point climatology) → forecast-latest (.nc/.npz, glob and per-variable sources) → 40-step rollout plausibility → HTTP API and dashboard (descending latitude, bad input, startup without a checkpoint) |

`conftest.py` trains one tiny checkpoint on synthetic data shared by the end-to-end tests.

## What the tests do not prove

Passing tests do not establish meteorological skill or validate FNGFT as a physical theory. They show
that data handling, numerics, training, evaluation, forecasting and serving work and are consistent.
Forecast skill must be measured with `evaluate` on real held-out data.
