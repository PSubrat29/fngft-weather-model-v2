# Test suite

Run with `pytest -q` (about 10 seconds on a CPU).

| File | What it checks |
|---|---|
| `test_pipeline.py` | NetCDF → validation → standardization → windows → model rollout; alpha/beta bounds |
| `test_data.py` | descending latitude, level selection, extra dimensions, region/coarsen, missing values, time gaps, irregular time, multi-file folders, config typos, shipped config templates |
| `test_physics.py` | semi-Lagrangian advection (identity, exact one-cell shift), 200-step stability with a 50 m/s jet, fractional operator bounds, regional (non-periodic) grids |
| `test_end_to_end.py` | train → checkpoint contents → evaluate with baselines → forecast-latest (.nc/.npz) → HTTP API and dashboard, including startup without a checkpoint |

`conftest.py` trains one tiny checkpoint on synthetic data shared by the end-to-end tests.

## What the tests do not prove

Passing tests do not establish meteorological skill or validate FNGFT as a physical theory. They show
that data handling, numerics, training, evaluation, forecasting and serving work and are consistent.
Forecast skill must be measured with `evaluate` on real held-out data.
