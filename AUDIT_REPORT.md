# Audit report — v0.2 → v0.3

Every file of the uploaded v0.2 repository was reviewed and run. This report lists what was broken,
what was changed, and how the result was verified.

## 1. Defects found in v0.2

### Blocking (the software could not be used)

| # | Defect | Effect |
|---|---|---|
| 1 | `fngft/train.py` imported `from io import open_weather_dataset` (Python's standard-library `io`) plus a `sys.path` hack | `python -m fngft.cli …` crashed on import: **no command worked** (inspect, train, evaluate, forecast) |
| 2 | Project nested in `fngft_weather_model_v2/` inside the repository | README commands failed from the repository root; `pip install -e .` failed there |
| 3 | API raised an exception at startup when the checkpoint was missing | the web service (and Docker container) could not start until a model existed |
| 4 | No NetCDF4/HDF5 reader in the requirements | most real NetCDF files (NetCDF4 format) could not be opened |
| 5 | Directory sources only worked in "latest" mode; a `.zarr` store was treated as a folder | multi-file historical archives and Zarr real-time mode failed |
| 6 | `model.dt_hours: 1.0` hard default | 3-hourly/6-hourly/daily datasets were rejected unless the user knew to change it |

### Numerical / scientific correctness

| # | Defect | Effect |
|---|---|---|
| 7 | Physics applied to **standardized** values (z-scores) as if they were m/s and K | fields advected with wrong speeds and directions |
| 8 | Forward-Euler + centred-difference advection | unconditionally unstable; blows up in multi-step rollouts on real grids |
| 9 | Coriolis rotation without any pressure-gradient force (no pressure in the state) | balanced real winds rotated ~100° per 6 h; model 3× worse than persistence on ERA5 |
| 10 | Closure scales 0.05 × 0.02 and theta/q flux divergence divided by Earth radius (no time step) | learned correction ≈10⁻³ (winds) and ≈10⁻⁹ (theta, q) per step: the network could not learn. On ERA5 the result was **14× worse than persistence** |
| 11 | Fractional operator used integer wavenumbers | multipliers up to ~10⁶ on 0.25° grids; resolution-dependent |
| 12 | FFT periodic in latitude | artificial jump between north and south boundaries |
| 13 | Transformer memory had no positional encoding | the order of the history was invisible to attention |
| 14 | `longitude_periodic` setting ignored | regional grids treated as wrapping around the globe |
| 15 | `forecast-latest` used the config's `dt_hours`, not the checkpoint's | wrong forecast valid times |
| 16 | Evaluation averaged per-batch RMSE, had no baselines, no physical units | numbers could not show whether the model was useful |

### Robustness and hygiene

17. Extra dimensions (levels, ensemble members) produced cryptic errors; size-1 dims were not squeezed.
18. NaNs stopped everything with no alternative; missing timestamps were rejected outright.
19. Unknown config keys gave a raw Python `TypeError`.
20. A NaN loss silently produced no checkpoint, then evaluation failed with "file not found".
21. Committed build/cache files (`__pycache__`, `*.egg-info`), a temporary `.nc` file and a scratch script; no `.gitignore`.
22. `fngft.__version__` referenced in the docs but missing; no CI; Docker image pulled a multi-GB CUDA PyTorch.
23. The test only covered loading + one untrained forward pass, which is why defect 1 went unnoticed.

## 2. What was changed

* Repository flattened to the root; junk files removed; `.gitignore`, `.dockerignore`.
* Imports fixed; `python -m fngft` entry point and `fngft` console script.
* Data: files, folders, globs and Zarr; levels; size-1 dims squeezed; region crop; coarsening;
  missing-value interpolation; gap-aware windows; time step inferred; clear error messages listing
  available names.
* Physics in physical units; semi-Lagrangian advection (stable at any Courant number); exact
  Coriolis rotation, **off by default** (see `fngft/physics.md`).
* Fractional operator: normalised wavenumbers, latitude mirroring, batched over history.
* Model: positional embedding, zero-initialised residual heads, closure/residual scales 1.0,
  normalization buffers in the checkpoint.
* Training: persistence baseline printed each epoch, cosine LR, early stopping, NaN guard,
  provenance and history in the checkpoint.
* Evaluation: latitude-weighted RMSE/MAE in physical units per variable and lead time, persistence
  and climatology baselines, skill score, anomaly correlation.
* Forecast output as NetCDF or NPZ.
* API: lifespan loading, degraded mode, physical/standardized input, `/forecast-latest`, `/reload`,
  web dashboard at `/`.
* `demo` command, ERA5 sample download script, 22 tests, GitHub Actions CI (Python 3.11–3.14),
  CPU Docker image, rewritten documentation.

## 3. Verification

### Test suite

`pytest -q` → 22 passed (data handling, physics stability, operator bounds, end-to-end train →
evaluate → forecast, HTTP API and dashboard, startup without a checkpoint). `python -m fngft demo`
→ `DEMO PASSED`. The dashboard was rendered in a headless browser in light, dark and phone-width
layouts with no console errors.

### Real data: ERA5 reanalysis

Public ERA5 from the WeatherBench2 archive: 850 hPa u, v, temperature, specific humidity; 6-hourly;
5.625° global grid (32 × 64); 2018–2020. Train 2018–2019, validate Jan–Jun 2020, **test Jul–Dec 2020
(held out)**. Default model (`configs/era5_sample.yaml`), 10 epochs on a CPU (~5 minutes).

Design comparison on the validation split (3 short epochs, 800 training windows; persistence MSE 0.080):

| Variant | Validation MSE | vs persistence |
|---|---|---|
| v0.2 settings (Coriolis on, closure 0.05 × 0.02) with stable physics | 1.092 | 14× worse |
| Coriolis on, new closure scales | 0.250 | 3× worse |
| v0.2 closure scales, Coriolis off | 0.073 | 9% better |
| **v0.3 defaults** | **0.054** | **32% better** |

Full training, best epoch 9: validation MSE 0.034 vs persistence 0.080.

Held-out test period (733 windows, latitude-weighted RMSE in physical units):

| Lead | u (m/s) model / persistence | v (m/s) | temperature (K) | q (g/kg) | skill vs persistence |
|---|---|---|---|---|---|
| +6 h | 1.40 / 2.03 | 1.46 / 2.56 | 0.96 / 1.19 | 0.47 / 0.61 | 19–43% |
| +12 h | 2.08 / 3.23 | 2.21 / 4.10 | 1.44 / 1.91 | 0.69 / 0.96 | 25–46% |
| +18 h | 2.69 / 4.06 | 2.87 / 5.06 | 1.69 / 2.34 | 0.85 / 1.19 | 28–43% |
| +24 h | 3.15 / 4.60 | 3.39 / 5.64 | 1.89 / 2.65 | 0.97 / 1.34 | 28–40% |

The model beats persistence for every variable at every lead time; climatology RMSE is far larger
(u 7.8 m/s, temperature 13.7 K). Anomaly correlation at +24 h: u 0.92, v 0.78, temperature 0.99,
q 0.97.

Observation for the analysis phase: the learned `alpha` field sits at its lower bound
(`alpha_min = 0.25`) everywhere on this dataset (mean beta 0.90, mean kappa 3.0). Whether that is a
property of coarse 6-hourly data or of the bound itself is one of the questions to examine on your
datasets (e.g. by lowering `alpha_min` and comparing).

These numbers are for a coarse 5.625° grid and a small model; they show the pipeline works on real
data and learns real skill, not operational quality.
