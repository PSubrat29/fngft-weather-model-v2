# Audit report — v0.2 → v0.3.1

Every file of the uploaded v0.2 repository was reviewed and run (round 1, release 0.3.0). The merged
0.3.0 was then verified independently end to end on real data and every reproduced defect was fixed
(round 2, release 0.3.1). This report lists what was broken, what was changed, and how the result was
verified.

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

### Round-1 design comparison on ERA5 (validation, 3 short epochs; persistence MSE 0.080)

| Variant | Validation MSE | vs persistence |
|---|---|---|
| v0.2 settings (Coriolis on, closure 0.05 × 0.02) with stable physics | 1.092 | 14× worse |
| Coriolis on, new closure scales | 0.250 | 3× worse |
| v0.2 closure scales, Coriolis off | 0.073 | 9% better |
| 0.3.0 defaults | 0.054 | 32% better |

## 3. Round 2 — independent verification of 0.3.0

Five independent checks ran against the merged `main`: a clean install from GitHub, the full user
walkthrough on real ERA5, twelve real-world data layouts (Copernicus CDS NetCDF, yearly and
per-variable folders, globs, regional India grid with NaNs, daily data, gaps, Zarr, GRIB, …), finer
and polar ERA5 grids (1.5°, hourly), and an adversarial code review. Every reported failure was then
reproduced by a second, independent check. Installation, tests, the demo and short-range skill were
confirmed; these defects were reproduced and are now fixed:

| Severity | Defect in 0.3.0 | Fix in 0.3.1 |
|---|---|---|
| blocker | multi-day forecasts blew up (4,699 m/s winds by day 10 on the clean ERA5 grid; 9,293 m/s on 1.5° with poles; >150 m/s after 12 h hourly) | closure off on the outermost rows, boundary rows held at the last state, q floored at 0, 2-step training, plausibility warnings |
| major | rows at the poles (every global CDS ERA5 file) scrambled by the advection (cos(lat)≈0) | no zonal displacement on pole rows |
| major | evaluation "climatology" was one global number per variable: climatology RMSE and ACC inflated (0.3.0 report quoted 13.7 K and ACC 0.99 for temperature) | per-grid-point training climatology; true values 5.5 K and 0.95 |
| major | `forecast-latest` / dashboard failed for glob sources and one-variable-per-file folders, ignored data in other files, and chose "latest" by file modification time | real-time mode reads the combined source and uses its last complete time steps |
| major | API silently returned degraded forecasts for descending latitude (ERA5 order) | grid reordered for the model and restored in the response; non-monotonic grids rejected |
| major | `level_value` silently matched the nearest level (850 vs a file in Pa → 500 hPa used) | must match within 1%; chosen level logged and stored |
| major | training log weighted all rows equally while evaluate used cos(lat) (zero at the poles): contradictory verdicts | both use grid-cell area weights (non-zero at the poles) |
| minor | `inspect` ignored level/region/coarsen and NaNs; NaN gap filling crossed split boundaries; open-ended or overlapping splits, null keys, time-zone dates, epochs 0 mishandled; config not read as UTF-8 (Windows); paths with `[`; `expver` message; `.npz` path misreported; ragged API input gave HTTP 500; `engine: h5netcdf` failed (h5py missing); dashboard rounded humidity to 0.01; whole archive loaded for one forecast | all fixed, with regression tests (38 tests) |

### Stability design — measured on four real ERA5 grids

Shared harness: area-weighted 1-step skill, and 10-day (6-hourly, 40 steps) or 4-day (hourly, 96
steps) rollouts from test-period starts. "Unphysical" = wind > 150 m/s or a standardized value beyond
±10.

| Variant (64×32 grids, 6 epochs) | no-pole grid: first unphysical step / max wind | pole grid: first unphysical step / max wind |
|---|---|---|
| pole-row fix + q floor only | step 8 / 272 m/s | step 34 / 184 m/s |
| + closure off on boundary rows | never / 50 m/s | never / 56 m/s |
| + 2-step training | never / 41 m/s | never / 35 m/s |
| **+ boundary rows held at last state (default)** | **never / 42 m/s** | **never / 35 m/s** |

The default also confirmed on 240×121 (1.5°) grids with poles: 6-hourly 10 days max wind 45 m/s,
hourly 4 days max wind 80 m/s, q ≥ 0; boundary-row error equals persistence; the 1.5° 6-hourly model
beats persistence in the training log (0.109 vs 0.178).

## 4. Round 3 — re-verification of the fixes

Three further independent checks re-ran every earlier reproduction against the fixed branch: 44 of 45
were confirmed fixed (the remaining one, `train --epochs 0` from the command line, is now fixed too).
They also found these, all fixed in 0.3.1 with regression tests (45 tests):

| Severity | Defect | Fix |
|---|---|---|
| major (pre-existing since v0.2) | the learned 3×3 convolutions zero-padded at the 0/360° seam: forecasts depended on where the seam lay and a temperature discontinuity grew along the prime meridian (hottest point of 10-day runs at lon 0, up to 335 K) | convolutions wrap around in longitude on global grids; forecasts are identical when the seam is moved |
| minor | humidity drifted upward in long hourly runs without a warning (z-based check too loose) | plausibility check uses each variable's training range stored in the checkpoint (flags values more than half the range outside it) and winds > 150 m/s; genuine extremes such as a cyclone are no longer flagged |
| minor | evaluating a file without the training period failed (climatology computed from the data) | per-grid-point climatology stored in the checkpoint |
| minor | `inspect` rejected per-variable archives whose files end at different times, which training and forecasting accept | leading/trailing incomplete time steps skipped everywhere and reported |
| minor | `level_value` without `level_dim`, null/string region bounds, `coarsen: 2.5`, float/null `closure_boundary_rows`, unquoted year `2019`, glob inside a folder named `[..]`, unweighted `train_mse` next to weighted `val_mse` | validated, handled or made consistent |

## 5. Current results on real ERA5 (`configs/era5_sample.yaml`)

Public ERA5 from the WeatherBench2 archive: 850 hPa u, v, temperature, specific humidity; 6-hourly;
5.625° global grid (32 × 64); 2018–2020. Train 2018–2019, validate Jan–Jun 2020, **test Jul–Dec 2020
(held out)**. Default model, 10 epochs, 2-step training, CPU (~10–12 minutes).

Validation (area-weighted MSE, standardized): 0.045 vs persistence 0.134.

Held-out test period (733 windows; area-weighted RMSE in physical units; climatology = per-grid-point
training mean):

| Lead | u (m/s) model / persistence / climatology | v (m/s) | temperature (K) | q (g/kg) | skill vs persistence | ACC u / v / T / q |
|---|---|---|---|---|---|---|
| +6 h | 1.37 / 2.02 / 5.42 | 1.44 / 2.56 / 5.13 | 0.96 / 1.19 / 5.47 | 0.48 / 0.61 / 2.09 | 19–44% | 0.97 / 0.96 / 0.98 / 0.97 |
| +12 h | 1.96 / 3.23 / 5.42 | 2.10 / 4.10 / 5.13 | 1.42 / 1.91 / 5.47 | 0.69 / 0.96 / 2.09 | 26–49% | 0.93 / 0.91 / 0.97 / 0.94 |
| +18 h | 2.49 / 4.06 / 5.42 | 2.67 / 5.06 / 5.13 | 1.64 / 2.34 / 5.47 | 0.83 / 1.19 / 2.09 | 30–47% | 0.89 / 0.85 / 0.96 / 0.92 |
| +24 h | 2.90 / 4.59 / 5.42 | 3.14 / 5.64 / 5.13 | 1.81 / 2.65 / 5.47 | 0.94 / 1.34 / 2.09 | 30–44% | 0.85 / 0.79 / 0.95 / 0.90 |

10-day rollouts from 8 test-period starts:

| Lead | u RMSE model / persistence / climatology (m/s) | temperature (K) | q (g/kg) | max wind anywhere |
|---|---|---|---|---|
| +48 h | 4.27 / 5.74 / 5.40 | 2.81 / 3.50 / 5.53 | 1.39 / 1.69 / 2.10 | 35 m/s |
| +72 h | 5.21 / 6.15 / 5.40 | 3.61 / 3.78 / 5.54 | 1.66 / 1.75 / 2.09 | 37 m/s |
| +120 h | 6.09 / 6.70 / 5.48 | 4.88 / 4.12 / 5.50 | 2.06 / 1.84 / 2.11 | 33 m/s |
| +240 h | 6.84 / 7.09 / 5.51 | 7.59 / 4.31 / 5.58 | 2.66 / 1.94 / 2.12 | 37 m/s |

(0.3.0 on the same harness: 316 m/s at +120 h, 4,699 m/s and temperatures of −338…686 K at +240 h.)

**Usable lead time:** at this coarse resolution and model size the forecast beats persistence for all
variables up to about 3 days, and for winds up to 10 days; temperature and humidity drift beyond about
3 days (bounded, but worse than persistence and above climatology by day 10). On hourly data
(1.5°, a 2-week training period) the forecast beats persistence up to about 1–1.5 days; humidity drift
in longer hourly runs is flagged by the plausibility check. Forecasts beyond those leads should be
treated as experimental.

Observation for the analysis phase: the learned `alpha` field sits at its lower bound
(`alpha_min = 0.25`) everywhere on this dataset (mean beta 0.61, mean kappa 2.8). Whether that is a
property of coarse 6-hourly data or of the bound itself is one of the questions to examine on your
datasets (e.g. by lowering `alpha_min` and comparing).

These numbers are for a coarse 5.625° grid and a small model; they show the pipeline works on real
data and learns real skill, not operational quality.
