# FNGFT-AI Real-Data Weather Forecast Prototype v0.3

A research prototype that learns a **fractional closure** around a reduced **physics proxy** and
forecasts gridded weather fields (u, v, theta, q) from real historical datasets.

```text
real gridded data (NetCDF / Zarr / GRIB, one file or many)
          ↓
validated xarray dataset  →  canonical state X = (u, v, theta, q)   [time, 4, lat, lon]
          ↓
training-period normalization (no leakage)  →  history windows
          ↓
Memory Transformer → H_t → Order Head → alpha, beta, kappa
          ↓
fractional spatial operator + causal power-law memory → learned closure / fluxes
          ↓
physics proxy in physical units (semi-Lagrangian advection, optional Coriolis, diffusion)
          ↓
autoregressive forecast  →  NetCDF / NPZ / HTTP API / web dashboard
```

**New here? Read [`USER_GUIDE.md`](USER_GUIDE.md)** — it lists exactly what you need to do on your
side, step by step. [`AUDIT_REPORT.md`](AUDIT_REPORT.md) lists what was broken in v0.2 and how it
was fixed, including results on real ERA5 data.

## Install

Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For GPU training install the CUDA build of PyTorch first (see https://pytorch.org). GRIB files
additionally need `cfgrib` + ecCodes.

## Check the installation (2 minutes, no data needed)

```bash
pytest -q
python -m fngft demo
```

`demo` creates a synthetic dataset and runs inspect → train → evaluate → forecast. It must end with
`DEMO PASSED`.

## Check on real data (optional, public ERA5 sample)

```bash
pip install gcsfs
python scripts/download_era5_sample.py                 # ~150 MB into data/
python -m fngft inspect  --config configs/era5_sample.yaml
python -m fngft train    --config configs/era5_sample.yaml
python -m fngft evaluate --config configs/era5_sample.yaml --checkpoint artifacts/era5_sample.pt --steps 4
```

## Use your own historical data

1. Copy `configs/real_data.yaml` and edit `data.source`, the dimension names, the four variable
   names and the train/val/test dates.
2. Validate: `python -m fngft inspect --config configs/my_data.yaml`
3. Train: `python -m fngft train --config configs/my_data.yaml`
4. Evaluate against persistence/climatology:
   `python -m fngft evaluate --config configs/my_data.yaml --checkpoint artifacts/fngft_real.pt --split test --steps 4 --output artifacts/eval_test.json`
5. Forecast from the newest file:
   `python -m fngft forecast-latest --config configs/my_data.yaml --checkpoint artifacts/fngft_real.pt --steps 6 --output artifacts/latest_forecast.nc`

`python -m fngft` and the installed `fngft` command are the same CLI (`python -m fngft.cli` also works).

### What the data may look like

| Requirement | Detail |
|---|---|
| Dimensions | time, latitude, longitude (any names; set them in the config) |
| Variables | four fields mapped to `u`, `v` (wind, **m/s**), `theta` (temperature), `q` (humidity) |
| Grid | regular lat/lon; ascending or descending; global or regional |
| Time | fixed step (hourly, 3-hourly, 6-hourly, daily…); missing timestamps are allowed and skipped |
| Levels | multi-level files: set `level_dim` + `level_value` |
| Files | one file, a folder of files, a glob (`data/era5_*.nc`) or a Zarr store |
| Missing values | `missing_values: interpolate` fills NaNs (linear in time) |
| Size | `region:` crops an area, `coarsen:` block-averages the grid |

See [`DATA_CONTRACT.md`](DATA_CONTRACT.md).

## Web dashboard and HTTP API

```bash
python -m fngft serve --checkpoint artifacts/fngft_real.pt --config configs/my_data.yaml --port 8080
```

Open http://127.0.0.1:8080 for the dashboard (model status, training metrics, latest-forecast maps of
u/v/theta/q and alpha/beta/kappa). Interactive API docs are at `/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /` | web dashboard |
| `GET /health` | service and model status (the service stays up when the checkpoint is missing) |
| `GET /model-info` | grid, variables, time step, training history, provenance |
| `POST /forecast` | forecast from a posted history (`units`: `standardized` or `physical`) |
| `GET /forecast-latest?steps=N` | forecast from the newest file in the configured data source |
| `POST /reload` | reload the checkpoint after retraining |

Docker: `docker build -t fngft .` then
`docker run -p 8080:8080 -v $PWD/artifacts:/app/artifacts fngft`.

## Outputs

* `artifacts/<name>.pt` — checkpoint: weights, model config (incl. inferred `dt_hours`), normalization
  statistics, data mapping, units, grid, training history, provenance (time, code revision, versions).
* `artifacts/<name>.history.json` — per-epoch losses next to the persistence baseline.
* `evaluate` JSON — per lead time and variable: RMSE/MAE in physical units, persistence and
  climatology RMSE, skill vs persistence, anomaly correlation; latitude-weighted.
* `forecast-latest` — NetCDF (`.nc`) or NumPy (`.npz`) with u, v, theta, q in physical units plus
  alpha/beta/kappa maps.

## Scientific boundary

This is a **research prototype**, not an operational NWP system. The physics branch is a reduced
transport proxy (no pressure gradient, radiation, moist physics or data assimilation). The fractional
operator is an FFT approximation (latitude mirrored, longitude periodic for global grids). Forecasts
must not replace meteorological review for high-impact decisions — see [`GOVERNANCE.md`](GOVERNANCE.md).

The research questions this stage supports:

1. Can the architecture ingest real atmospheric fields without data leakage?
2. Does it forecast held-out real data better than persistence and climatology?
3. Do alpha/beta/kappa fields remain stable and interpretable across regimes?
4. Does an SGS-supervised version recover a reproducible fractional response?
5. Does the learned closure transfer across resolution and weather regimes?

Recommended progression: real reanalysis → multi-resolution SGS targets → observation streams with
data assimilation → sphere-aware operator + validated dynamical core → probabilistic forecasts.

## Repository layout

```text
fngft/            Python package (every module has a same-name .md explanation)
  static/         web dashboard
configs/          real_data.yaml (template for your data), era5_sample.yaml
scripts/          download_era5_sample.py
tests/            pytest suite (data handling, physics, end-to-end, API)
.github/workflows CI: tests + synthetic demo on Python 3.11-3.14
```
