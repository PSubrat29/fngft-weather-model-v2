# FNGFT-AI Real-Data Weather Forecast Prototype v0.2

This repository is a deliberately stricter rebuild of the first FNGFT-AI prototype for experimentation with **real gridded weather datasets**.

The intended research architecture is:

```text
real observations / analysis fields
          ↓
validated xarray dataset
          ↓
canonical state X = (u,v,theta,q)
          ↓
training-only normalization
          ↓
history window
          ↓
Memory Transformer → H_t
          ↓
Order Head → alpha, beta, kappa
          ↓
Fractional spatial operator + causal memory
          ↓
learned closure / fluxes
          ↓
explicit real-grid physics proxy
          ↓
autoregressive forecast
```

## What this release changes

The previous prototype was synthetic-first. This version is real-data-first:

- explicit xarray ingestion
- NetCDF / Zarr / optional GRIB support
- variable-name mapping
- strict time/grid validation
- train-only normalization
- temporal windowing
- real-data training/evaluation commands
- newest-file real-time forecast operation
- checkpoint provenance data
- HTTP inference API
- per-module Markdown documentation

## Important scientific boundary

This is still a **research prototype**.

The current physics branch is a reduced rotating/stratified transport proxy. The fractional operator is an FFT approximation that is periodic in both spatial directions. It is therefore not yet a validated global NWP model.

The purpose of this stage is to answer experimentally:

1. Can the architecture ingest real atmospheric fields without data leakage?
2. Can it forecast held-out real data better than suitable baselines?
3. Do alpha/beta/kappa fields remain stable and interpretable across regimes?
4. Does an SGS-supervised version recover a reproducible fractional response?
5. Does the learned closure transfer across spatial resolution and weather regimes?

## Install

```bash
python -m pip install -r requirements.txt
```

Optional GRIB support requires a compatible xarray GRIB backend such as `cfgrib` with ecCodes.

## Configure your data

Edit:

```text
configs/real_data.yaml
```

The critical section is:

```yaml
variables:
  u: your_u_variable
  v: your_v_variable
  theta: your_temperature_variable
  q: your_humidity_variable
```

The current model requires exactly four 2-D fields on a common `time × lat × lon` grid. A vertical level can be selected with `level_dim` and `level_value`.

## Inspect the dataset first

```bash
python -m fngft.cli inspect --config configs/real_data.yaml
```

Do not train until inspection succeeds.

## Train

```bash
python -m fngft.cli train --config configs/real_data.yaml
```

The training job:

1. loads the dataset,
2. validates dimensions and timestamps,
3. fits normalization statistics only on the training interval,
4. builds history/future samples,
5. trains the hybrid model,
6. validates on the configured validation interval,
7. saves the best checkpoint together with model/data/grid metadata.

## Evaluate

```bash
python -m fngft.cli evaluate   --config configs/real_data.yaml   --checkpoint artifacts/fngft_real.pt   --split test
```

## Run the newest-file forecast

Set `data.source` to a directory where new files appear.

Then:

```bash
python -m fngft.cli forecast-latest   --config configs/real_data.yaml   --checkpoint artifacts/fngft_real.pt   --steps 6   --output artifacts/latest_forecast.npz
```

The newest supported file is selected by filesystem modification time.

## Run the HTTP API

```bash
uvicorn fngft.api:app --host 0.0.0.0 --port 8080
```

Set the checkpoint explicitly in deployment:

```bash
MODEL_PATH=artifacts/fngft_real.pt uvicorn fngft.api:app --host 0.0.0.0 --port 8080
```

## Data contract

### Required dimensions

```text
time
lat
lon
```

### Required canonical variables

```text
u
v
theta
q
```

Source datasets can use different names; the configuration maps them.

## Dataset strategy for serious experiments

The recommended progression is:

### Level 1 — real analysis/reanalysis

Train and test forecasting on a clean, time-split atmospheric analysis dataset.

### Level 2 — multi-resolution atmospheric data

Add paired high-resolution and coarse-resolution fields. Compute the unresolved tendency/flux:

```text
SGS = high-resolution truth tendency - coarse resolved tendency
```

Use the SGS target to train and test fractional operator consistency.

### Level 3 — observations + assimilation

Add radar, satellite, radiosonde and other observation streams through a separate data-assimilation layer.

### Level 4 — global model

Replace the periodic FFT operator with a sphere-aware fractional operator and replace the reduced physics proxy with a validated atmospheric dynamical core.

### Level 5 — probabilistic forecast

Add the probabilistic generator only after the deterministic real-data path is stable and properly evaluated.

## Governance

Every trained checkpoint stores:

- model configuration
- variable mapping
- normalization statistics
- grid coordinates
- training configuration

For serious experiments also record dataset version, preprocessing version, source provenance, code revision, random seeds, hardware, and evaluation results externally.

## Tests

```bash
pytest -q
```

The integration test creates a tiny NetCDF dataset and exercises the complete ingestion → preprocessing → model path.

## Module documentation

Every Python module has a same-name Markdown explanation next to it. Read the `.md` file before modifying its corresponding `.py` file.
