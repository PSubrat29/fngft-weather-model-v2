# `fngft/api.py`

## Purpose

Serves a trained checkpoint as an HTTP prediction service with a web dashboard.

```bash
python -m fngft serve --checkpoint artifacts/fngft_real.pt --config configs/my_data.yaml --port 8080
# or
MODEL_PATH=artifacts/fngft_real.pt FNGFT_CONFIG=configs/my_data.yaml uvicorn fngft.api:app --port 8080
```

| Variable | Meaning |
|---|---|
| `MODEL_PATH` | checkpoint to serve (default `artifacts/fngft_real.pt`) |
| `FNGFT_CONFIG` | optional config; enables `GET /forecast-latest` |

## Endpoints

- `GET /` — dashboard (`fngft/static/index.html`): service/model status, training metrics vs
  persistence, and latest-forecast maps with hover values.
- `GET /health` — `status` is `ok` or `degraded`; `load_error` explains a failed checkpoint load.
  The service starts even without a checkpoint (v0.2 crashed at startup).
- `GET /model-info` — history, `dt_hours`, grid, variables, units, normalizer, training history,
  provenance.
- `POST /forecast` — body `{history: [time,4,lat,lon], lat?, lon?, steps: 1..24, units}`;
  `units` is `standardized` (default, training z-scores) or `physical`. `lat`/`lon` default to the
  training grid. Returns standardized and physical forecasts plus alpha/beta/kappa maps.
- `GET /forecast-latest?steps=N&max_size=M` — forecast from the newest file of the deployment's data
  source; fields are subsampled to at most `M` points per axis for display.
- `POST /reload` — reload `MODEL_PATH` after retraining.

## Security design

No endpoint accepts a filesystem path. Dataset location and checkpoint come from deployment
configuration only.

## Production extensions

Add authentication, request limits, structured logging, model/version headers, provenance IDs and
explicit approval gates before exposing forecasts to downstream users.
