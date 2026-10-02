# `fngft/api.py`

## Purpose

Serves a trained checkpoint as an HTTP prediction service.

## Endpoints

### `GET /health`

Returns service/model status.

### `GET /model-info`

Returns model history length, canonical channels and alpha/beta ranges.

### `POST /forecast`

Accepts:

```text
history: [time,4,lat,lon]
lat: [lat]
lon: [lon]
steps: 1..24
```

The history is expected to use the same standardization convention as the training checkpoint.

## Security design

The API does not accept an arbitrary filesystem path from the caller. Dataset ingestion is controlled by deployment configuration instead. This prevents a prediction request from becoming an unrestricted file-access mechanism.

## Production extensions

A production service should add authentication, request limits, structured logging, model/version headers, provenance IDs, and explicit approval gates before exposing forecasts to downstream users.
