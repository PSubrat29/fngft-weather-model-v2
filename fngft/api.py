"""HTTP inference service and web dashboard.

Environment variables:
    MODEL_PATH    checkpoint to serve (default artifacts/fngft_real.pt)
    FNGFT_CONFIG  optional YAML config; enables GET /forecast-latest (newest file in data.source)
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .evaluate import load_checkpoint
from .model import FNGFTWeatherModel
from .preprocess import Standardizer
from .realtime import forecast_latest, public_summary, run_forecast
from .schema import CANONICAL_CHANNELS

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/fngft_real.pt")
CONFIG_PATH = os.getenv("FNGFT_CONFIG", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class _ServiceState:
    def __init__(self) -> None:
        self.model: Optional[FNGFTWeatherModel] = None
        self.normalizer: Optional[Standardizer] = None
        self.blob: dict = {}
        self.error: Optional[str] = None
        self.lock = threading.Lock()


state = _ServiceState()


def load_model(path: str = "") -> None:
    """Load (or reload) the checkpoint. A failure leaves the service up and reports it on /health."""
    path = path or MODEL_PATH
    with state.lock:
        try:
            model, normalizer, blob = load_checkpoint(path, DEVICE)
        except Exception as exc:  # keep serving /health and the dashboard
            state.model, state.normalizer, state.blob = None, None, {}
            state.error = f"{type(exc).__name__}: {exc}"
            return
        state.model, state.normalizer, state.blob, state.error = model, normalizer, blob, None


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="FNGFT-AI Real-Data Forecast API", version=__version__, lifespan=lifespan)


class ForecastRequest(BaseModel):
    history: List[List[List[List[float]]]] = Field(..., description="State history [time,4,lat,lon], channels u,v,theta,q")
    lat: Optional[List[float]] = Field(
        None, description="Latitude of the history rows, ascending or descending (defaults to the training grid, ascending)"
    )
    lon: Optional[List[float]] = Field(None, description="Longitude of the history columns (defaults to the training grid)")
    steps: int = Field(1, ge=1, le=24)
    units: Literal["standardized", "physical"] = Field(
        "standardized", description="Units of 'history': standardized (training z-scores) or physical (dataset units)"
    )


def _require_model() -> tuple[FNGFTWeatherModel, Standardizer]:
    if state.model is None or state.normalizer is None:
        raise HTTPException(status_code=503, detail=f"Model is not loaded: {state.error or 'unknown error'}")
    return state.model, state.normalizer


def _stride(n: int, max_size: int) -> int:
    return max(1, int(np.ceil(n / max_size)))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status": "ok" if state.model is not None else "degraded",
        "model_loaded": state.model is not None,
        "model_path": MODEL_PATH,
        "load_error": state.error,
        "latest_forecast_enabled": bool(CONFIG_PATH),
        "device": str(DEVICE),
        "version": __version__,
    }


@app.get("/model-info")
def model_info():
    model, normalizer = _require_model()
    grid = state.blob.get("grid") or {}
    data_cfg = state.blob.get("data_config") or {}
    metrics = state.blob.get("metrics") or {}
    return {
        "history": model.cfg.history,
        "dt_hours": model.cfg.dt_hours,
        "channels": list(CANONICAL_CHANNELS),
        "variables": data_cfg.get("variables"),
        "units": state.blob.get("data_units"),
        "grid_shape": [len(grid.get("lat", [])), len(grid.get("lon", []))],
        "lat_range": [min(grid["lat"]), max(grid["lat"])] if grid.get("lat") else None,
        "lon_range": [min(grid["lon"]), max(grid["lon"])] if grid.get("lon") else None,
        "longitude_periodic": model.cfg.longitude_periodic,
        "alpha_range": [model.cfg.alpha_min, model.cfg.alpha_max],
        "beta_range": [model.cfg.beta_min, model.cfg.beta_max],
        "normalizer": normalizer.to_dict(),
        "train_period": [data_cfg.get("train_start"), data_cfg.get("train_end")],
        "best_epoch": metrics.get("best_epoch"),
        "training_history": metrics.get("history"),
        "provenance": state.blob.get("provenance"),
        "device": str(DEVICE),
    }


@app.post("/reload")
def reload_model():
    """Reload the checkpoint at MODEL_PATH (e.g. after retraining)."""
    load_model()
    return health()


def _orientation(coord: np.ndarray, name: str) -> bool:
    """True when ``coord`` is descending; raise 422 unless it is strictly monotonic."""
    d = np.diff(coord)
    if coord.size >= 2 and np.all(d > 0):
        return False
    if coord.size >= 2 and np.all(d < 0):
        return True
    raise HTTPException(status_code=422, detail=f"{name} must be strictly increasing or strictly decreasing")


@app.post("/forecast")
def forecast(req: ForecastRequest):
    model, normalizer = _require_model()
    try:
        x = np.asarray(req.history, dtype=np.float32)
    except ValueError as exc:  # ragged nested lists
        raise HTTPException(status_code=422, detail="history must have shape [time,4,lat,lon]") from exc
    if x.ndim != 4 or x.shape[1] != 4:
        raise HTTPException(status_code=422, detail="history must have shape [time,4,lat,lon]")
    if x.shape[0] != model.cfg.history:
        raise HTTPException(status_code=422, detail=f"expected {model.cfg.history} history steps, got {x.shape[0]}")
    if not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail="history contains non-finite values")
    grid = state.blob.get("grid") or {}
    lat = np.asarray(req.lat if req.lat is not None else grid.get("lat", []), dtype=np.float32)
    lon = np.asarray(req.lon if req.lon is not None else grid.get("lon", []), dtype=np.float32)
    if x.shape[-2] != lat.size or x.shape[-1] != lon.size:
        raise HTTPException(status_code=422, detail="lat/lon lengths do not match history grid")
    if lat.size < 2 or lon.size < 2:
        raise HTTPException(status_code=422, detail="the grid needs at least 2 latitude and 2 longitude points")
    # The model was trained on south-to-north, west-to-east arrays: reorder, forecast, restore.
    flip_lat, flip_lon = _orientation(lat, "lat"), _orientation(lon, "lon")
    if flip_lat:
        x, lat = x[:, :, ::-1, :], lat[::-1]
    if flip_lon:
        x, lon = x[:, :, :, ::-1], lon[::-1]
    physical = x if req.units == "physical" else normalizer.inverse(x)
    try:
        result = run_forecast(
            model, normalizer, np.ascontiguousarray(physical), np.ascontiguousarray(lat), np.ascontiguousarray(lon),
            req.steps, DEVICE, state.blob.get("data_range"),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def restore(a: np.ndarray) -> np.ndarray:
        if flip_lat:
            a = a[..., ::-1, :]
        if flip_lon:
            a = a[..., ::-1]
        return a

    return {
        "channels": list(CANONICAL_CHANNELS),
        "lead_hours": [model.cfg.dt_hours * (i + 1) for i in range(req.steps)],
        "warnings": result["warnings"],
        "first_unphysical_step": result["first_unphysical_step"],
        "forecast_standardized": restore(result["forecast_standardized"]).tolist(),
        "forecast_physical": restore(result["forecast_physical"]).tolist(),
        "alpha_mean": float(result["alpha"].mean()),
        "beta_mean": float(result["beta"].mean()),
        "kappa_mean": float(result["kappa"].mean()),
        "alpha_map": restore(result["alpha"]).tolist(),
        "beta_map": restore(result["beta"]).tolist(),
        "kappa_map": restore(result["kappa"]).tolist(),
    }


@app.get("/forecast-latest")
def forecast_latest_endpoint(
    steps: int = Query(4, ge=1, le=24),
    max_size: int = Query(128, ge=8, le=1024, description="Fields are subsampled to at most this many points per axis"),
):
    """Forecast from the most recent data in the deployment's configured data source (FNGFT_CONFIG)."""
    _require_model()
    if not CONFIG_PATH:
        raise HTTPException(status_code=404, detail="Set the FNGFT_CONFIG environment variable to enable latest-file forecasts")
    try:
        out = forecast_latest(CONFIG_PATH, MODEL_PATH, steps=steps, output="", device=str(DEVICE))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    lat, lon, result = out["_lat"], out["_lon"], out["_result"]
    sy, sx = _stride(lat.size, max_size), _stride(lon.size, max_size)
    fields = {name: result["forecast_physical"][:, c, ::sy, ::sx].round(6).tolist() for c, name in enumerate(CANONICAL_CHANNELS)}
    return {
        **public_summary(out),
        "units": state.blob.get("data_units") or {},
        "times": [str(t) for t in out["_times"]],
        "lat": lat[::sy].tolist(),
        "lon": lon[::sx].tolist(),
        "fields": fields,
        "orders": {key: result[key][::sy, ::sx].tolist() for key in ("alpha", "beta", "kappa")},
    }
