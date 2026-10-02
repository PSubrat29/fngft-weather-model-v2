from __future__ import annotations

import os
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import FNGFTWeatherModel

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/fngft_real.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="FNGFT-AI Real-Data Forecast API", version="0.2.0")
_model: Optional[FNGFTWeatherModel] = None
_lat: Optional[torch.Tensor] = None
_lon: Optional[torch.Tensor] = None


class ForecastRequest(BaseModel):
    history: List[List[List[List[float]]]] = Field(..., description="Standardized state: [time,4,lat,lon]")
    lat: List[float] = Field(..., description="Latitude coordinate")
    lon: List[float] = Field(..., description="Longitude coordinate")
    steps: int = Field(1, ge=1, le=24)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "device": str(DEVICE)}


@app.get("/model-info")
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "history": _model.cfg.history,
        "channels": ["u", "v", "theta", "q"],
        "alpha_range": [_model.cfg.alpha_min, _model.cfg.alpha_max],
        "beta_range": [_model.cfg.beta_min, _model.cfg.beta_max],
        "device": str(DEVICE),
    }


@app.on_event("startup")
def startup() -> None:
    global _model, _lat, _lon
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"MODEL_PATH does not exist: {MODEL_PATH}")
    blob = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    from .config import ModelConfig
    _model = FNGFTWeatherModel(ModelConfig(**blob["model_config"])).to(DEVICE)
    _model.load_state_dict(blob["model_state"])
    _model.eval()
    grid = blob.get("grid") or {}
    _lat = torch.tensor(grid.get("lat", []), dtype=torch.float32, device=DEVICE)
    _lon = torch.tensor(grid.get("lon", []), dtype=torch.float32, device=DEVICE)


@app.post("/forecast")
def forecast(req: ForecastRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    x = torch.tensor(req.history, dtype=torch.float32, device=DEVICE)
    if x.ndim != 4 or x.shape[1] != 4:
        raise HTTPException(status_code=422, detail="history must have shape [time,4,lat,lon]")
    if x.shape[0] != _model.cfg.history:
        raise HTTPException(status_code=422, detail=f"expected {_model.cfg.history} history steps")
    lat = torch.tensor(req.lat, dtype=torch.float32, device=DEVICE)
    lon = torch.tensor(req.lon, dtype=torch.float32, device=DEVICE)
    if x.shape[-2] != lat.numel() or x.shape[-1] != lon.numel():
        raise HTTPException(status_code=422, detail="lat/lon lengths do not match history grid")
    with torch.inference_mode():
        pred, info = _model(x.unsqueeze(0), lat, lon, steps=req.steps)
    return {
        "forecast_standardized": pred[0].cpu().tolist(),
        "alpha_mean": float(info["alpha"].mean().cpu()),
        "beta_mean": float(info["beta"].mean().cpu()),
        "kappa_mean": float(info["kappa"].mean().cpu()),
        "alpha_map": info["alpha"][0, 0].cpu().tolist(),
        "beta_map": info["beta"][0, 0].cpu().tolist(),
        "kappa_map": info["kappa"][0, 0].cpu().tolist(),
    }
