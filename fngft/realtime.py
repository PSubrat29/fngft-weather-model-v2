from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .config import load_config
from .evaluate import check_variable_mapping, load_checkpoint
from .io import discover_latest_source, open_weather_dataset
from .model import FNGFTWeatherModel
from .preprocess import Standardizer, infer_time_step_hours, prepare_dataset
from .schema import CANONICAL_CHANNELS, NS_PER_HOUR, time_values_ns
from .train import resolve_device


def run_forecast(
    model: FNGFTWeatherModel,
    normalizer: Standardizer,
    history_physical: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    steps: int,
    device: torch.device,
) -> dict:
    """Forecast ``steps`` time steps from a physical-unit history [history, 4, lat, lon]."""
    history_std = normalizer.transform(history_physical.astype(np.float32))
    x = torch.from_numpy(history_std).unsqueeze(0).to(device)
    lat_t = torch.as_tensor(lat, dtype=torch.float32, device=device)
    lon_t = torch.as_tensor(lon, dtype=torch.float32, device=device)
    with torch.inference_mode():
        pred, info = model(x, lat_t, lon_t, steps=steps)
    forecast_std = pred[0].cpu().numpy()
    forecast_physical = normalizer.inverse(forecast_std)
    if not np.isfinite(forecast_physical).all():
        raise RuntimeError("Model produced non-finite forecast values")
    return {
        "forecast_standardized": forecast_std,
        "forecast_physical": forecast_physical.astype(np.float32),
        "alpha": info["alpha"][0, 0].cpu().numpy(),
        "beta": info["beta"][0, 0].cpu().numpy(),
        "kappa": info["kappa"][0, 0].cpu().numpy(),
    }


def write_forecast(path: str, result: dict, forecast_times: np.ndarray, lat: np.ndarray, lon: np.ndarray, units: dict, attrs: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".nc", ".nc4"}:
        data_vars = {
            name: (("time", "lat", "lon"), result["forecast_physical"][:, c], {"units": units.get(name) or ""})
            for c, name in enumerate(CANONICAL_CHANNELS)
        }
        for key in ("alpha", "beta", "kappa"):
            data_vars[key] = (("lat", "lon"), result[key])
        xr.Dataset(data_vars, coords={"time": forecast_times, "lat": lat, "lon": lon}, attrs=attrs).to_netcdf(out)
    else:
        np.savez_compressed(
            out,
            forecast=result["forecast_physical"],
            channels=np.array(CANONICAL_CHANNELS),
            time=forecast_times,
            lat=lat,
            lon=lon,
            alpha=result["alpha"],
            beta=result["beta"],
            kappa=result["kappa"],
        )


def forecast_latest(config_path: str, checkpoint: str, steps: int = 1, output: str = "", device: str = "auto") -> dict:
    """Run one forecast cycle from the newest file in ``data.source`` (or the file itself).

    ``output`` ending in .nc writes NetCDF (variables u, v, theta, q, alpha, beta, kappa); any other
    suffix writes a compressed NumPy .npz archive. Forecast values are in physical units.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    cfg = load_config(config_path)
    dev = resolve_device(device)
    model, normalizer, blob = load_checkpoint(checkpoint, dev)
    check_variable_mapping(blob, cfg.data.variables)
    with open_weather_dataset(cfg.data, latest=True) as ds:
        raw, times, lat, lon = prepare_dataset(ds, cfg.data)
    hist = model.cfg.history
    if raw.shape[0] < hist:
        raise ValueError(f"Need at least {hist} time steps in the latest data; found {raw.shape[0]}")
    if raw.shape[0] >= 2:
        inferred_dt_hours = infer_time_step_hours(times)
        if not math.isclose(inferred_dt_hours, model.cfg.dt_hours, abs_tol=max(1e-6, 0.01 * model.cfg.dt_hours)):
            raise ValueError(f"Dataset time step {inferred_dt_hours:.6g} h does not match checkpoint dt_hours {model.cfg.dt_hours:.6g} h")
    recent = time_values_ns(times[-hist:])
    expected = (hist - 1) * model.cfg.dt_hours * NS_PER_HOUR
    if hist > 1 and abs((recent[-1] - recent[0]) - expected) > 0.05 * model.cfg.dt_hours * NS_PER_HOUR:
        raise ValueError(f"The last {hist} time steps are not contiguous (missing timestamps); cannot build the history window")
    result = run_forecast(model, normalizer, raw[-hist:], lat, lon, steps, dev)
    step = np.timedelta64(int(round(model.cfg.dt_hours * 3600)), "s")
    forecast_times = np.array([times[-1] + step * (i + 1) for i in range(steps)]).astype("datetime64[ns]")
    source = discover_latest_source(cfg.data.source)
    if output:
        write_forecast(
            output,
            result,
            forecast_times,
            lat,
            lon,
            blob.get("data_units") or {},
            {"source": source, "analysis_time": str(times[-1]), "checkpoint": str(checkpoint), "model": "FNGFT-AI"},
        )
    return {
        "source": source,
        "analysis_time": str(times[-1]),
        "forecast_time_end": str(forecast_times[-1]),
        "steps": steps,
        "dt_hours": model.cfg.dt_hours,
        "grid": [int(lat.size), int(lon.size)],
        "alpha_mean": float(result["alpha"].mean()),
        "beta_mean": float(result["beta"].mean()),
        "kappa_mean": float(result["kappa"].mean()),
        "output": output,
        "_result": result,
        "_times": forecast_times,
        "_lat": lat,
        "_lon": lon,
    }


def public_summary(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-shot forecast from the newest real-data file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", default="artifacts/latest_forecast.nc")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(public_summary(forecast_latest(args.config, args.checkpoint, args.steps, args.output, args.device)), indent=2))


if __name__ == "__main__":
    main()
