from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .evaluate import load_checkpoint
from .io import open_weather_dataset
from .preprocess import prepare_dataset


def forecast_latest(config_path: str, checkpoint: str, steps: int = 1, output: str = "") -> dict:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, blob = load_checkpoint(checkpoint, device)
    with open_weather_dataset(cfg.data, latest=True) as ds:
        raw, times, lat, lon = prepare_dataset(ds, cfg.data)
    inferred_dt_hours = float(np.median(np.diff(times.astype("datetime64[ns]").astype(np.int64)) / 3_600_000_000_000.0))
    if not np.isclose(inferred_dt_hours, model.cfg.dt_hours, rtol=0.0, atol=max(1e-6, 0.01 * model.cfg.dt_hours)):
        raise ValueError(f"Dataset time step {inferred_dt_hours:.6g} h does not match checkpoint dt_hours {model.cfg.dt_hours:.6g} h")
    if raw.shape[0] < model.cfg.history:
        raise ValueError(f"Need at least {model.cfg.history} observations; found {raw.shape[0]}")
    history = normalizer.transform(raw[-model.cfg.history:])
    history_t = torch.from_numpy(history).unsqueeze(0).to(device)
    lat_t = torch.from_numpy(lat).to(device)
    lon_t = torch.from_numpy(lon).to(device)
    model.eval()
    with torch.inference_mode():
        pred, info = model(history_t, lat_t, lon_t, steps=steps)
    forecast_std = pred[0].cpu().numpy()
    forecast_physical = normalizer.inverse(forecast_std)
    forecast_times = np.array([times[-1] + np.timedelta64(int(round(cfg.model.dt_hours * 3600 * (i + 1))), "s") for i in range(steps)])
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            forecast=forecast_physical,
            time=forecast_times,
            lat=lat,
            lon=lon,
            alpha=info["alpha"][0].cpu().numpy(),
            beta=info["beta"][0].cpu().numpy(),
            kappa=info["kappa"][0].cpu().numpy(),
        )
    return {
        "analysis_time": str(times[-1]),
        "forecast_time_end": str(forecast_times[-1]),
        "steps": steps,
        "alpha_mean": float(info["alpha"].mean().cpu()),
        "beta_mean": float(info["beta"].mean().cpu()),
        "kappa_mean": float(info["kappa"].mean().cpu()),
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-shot forecast from the newest real-data file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", default="artifacts/latest_forecast.npz")
    args = parser.parse_args()
    print(forecast_latest(args.config, args.checkpoint, args.steps, args.output))


if __name__ == "__main__":
    main()
