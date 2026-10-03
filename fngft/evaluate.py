from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .io import open_weather_dataset
from .model import FNGFTWeatherModel, build_model
from .preprocess import Standardizer, TemporalWindowDataset, area_weights, prepare_dataset, time_slice_indices
from .schema import CANONICAL_CHANNELS
from .train import resolve_device


def load_checkpoint(path: str, device: torch.device) -> tuple[FNGFTWeatherModel, Standardizer, dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train a model first (python -m fngft.cli train ...).")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = build_model(blob["model_config"], blob["normalizer"]).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, Standardizer.from_dict(blob["normalizer"]), blob


def check_variable_mapping(blob: dict, variables: dict) -> None:
    trained = (blob.get("data_config") or {}).get("variables")
    if trained and dict(trained) != dict(variables):
        raise ValueError(
            f"Config variable mapping {dict(variables)} differs from the mapping the checkpoint was trained with {dict(trained)}"
        )


def anomaly_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Centred pattern correlation (kept for backward compatibility)."""
    p = pred.reshape(-1).astype(np.float64)
    t = target.reshape(-1).astype(np.float64)
    p = p - p.mean()
    t = t - t.mean()
    denom = np.linalg.norm(p) * np.linalg.norm(t)
    return float(np.dot(p, t) / denom) if denom > 0 else float("nan")


def latitude_weights(lat: np.ndarray) -> np.ndarray:
    """Grid-cell area weights (kept under the old name for compatibility)."""
    return area_weights(lat)


def evaluate(
    config_path: str,
    checkpoint: str,
    split: str = "test",
    *,
    steps: int = 1,
    batch_size: int = 8,
    device: str = "auto",
    max_windows: Optional[int] = None,
) -> dict:
    """Score a checkpoint on a held-out split against persistence and climatology baselines.

    Errors are area-weighted (grid-cell area, so pole rows count with their small but non-zero area).
    RMSE/MAE are reported per variable in physical units and overall in standardized units;
    skill_vs_persistence = 1 - RMSE_model / RMSE_persistence (> 0 is better). Climatology is the
    training-period mean of every grid point; rmse_climatology is the error of forecasting that map and
    acc is the anomaly correlation against it.
    """
    cfg = load_config(config_path)
    dev = resolve_device(device)
    model, normalizer, blob = load_checkpoint(checkpoint, dev)
    check_variable_mapping(blob, cfg.data.variables)
    with open_weather_dataset(cfg.data) as ds:
        raw, times, lat, lon = prepare_dataset(ds, cfg.data, crop_to_splits=True)
    state = normalizer.transform(raw)
    del raw
    tr_lo, tr_hi = time_slice_indices(times, cfg.data.train_start, cfg.data.train_end)
    clim = torch.from_numpy(state[tr_lo:tr_hi].mean(axis=0, dtype=np.float64)).unsqueeze(0)  # [1,4,lat,lon]
    ranges = {
        "train": (cfg.data.train_start, cfg.data.train_end),
        "val": (cfg.data.val_start, cfg.data.val_end),
        "test": (cfg.data.test_start, cfg.data.test_end),
    }
    if split not in ranges:
        raise ValueError(f"Unknown split: {split}")
    start, end = ranges[split]
    if start is None and end is None:
        raise ValueError(f"No {split} time range is configured")
    dataset = TemporalWindowDataset(
        state, times, history=model.cfg.history, horizon=steps, start=start, end=end, max_windows=max_windows
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    lat_t = torch.from_numpy(lat).to(dev)
    lon_t = torch.from_numpy(lon).to(dev)
    w = torch.from_numpy(latitude_weights(lat)).to(torch.float64).view(1, 1, -1, 1)
    std = torch.from_numpy(normalizer.std.astype(np.float64)).view(1, -1, 1, 1)

    n_cells = 0.0
    sums = {k: torch.zeros(steps, 4, dtype=torch.float64) for k in ("se", "ae", "se_pers", "se_clim", "pt", "pp", "tt")}
    orders = {"alpha": [], "beta": [], "kappa": []}
    with torch.inference_mode():
        for history, future in loader:
            pred, info = model(history.to(dev), lat_t, lon_t, steps=steps)
            pred = pred.cpu().to(torch.float64)
            future = future.to(torch.float64)
            last = history[:, -1].to(torch.float64)
            for lead in range(steps):
                p, t = pred[:, lead], future[:, lead]
                pa, ta = p - clim, t - clim
                sums["se"][lead] += ((p - t) ** 2 * w).sum(dim=(0, 2, 3))
                sums["ae"][lead] += ((p - t).abs() * w).sum(dim=(0, 2, 3))
                sums["se_pers"][lead] += ((last - t) ** 2 * w).sum(dim=(0, 2, 3))
                sums["se_clim"][lead] += (ta**2 * w).sum(dim=(0, 2, 3))
                sums["pt"][lead] += (pa * ta * w).sum(dim=(0, 2, 3))
                sums["pp"][lead] += (pa * pa * w).sum(dim=(0, 2, 3))
                sums["tt"][lead] += (ta * ta * w).sum(dim=(0, 2, 3))
            n_cells += history.shape[0] * lat.size * lon.size
            for key in orders:
                orders[key].append(float(info[key].mean().cpu()))

    std_1d = std.view(-1)
    per_lead = []
    for lead in range(steps):
        mse = sums["se"][lead] / n_cells
        variables = {}
        for c, name in enumerate(CANONICAL_CHANNELS):
            rmse = float(mse[c].sqrt() * std_1d[c])
            rmse_p = float((sums["se_pers"][lead][c] / n_cells).sqrt() * std_1d[c])
            rmse_c = float((sums["se_clim"][lead][c] / n_cells).sqrt() * std_1d[c])
            denom = float((sums["pp"][lead][c] * sums["tt"][lead][c]).sqrt())
            variables[name] = {
                "rmse": rmse,
                "mae": float(sums["ae"][lead][c] / n_cells * std_1d[c]),
                "rmse_persistence": rmse_p,
                "rmse_climatology": rmse_c,
                "skill_vs_persistence": 1.0 - rmse / rmse_p if rmse_p > 0 else float("nan"),
                "acc": float(sums["pt"][lead][c]) / denom if denom > 0 else float("nan"),
            }
        per_lead.append(
            {
                "lead_hours": float((lead + 1) * model.cfg.dt_hours),
                "rmse_standardized": float(mse.mean().sqrt()),
                "rmse_persistence_standardized": float((sums["se_pers"][lead] / n_cells).mean().sqrt()),
                "variables": variables,
            }
        )
    first = per_lead[0]
    pt, pp, tt = (sums[k][0].sum() for k in ("pt", "pp", "tt"))
    return {
        "split": split,
        "windows": len(dataset),
        "steps": steps,
        "dt_hours": model.cfg.dt_hours,
        "units": {name: (blob.get("data_units") or {}).get(name) for name in CANONICAL_CHANNELS},
        "rmse_standardized": first["rmse_standardized"],
        "rmse_persistence_standardized": first["rmse_persistence_standardized"],
        "mae_standardized": float((sums["ae"][0] / n_cells).mean()),
        "anomaly_correlation": float(pt / (pp * tt).sqrt()) if float(pp * tt) > 0 else float("nan"),
        "alpha_mean": float(np.mean(orders["alpha"])),
        "beta_mean": float(np.mean(orders["beta"])),
        "kappa_mean": float(np.mean(orders["kappa"])),
        "leads": per_lead,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained FNGFT-AI real-data checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--steps", type=int, default=1, help="Number of autoregressive lead times to score")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = evaluate(
        args.config,
        args.checkpoint,
        args.split,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        max_windows=args.max_windows,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
