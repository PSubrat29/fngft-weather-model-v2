from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, ModelConfig, DataConfig
from .io import open_weather_dataset
from .model import FNGFTWeatherModel
from .preprocess import Standardizer, TemporalWindowDataset, prepare_dataset, time_slice_indices


def load_checkpoint(path: str, device: torch.device):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = FNGFTWeatherModel(ModelConfig(**blob["model_config"])).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, Standardizer.from_dict(blob["normalizer"]), blob


def anomaly_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred.reshape(-1).astype(np.float64)
    t = target.reshape(-1).astype(np.float64)
    p = p - p.mean()
    t = t - t.mean()
    denom = np.linalg.norm(p) * np.linalg.norm(t)
    return float(np.dot(p, t) / denom) if denom > 0 else float("nan")


def evaluate(config_path: str, checkpoint: str, split: str = "test") -> dict:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, blob = load_checkpoint(checkpoint, device)
    with open_weather_dataset(cfg.data) as ds:
        raw, times, lat, lon = prepare_dataset(ds, cfg.data)
    state = normalizer.transform(raw)
    starts = {"train": (cfg.data.train_start, cfg.data.train_end), "val": (cfg.data.val_start, cfg.data.val_end), "test": (cfg.data.test_start, cfg.data.test_end)}
    if split not in starts:
        raise ValueError(f"Unknown split: {split}")
    start, end = starts[split]
    if start is None and end is None:
        raise ValueError(f"No {split} time range is configured")
    dataset = TemporalWindowDataset(state, times, history=model.cfg.history, horizon=1, start=start, end=end)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    lat_t = torch.from_numpy(lat).to(device)
    lon_t = torch.from_numpy(lon).to(device)
    rmses, maes, accs, alphas, betas, kappas = [], [], [], [], [], []
    with torch.inference_mode():
        for history, future in loader:
            history = history.to(device)
            pred, info = model(history, lat_t, lon_t, steps=1)
            p = pred[:, 0].cpu().numpy()
            t = future[:, 0].numpy()
            rmses.append(float(np.sqrt(np.mean((p - t) ** 2))))
            maes.append(float(np.mean(np.abs(p - t))))
            accs.append(anomaly_correlation(p, t))
            alphas.append(float(info["alpha"].mean().cpu()))
            betas.append(float(info["beta"].mean().cpu()))
            kappas.append(float(info["kappa"].mean().cpu()))
    result = {
        "split": split,
        "windows": len(dataset),
        "rmse_standardized": float(np.mean(rmses)),
        "mae_standardized": float(np.mean(maes)),
        "anomaly_correlation": float(np.nanmean(accs)),
        "alpha_mean": float(np.mean(alphas)),
        "beta_mean": float(np.mean(betas)),
        "kappa_mean": float(np.mean(kappas)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained FNGFT-AI real-data checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = evaluate(args.config, args.checkpoint, args.split)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
