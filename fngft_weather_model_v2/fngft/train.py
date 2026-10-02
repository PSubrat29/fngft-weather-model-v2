from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import sys
import os

# This adds the folder containing this script to Python's search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now you can use a normal absolute import safely
from config import AppConfig, load_config
from io import open_weather_dataset
from losses import composite_real_data_loss
from model import FNGFTWeatherModel
from preprocess import Standardizer, TemporalWindowDataset, build_train_normalizer, prepare_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(config: AppConfig) -> Path:
    set_seed(config.training.seed)
    device = resolve_device(config.training.device)
    with open_weather_dataset(config.data) as ds:
        state_raw, times, lat, lon = prepare_dataset(ds, config.data)

    inferred_dt_hours = float(np.median(np.diff(times.astype("datetime64[ns]").astype(np.int64)) / 3_600_000_000_000.0))
    expected_dt_hours = config.data.time_step_hours or config.model.dt_hours
    if not np.isclose(inferred_dt_hours, expected_dt_hours, rtol=0.0, atol=max(1e-6, 0.01 * expected_dt_hours)):
        raise ValueError(
            f"Dataset time step is {inferred_dt_hours:.6g} h but model time step is {expected_dt_hours:.6g} h. "
            "Set data.time_step_hours only when it intentionally matches the dataset, then set model.dt_hours accordingly."
        )

    normalizer = build_train_normalizer(state_raw, times, config.data)
    state = normalizer.transform(state_raw)
    train_ds = TemporalWindowDataset(
        state,
        times,
        history=config.model.history,
        horizon=config.training.rollout_steps,
        start=config.data.train_start,
        end=config.data.train_end,
        max_windows=config.training.max_train_windows,
    )
    val_start, val_end = config.data.val_start, config.data.val_end
    val_ds = None
    if val_start or val_end:
        val_ds = TemporalWindowDataset(
            state,
            times,
            history=config.model.history,
            horizon=config.training.rollout_steps,
            start=val_start,
            end=val_end,
            max_windows=config.training.max_val_windows,
        )
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers) if val_ds else None

    model = FNGFTWeatherModel(config.model).to(device)
    lat_t = torch.from_numpy(lat).to(device)
    lon_t = torch.from_numpy(lon).to(device)
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    best_val = float("inf")

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running = 0.0
        for history, future in train_loader:
            history = history.to(device)
            future = future.to(device)
            pred, info = model(history, lat_t, lon_t, steps=config.training.rollout_steps)
            per_step = []
            for i in range(pred.shape[1]):
                per_step.append(composite_real_data_loss(pred[:, i], future[:, i], info))
            loss = torch.stack([item["total"] for item in per_step]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()
            running += loss.item()
        train_mean = running / max(len(train_loader), 1)

        val_mean = float("nan")
        if val_loader:
            model.eval()
            vals = []
            with torch.inference_mode():
                for history, future in val_loader:
                    history = history.to(device)
                    future = future.to(device)
                    pred, info = model(history, lat_t, lon_t, steps=config.training.rollout_steps)
                    vals.append(torch.stack([composite_real_data_loss(pred[:, i], future[:, i], info)["total"] for i in range(pred.shape[1])]).mean().item())
            val_mean = float(np.mean(vals))
        print(f"epoch={epoch} train_loss={train_mean:.6f} val_loss={val_mean:.6f}")

        score = val_mean if np.isfinite(val_mean) else train_mean
        if score < best_val:
            best_val = score
            output = Path(config.training.checkpoint)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config.model.__dict__,
                    "normalizer": normalizer.to_dict(),
                    "data_config": config.data.__dict__,
                    "grid": {"lat": lat.tolist(), "lon": lon.tolist()},
                    "training": config.training.__dict__,
                },
                output,
            )
    return Path(config.training.checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FNGFT-AI on a real gridded weather dataset.")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    args = parser.parse_args()
    path = train(load_config(args.config))
    print(f"checkpoint={path}")


if __name__ == "__main__":
    main()
