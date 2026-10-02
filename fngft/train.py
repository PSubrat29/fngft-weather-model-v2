from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from . import __version__
from .config import AppConfig, load_config
from .io import open_weather_dataset
from .losses import composite_real_data_loss
from .model import FNGFTWeatherModel
from .preprocess import (
    TemporalWindowDataset,
    build_train_normalizer,
    infer_time_step_hours,
    prepare_dataset,
    resolve_longitude_periodic,
)
from .schema import CANONICAL_CHANNELS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def resolve_dt_hours(config: AppConfig, inferred: float) -> float:
    """Check the dataset time step against the config and return the model step in hours."""
    tol = max(1e-6, 0.01 * inferred)
    if config.data.time_step_hours is not None and not math.isclose(config.data.time_step_hours, inferred, abs_tol=tol):
        raise ValueError(
            f"Dataset time step is {inferred:g} h but data.time_step_hours is {config.data.time_step_hours:g} h. "
            "Fix data.time_step_hours or set it to null."
        )
    if config.model.dt_hours is None:
        return inferred
    if not math.isclose(config.model.dt_hours, inferred, abs_tol=tol):
        raise ValueError(
            f"Dataset time step is {inferred:g} h but model.dt_hours is {config.model.dt_hours:g} h. "
            "Set model.dt_hours to null (inferred from the data) or to the dataset step."
        )
    return float(config.model.dt_hours)


def _loss_over_rollout(pred: torch.Tensor, future: torch.Tensor, info: dict) -> tuple[torch.Tensor, torch.Tensor]:
    parts = [composite_real_data_loss(pred[:, i], future[:, i], info) for i in range(pred.shape[1])]
    total = torch.stack([p["total"] for p in parts]).mean()
    mse = torch.stack([p["pred"] for p in parts]).mean()
    return total, mse


def train(config: AppConfig, *, log=print) -> Path:
    set_seed(config.training.seed)
    device = resolve_device(config.training.device)
    t0 = time.time()
    with open_weather_dataset(config.data) as ds:
        state_raw, times, lat, lon = prepare_dataset(ds, config.data, crop_to_splits=True)
        units = {c: ds[config.data.variables[c]].attrs.get("units") for c in CANONICAL_CHANNELS}
    log(f"loaded state {tuple(state_raw.shape)} [time,channel,lat,lon] covering {times[0]} .. {times[-1]} in {time.time() - t0:.1f}s")

    dt_hours = resolve_dt_hours(config, infer_time_step_hours(times))
    periodic = resolve_longitude_periodic(lon, config.data)
    model_cfg = replace(config.model, dt_hours=dt_hours, longitude_periodic=periodic)

    normalizer = build_train_normalizer(state_raw, times, config.data)
    state = normalizer.transform(state_raw)
    del state_raw
    train_ds = TemporalWindowDataset(
        state,
        times,
        history=model_cfg.history,
        horizon=config.training.rollout_steps,
        start=config.data.train_start,
        end=config.data.train_end,
        max_windows=config.training.max_train_windows,
    )
    val_ds = None
    if config.data.val_start or config.data.val_end:
        val_ds = TemporalWindowDataset(
            state,
            times,
            history=model_cfg.history,
            horizon=config.training.rollout_steps,
            start=config.data.val_start,
            end=config.data.val_end,
            max_windows=config.training.max_val_windows,
        )
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers, pin_memory=pin
    )
    val_loader = (
        DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers, pin_memory=pin)
        if val_ds
        else None
    )
    log(
        f"dt_hours={dt_hours:g} longitude_periodic={periodic} train_windows={len(train_ds)} "
        f"val_windows={len(val_ds) if val_ds else 0} device={device}"
    )

    model = FNGFTWeatherModel(model_cfg).to(device)
    model.set_normalization(normalizer.mean, normalizer.std)
    lat_t = torch.from_numpy(lat).to(device)
    lon_t = torch.from_numpy(lon).to(device)
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    scheduler = None
    if config.training.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.training.epochs * len(train_loader)))

    persistence_val = float("nan")
    if val_loader:
        errs = []
        for history, future in val_loader:
            last = history[:, -1:].expand_as(future)
            errs.append(torch.mean((last - future) ** 2).item())
        persistence_val = float(np.mean(errs))
        log(f"persistence baseline val_mse={persistence_val:.6f} (standardized units; the model should go below this)")

    output = Path(config.training.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_log = []
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, config.training.epochs + 1):
        e0 = time.time()
        model.train()
        running, running_mse = 0.0, 0.0
        for history, future in train_loader:
            history = history.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            pred, info = model(history, lat_t, lon_t, steps=config.training.rollout_steps)
            loss, mse = _loss_over_rollout(pred, future, info)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}. Lower training.learning_rate, check the input data "
                    "for extreme values, or reduce training.rollout_steps."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()
            running += loss.item()
            running_mse += mse.item()
        train_mean = running / max(len(train_loader), 1)
        train_mse = running_mse / max(len(train_loader), 1)

        val_mean, val_mse = float("nan"), float("nan")
        if val_loader:
            model.eval()
            vals, mses = [], []
            with torch.inference_mode():
                for history, future in val_loader:
                    history = history.to(device, non_blocking=True)
                    future = future.to(device, non_blocking=True)
                    pred, info = model(history, lat_t, lon_t, steps=config.training.rollout_steps)
                    loss, mse = _loss_over_rollout(pred, future, info)
                    vals.append(loss.item())
                    mses.append(mse.item())
            val_mean, val_mse = float(np.mean(vals)), float(np.mean(mses))
        record = {
            "epoch": epoch,
            "train_loss": train_mean,
            "train_mse": train_mse,
            "val_loss": val_mean,
            "val_mse": val_mse,
            "persistence_val_mse": persistence_val,
            "seconds": round(time.time() - e0, 2),
        }
        history_log.append(record)
        log(
            f"epoch={epoch} train_loss={train_mean:.6f} val_loss={val_mean:.6f} "
            f"val_mse={val_mse:.6f} persistence_val_mse={persistence_val:.6f} time={record['seconds']}s"
        )

        score = val_mean if np.isfinite(val_mean) else train_mean
        if score < best_val:
            best_val, best_epoch = score, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model_cfg),
                    "normalizer": normalizer.to_dict(),
                    "data_config": asdict(config.data),
                    "grid": {"lat": lat.tolist(), "lon": lon.tolist()},
                    "data_units": units,
                    "training": asdict(config.training),
                    "metrics": {"best_epoch": epoch, "best_score": score, "history": history_log},
                    "provenance": {
                        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "fngft_version": __version__,
                        "torch_version": torch.__version__,
                        "git_revision": _git_revision(),
                        "data_time_start": str(times[0]),
                        "data_time_end": str(times[-1]),
                        "data_time_count": int(len(times)),
                        "device": str(device),
                    },
                },
                output,
            )
        patience = config.training.early_stopping_patience
        if patience and epoch - best_epoch >= patience:
            log(f"early stopping: no improvement for {patience} epochs")
            break

    output.with_suffix(".history.json").write_text(json.dumps(history_log, indent=2))
    log(f"best epoch={best_epoch} score={best_val:.6f} checkpoint={output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FNGFT-AI on a real gridded weather dataset.")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    args = parser.parse_args()
    path = train(load_config(args.config))
    print(f"checkpoint={path}")


if __name__ == "__main__":
    main()
