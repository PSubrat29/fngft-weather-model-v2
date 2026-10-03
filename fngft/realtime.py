from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xarray as xr

from .config import load_config
from .evaluate import check_variable_mapping, load_checkpoint
from .io import open_weather_dataset
from .model import FNGFTWeatherModel
from .preprocess import Standardizer, infer_time_step_hours, select_dataset, state_from_selection
from .schema import CANONICAL_CHANNELS, NS_PER_HOUR, time_values_ns
from .train import resolve_device

# Forecast steps beyond these limits are flagged: winds no real atmosphere produces, or (for checkpoints
# without a stored training range) standardized values far outside the training data.
PLAUSIBLE_MAX_Z = 10.0
PLAUSIBLE_MAX_WIND = 150.0
# Extra recent time steps read so that trailing incomplete steps can be skipped.
LATEST_LOOKBACK_STEPS = 48


def run_forecast(
    model: FNGFTWeatherModel,
    normalizer: Standardizer,
    history_physical: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    steps: int,
    device: torch.device,
    data_range: Optional[dict] = None,
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
    warnings, first_bad = plausibility_warnings(forecast_std, forecast_physical, data_range)
    return {
        "warnings": warnings,
        "first_unphysical_step": first_bad,
        "forecast_standardized": forecast_std,
        "forecast_physical": forecast_physical.astype(np.float32),
        "alpha": info["alpha"][0, 0].cpu().numpy(),
        "beta": info["beta"][0, 0].cpu().numpy(),
        "kappa": info["kappa"][0, 0].cpu().numpy(),
    }


def plausibility_warnings(
    forecast_std: np.ndarray, forecast_physical: np.ndarray, data_range: Optional[dict] = None
) -> tuple[list, Optional[int]]:
    """Flag forecast steps with physically impossible winds or values far outside the training data.

    With ``data_range`` (per-channel training min/max, stored in checkpoints since 0.3.1) a value is
    flagged when it lies more than half the training range beyond the training min or max; older
    checkpoints fall back to |standardized value| > 10.
    """
    steps = forecast_std.shape[0]
    warnings, flagged = [], []
    wind = np.hypot(forecast_physical[:, 0], forecast_physical[:, 1]).reshape(steps, -1).max(axis=1)
    bad_wind = np.flatnonzero(wind > PLAUSIBLE_MAX_WIND)
    if bad_wind.size:
        k = int(bad_wind[0])
        flagged.append(k)
        warnings.append(f"Physically implausible winds from step {k + 1} (max {wind[k]:.1f} m/s); do not use steps from {k + 1} on.")
    if data_range is not None:
        lo = np.asarray(data_range["min"], dtype=float)
        hi = np.asarray(data_range["max"], dtype=float)
        margin = 0.5 * (hi - lo)
        for c, name in enumerate(CANONICAL_CHANNELS):
            field = forecast_physical[:, c].reshape(steps, -1)
            out = np.flatnonzero((field.min(axis=1) < lo[c] - margin[c]) | (field.max(axis=1) > hi[c] + margin[c]))
            if out.size:
                k = int(out[0])
                flagged.append(k)
                warnings.append(
                    f"{name} leaves the range of the training data (by more than half its span) from step {k + 1} "
                    f"(forecast {field[k].min():.4g}..{field[k].max():.4g}, training {lo[c]:.4g}..{hi[c]:.4g}); treat those steps with caution."
                )
    else:
        zmax = np.abs(forecast_std).reshape(steps, -1).max(axis=1)
        out = np.flatnonzero(zmax > PLAUSIBLE_MAX_Z)
        if out.size:
            k = int(out[0])
            flagged.append(k)
            warnings.append(f"Forecast leaves the range of the training data from step {k + 1} (max |z| {zmax[k]:.1f}); treat those steps with caution.")
    return warnings, (min(flagged) + 1 if flagged else None)


def write_forecast(path: str, result: dict, forecast_times: np.ndarray, lat: np.ndarray, lon: np.ndarray, units: dict, attrs: dict) -> str:
    """Write NetCDF (.nc/.nc4) or NumPy (.npz, appended when missing). Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    attrs = {k: v for k, v in attrs.items() if v is not None}
    if out.suffix.lower() in {".nc", ".nc4"}:
        data_vars = {
            name: (("time", "lat", "lon"), result["forecast_physical"][:, c], {"units": units.get(name) or ""})
            for c, name in enumerate(CANONICAL_CHANNELS)
        }
        for key in ("alpha", "beta", "kappa"):
            data_vars[key] = (("lat", "lon"), result[key])
        xr.Dataset(data_vars, coords={"time": forecast_times, "lat": lat, "lon": lon}, attrs=attrs).to_netcdf(out)
    else:
        if out.suffix.lower() != ".npz":
            out = out.with_name(out.name + ".npz")
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
    return str(out)


def forecast_latest(config_path: str, checkpoint: str, steps: int = 1, output: str = "", device: str = "auto") -> dict:
    """Run one forecast cycle from the most recent data in ``data.source``.

    All files of the source are combined and the last ``history`` time steps that contain every
    variable are used, so time-split archives, one-variable-per-file archives and real-time drop
    folders all work. ``output`` ending in .nc writes NetCDF (u, v, theta, q, alpha, beta, kappa);
    anything else writes a compressed NumPy .npz archive. Values are in physical units.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    cfg = load_config(config_path)
    dev = resolve_device(device)
    model, normalizer, blob = load_checkpoint(checkpoint, dev)
    check_variable_mapping(blob, cfg.data.variables)
    hist = model.cfg.history
    with open_weather_dataset(cfg.data) as ds:
        work, _ = select_dataset(ds, cfg.data, tail_steps=hist + LATEST_LOOKBACK_STEPS)
        raw, times, lat, lon = state_from_selection(work, cfg.data)  # drops incomplete trailing steps
    raw, times = raw[-max(hist, 2):], times[-max(hist, 2):]
    if raw.shape[0] < hist:
        raise ValueError(f"Need at least {hist} time steps in the data; found {raw.shape[0]}")
    if raw.shape[0] >= 2:
        inferred_dt_hours = infer_time_step_hours(times)
        if not math.isclose(inferred_dt_hours, model.cfg.dt_hours, abs_tol=max(1e-6, 0.01 * model.cfg.dt_hours)):
            raise ValueError(f"Dataset time step {inferred_dt_hours:.6g} h does not match checkpoint dt_hours {model.cfg.dt_hours:.6g} h")
    recent = time_values_ns(times[-hist:])
    expected = (hist - 1) * model.cfg.dt_hours * NS_PER_HOUR
    if hist > 1 and abs((recent[-1] - recent[0]) - expected) > 0.05 * model.cfg.dt_hours * NS_PER_HOUR:
        raise ValueError(f"The last {hist} time steps are not contiguous (missing timestamps); cannot build the history window")
    result = run_forecast(model, normalizer, raw[-hist:], lat, lon, steps, dev, blob.get("data_range"))
    step = np.timedelta64(int(round(model.cfg.dt_hours * 3600)), "s")
    forecast_times = np.array([times[-1] + step * (i + 1) for i in range(steps)]).astype("datetime64[ns]")
    if output:
        output = write_forecast(
            output,
            result,
            forecast_times,
            lat,
            lon,
            blob.get("data_units") or {},
            {
                "source": cfg.data.source,
                "analysis_time": str(times[-1]),
                "checkpoint": str(checkpoint),
                "model": "FNGFT-AI",
                "warnings": " | ".join(result["warnings"]) or None,
            },
        )
    return {
        "source": cfg.data.source,
        "analysis_time": str(times[-1]),
        "forecast_time_end": str(forecast_times[-1]),
        "steps": steps,
        "dt_hours": model.cfg.dt_hours,
        "grid": [int(lat.size), int(lon.size)],
        "alpha_mean": float(result["alpha"].mean()),
        "beta_mean": float(result["beta"].mean()),
        "kappa_mean": float(result["kappa"].mean()),
        "warnings": result["warnings"],
        "first_unphysical_step": result["first_unphysical_step"],
        "output": output,
        "_result": result,
        "_times": forecast_times,
        "_lat": lat,
        "_lon": lon,
    }


def public_summary(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-shot forecast from the most recent data in data.source.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", default="artifacts/latest_forecast.nc")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = forecast_latest(args.config, args.checkpoint, args.steps, args.output, args.device)
    for warning in result["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(json.dumps(public_summary(result), indent=2))


if __name__ == "__main__":
    main()
