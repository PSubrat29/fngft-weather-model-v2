from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import load_config


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def run_demo(workdir: str, epochs: int = 2) -> dict:
    """Synthetic end-to-end check: data -> inspect -> train -> evaluate -> forecast."""
    import yaml

    from .evaluate import evaluate
    from .io import inspect_dataset
    from .realtime import forecast_latest, public_summary
    from .synthetic import VARIABLES, make_synthetic_dataset
    from .train import train

    work = Path(workdir)
    data_dir = work / "data"
    make_synthetic_dataset(data_dir / "demo.nc", n_times=360)
    config_path = work / "demo_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "source": str(data_dir),
                    "variables": VARIABLES,
                    "train_start": "2026-01-01",
                    "train_end": "2026-02-10T23:59:59",
                    "val_start": "2026-02-11",
                    "val_end": "2026-02-25T23:59:59",
                    "test_start": "2026-02-26",
                    "test_end": "2026-03-31T23:59:59",
                },
                "model": {"hidden": 16, "memory_dim": 16, "memory_heads": 2, "memory_layers": 1, "history": 4},
                "training": {"batch_size": 8, "epochs": epochs, "learning_rate": 0.002, "checkpoint": str(work / "demo.pt")},
            },
            sort_keys=False,
        )
    )
    cfg = load_config(config_path)
    print("== inspect")
    _print(inspect_dataset(cfg.data, latest=True).__dict__)
    print("== train")
    checkpoint = train(cfg)
    print("== evaluate (test split, 3 lead times)")
    result = evaluate(str(config_path), str(checkpoint), "test", steps=3)
    for lead in result["leads"]:
        print(
            f"lead +{lead['lead_hours']:g} h: rmse_std={lead['rmse_standardized']:.4f} "
            f"persistence={lead['rmse_persistence_standardized']:.4f}"
        )
    print("== forecast-latest")
    out = forecast_latest(str(config_path), str(checkpoint), steps=4, output=str(work / "demo_forecast.nc"))
    summary = public_summary(out)
    _print(summary)
    ok = np.isfinite(out["_result"]["forecast_physical"]).all()
    print(f"DEMO {'PASSED' if ok else 'FAILED'}: outputs in {work}")
    return {"evaluation": result, "forecast": summary, "ok": bool(ok)}


def main() -> None:
    parser = argparse.ArgumentParser(prog="fngft", description="FNGFT-AI real-data research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Validate and profile the configured dataset")
    inspect_p.add_argument("--config", required=True)
    inspect_p.add_argument("--latest", action="store_true", help="Inspect only the newest file in data.source")

    train_p = sub.add_parser("train", help="Train a checkpoint")
    train_p.add_argument("--config", required=True)
    train_p.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    train_p.add_argument("--device", default=None, help="Override training.device (cpu, cuda, auto)")
    train_p.add_argument("--checkpoint", default=None, help="Override training.checkpoint")

    eval_p = sub.add_parser("evaluate", help="Score a checkpoint on a split against baselines")
    eval_p.add_argument("--config", required=True)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--split", default="test", choices=["train", "val", "test"])
    eval_p.add_argument("--steps", type=int, default=1, help="Number of autoregressive lead times")
    eval_p.add_argument("--batch-size", type=int, default=8)
    eval_p.add_argument("--max-windows", type=int, default=None)
    eval_p.add_argument("--device", default="auto")
    eval_p.add_argument("--output", default="", help="Write the JSON report to this file")

    forecast_p = sub.add_parser("forecast-latest", help="Forecast from the newest file in data.source")
    forecast_p.add_argument("--config", required=True)
    forecast_p.add_argument("--checkpoint", required=True)
    forecast_p.add_argument("--steps", type=int, default=1)
    forecast_p.add_argument("--output", default="artifacts/latest_forecast.nc", help=".nc for NetCDF, otherwise .npz")
    forecast_p.add_argument("--device", default="auto")

    serve_p = sub.add_parser("serve", help="Run the HTTP API and web dashboard")
    serve_p.add_argument("--checkpoint", default=None, help="Checkpoint to serve (sets MODEL_PATH)")
    serve_p.add_argument("--config", default=None, help="Config enabling /forecast-latest (sets FNGFT_CONFIG)")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8080)

    demo_p = sub.add_parser("demo", help="Synthetic end-to-end installation check")
    demo_p.add_argument("--workdir", default="artifacts/demo")
    demo_p.add_argument("--epochs", type=int, default=2)

    args = parser.parse_args()
    if args.command == "inspect":
        from .io import inspect_dataset

        _print(inspect_dataset(load_config(args.config).data, latest=args.latest).__dict__)
    elif args.command == "train":
        from .train import train

        cfg = load_config(args.config)
        overrides = {k: v for k, v in {"epochs": args.epochs, "device": args.device, "checkpoint": args.checkpoint}.items() if v is not None}
        if overrides:
            cfg = replace(cfg, training=replace(cfg.training, **overrides))
        print(f"checkpoint={train(cfg)}")
    elif args.command == "evaluate":
        from .evaluate import evaluate

        result = evaluate(
            args.config,
            args.checkpoint,
            args.split,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            max_windows=args.max_windows,
        )
        _print(result)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2))
    elif args.command == "forecast-latest":
        from .realtime import forecast_latest, public_summary

        _print(public_summary(forecast_latest(args.config, args.checkpoint, args.steps, args.output, args.device)))
    elif args.command == "serve":
        if args.checkpoint:
            os.environ["MODEL_PATH"] = args.checkpoint
        if args.config:
            os.environ["FNGFT_CONFIG"] = args.config
        import uvicorn

        uvicorn.run("fngft.api:app", host=args.host, port=args.port)
    elif args.command == "demo":
        run_demo(args.workdir, args.epochs)


if __name__ == "__main__":
    main()
