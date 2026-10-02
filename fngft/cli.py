from __future__ import annotations

import argparse
import json

from .config import load_config
from .evaluate import evaluate
from .io import inspect_dataset
from .realtime import forecast_latest
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="FNGFT-AI real-data research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect")
    inspect_p.add_argument("--config", required=True)
    inspect_p.add_argument("--latest", action="store_true")

    train_p = sub.add_parser("train")
    train_p.add_argument("--config", required=True)

    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--config", required=True)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--split", default="test", choices=["train", "val", "test"])

    forecast_p = sub.add_parser("forecast-latest")
    forecast_p.add_argument("--config", required=True)
    forecast_p.add_argument("--checkpoint", required=True)
    forecast_p.add_argument("--steps", type=int, default=1)
    forecast_p.add_argument("--output", default="artifacts/latest_forecast.npz")

    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_dataset(load_config(args.config).data, latest=args.latest).__dict__, indent=2))
    elif args.command == "train":
        print(train(load_config(args.config)))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.config, args.checkpoint, args.split), indent=2))
    elif args.command == "forecast-latest":
        print(json.dumps(forecast_latest(args.config, args.checkpoint, args.steps, args.output), indent=2))


if __name__ == "__main__":
    main()
