from pathlib import Path

import pytest
import yaml

from fngft.config import load_config
from fngft.synthetic import VARIABLES, make_synthetic_dataset
from fngft.train import train


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """A tiny synthetic dataset, config and trained checkpoint shared by the end-to-end tests."""
    work = tmp_path_factory.mktemp("e2e")
    data_dir = work / "data"
    make_synthetic_dataset(data_dir / "demo.nc", n_times=160, n_lat=12, n_lon=24)
    config_path = work / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "source": str(data_dir),
                    "variables": VARIABLES,
                    "train_start": "2026-01-01",
                    "train_end": "2026-01-25T23:59:59",
                    "val_start": "2026-01-26",
                    "val_end": "2026-02-03T23:59:59",
                    "test_start": "2026-02-04",
                    "test_end": "2026-02-09T23:59:59",
                },
                "model": {"hidden": 8, "memory_dim": 8, "memory_heads": 2, "memory_layers": 1, "history": 3},
                "training": {"batch_size": 8, "epochs": 2, "learning_rate": 0.002, "checkpoint": str(work / "model.pt")},
            }
        )
    )
    checkpoint = train(load_config(config_path), log=lambda *_: None)
    return {"work": work, "config": Path(config_path), "checkpoint": Path(checkpoint)}
