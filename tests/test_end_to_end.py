import json

import numpy as np
import pytest
import torch
import xarray as xr
from fastapi.testclient import TestClient

from fngft.evaluate import evaluate
from fngft.realtime import forecast_latest


def test_checkpoint_contents(trained):
    blob = torch.load(trained["checkpoint"], weights_only=False)
    assert blob["model_config"]["dt_hours"] == 6.0
    assert blob["model_config"]["longitude_periodic"] is True
    assert blob["data_units"]["theta"] == "K"
    assert len(blob["metrics"]["history"]) == 2
    assert trained["checkpoint"].with_suffix(".history.json").exists()


def test_evaluate_reports_baselines(trained):
    result = evaluate(str(trained["config"]), str(trained["checkpoint"]), "test", steps=2)
    assert result["windows"] > 0 and len(result["leads"]) == 2
    assert result["leads"][1]["lead_hours"] == 12.0
    for lead in result["leads"]:
        for metrics in lead["variables"].values():
            assert np.isfinite(metrics["rmse"]) and np.isfinite(metrics["rmse_persistence"])
    json.dumps(result)


@pytest.mark.parametrize("suffix", [".nc", ".npz"])
def test_forecast_latest_writes_output(trained, suffix):
    out = trained["work"] / f"forecast{suffix}"
    summary = forecast_latest(str(trained["config"]), str(trained["checkpoint"]), steps=3, output=str(out))
    assert summary["steps"] == 3 and out.exists()
    if suffix == ".nc":
        ds = xr.open_dataset(out)
        assert ds["theta"].shape == (3, 12, 24)
        assert np.isfinite(ds["theta"].values).all()
        assert (ds["time"].diff("time") == np.timedelta64(6, "h")).all()


def test_api_and_dashboard(trained, monkeypatch):
    import fngft.api as api

    monkeypatch.setattr(api, "MODEL_PATH", str(trained["checkpoint"]))
    monkeypatch.setattr(api, "CONFIG_PATH", str(trained["config"]))
    with TestClient(api.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert "FNGFT-AI" in client.get("/").text
        info = client.get("/model-info").json()
        assert info["history"] == 3 and info["grid_shape"] == [12, 24]
        ds = xr.open_dataset(trained["work"] / "data" / "demo.nc")
        hist = np.stack([ds[v].values[-3:] for v in ("u10", "v10", "theta2m", "q2m")], axis=1)
        r = client.post("/forecast", json={"history": hist.tolist(), "steps": 2, "units": "physical"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert np.asarray(body["forecast_physical"]).shape == (2, 4, 12, 24)
        assert body["lead_hours"] == [6.0, 12.0]
        assert client.post("/forecast", json={"history": hist[:2].tolist()}).status_code == 422
        latest = client.get("/forecast-latest", params={"steps": 2, "max_size": 8})
        assert latest.status_code == 200, latest.text
        assert np.asarray(latest.json()["fields"]["u"]).shape == (2, 6, 8)


def test_api_without_model_stays_up(tmp_path, monkeypatch):
    import fngft.api as api

    monkeypatch.setattr(api, "MODEL_PATH", str(tmp_path / "missing.pt"))
    with TestClient(api.app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is False and "not found" in health["load_error"]
        assert client.get("/").status_code == 200
        assert client.post("/forecast", json={"history": [[[[0.0]]]]}).status_code == 503
