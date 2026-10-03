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


def _variant_config(trained, tmp_path, source):
    import yaml

    raw = yaml.safe_load(trained["config"].read_text())
    raw["data"]["source"] = str(source)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def test_forecast_latest_with_glob_and_per_variable_files(trained, tmp_path):
    ds = xr.open_dataset(trained["work"] / "data" / "demo.nc").load()
    reference = forecast_latest(str(trained["config"]), str(trained["checkpoint"]), steps=2)
    (tmp_path / "split").mkdir()
    ds.isel(time=slice(0, 80)).to_netcdf(tmp_path / "split" / "demo_2026a.nc")
    ds.isel(time=slice(80, None)).to_netcdf(tmp_path / "split" / "demo_2026b.nc")
    (tmp_path / "pervar").mkdir()
    for name in ds.data_vars:
        ds[[name]].isel(time=slice(0, -2 if name == "q2m" else None)).to_netcdf(tmp_path / "pervar" / f"{name}.nc")
    globbed = forecast_latest(_variant_config(trained, tmp_path, tmp_path / "split" / "demo_*.nc"), str(trained["checkpoint"]), steps=2)
    assert globbed["analysis_time"] == reference["analysis_time"]
    assert np.allclose(globbed["_result"]["forecast_physical"], reference["_result"]["forecast_physical"])
    pervar = forecast_latest(_variant_config(trained, tmp_path, tmp_path / "pervar"), str(trained["checkpoint"]), steps=2)
    # q2m ends two steps earlier, so the latest complete analysis time is two steps back.
    assert pervar["analysis_time"] == str(ds.time.values[-3])


def test_forecast_output_suffix_is_reported(trained, tmp_path):
    out = forecast_latest(str(trained["config"]), str(trained["checkpoint"]), steps=1, output=str(tmp_path / "fc"))
    assert out["output"].endswith("fc.npz") and (tmp_path / "fc.npz").exists()
    assert out["warnings"] == [] or isinstance(out["warnings"], list)


def test_evaluate_climatology_is_per_grid_point(trained):
    from fngft.config import load_config
    from fngft.io import open_weather_dataset
    from fngft.preprocess import TemporalWindowDataset, area_weights, prepare_dataset, time_slice_indices

    result = evaluate(str(trained["config"]), str(trained["checkpoint"]), "test", steps=1)
    cfg = load_config(trained["config"])
    with open_weather_dataset(cfg.data) as ds:
        raw, times, lat, lon = prepare_dataset(ds, cfg.data, crop_to_splits=True)
    lo, hi = time_slice_indices(times, cfg.data.train_start, cfg.data.train_end)
    clim = raw[lo:hi].mean(axis=0)
    windows = TemporalWindowDataset(raw, times, history=3, horizon=1, start=cfg.data.test_start, end=cfg.data.test_end)
    targets = np.stack([raw[s + 3] for s in windows.starts])
    w = area_weights(lat)[None, :, None]
    expected = np.sqrt(((targets[:, 2] - clim[2]) ** 2 * w).mean())
    assert np.isclose(result["leads"][0]["variables"]["theta"]["rmse_climatology"], expected, rtol=1e-4)


def test_api_handles_descending_latitude_and_bad_input(trained, monkeypatch):
    import fngft.api as api

    monkeypatch.setattr(api, "MODEL_PATH", str(trained["checkpoint"]))
    with TestClient(api.app) as client:
        ds = xr.open_dataset(trained["work"] / "data" / "demo.nc")
        hist = np.stack([ds[v].values[-3:] for v in ("u10", "v10", "theta2m", "q2m")], axis=1)
        lat, lon = ds["lat"].values, ds["lon"].values
        asc = client.post("/forecast", json={"history": hist.tolist(), "lat": lat.tolist(), "lon": lon.tolist(), "units": "physical"}).json()
        desc = client.post(
            "/forecast",
            json={"history": hist[:, :, ::-1].tolist(), "lat": lat[::-1].tolist(), "lon": lon.tolist(), "units": "physical"},
        ).json()
        assert np.allclose(np.asarray(desc["forecast_physical"])[..., ::-1, :], np.asarray(asc["forecast_physical"]), atol=1e-4)
        assert "warnings" in asc
        ragged = hist.tolist()
        ragged[0][0][0] = ragged[0][0][0][:-1]
        assert client.post("/forecast", json={"history": ragged, "units": "physical"}).status_code == 422
        flat = client.post("/forecast", json={"history": hist.tolist(), "lat": [10.0] * len(lat), "lon": lon.tolist(), "units": "physical"})
        assert flat.status_code == 422


def test_long_rollout_stays_plausible(trained):
    from fngft.evaluate import load_checkpoint
    from fngft.realtime import plausibility_warnings

    model, normalizer, blob = load_checkpoint(str(trained["checkpoint"]), torch.device("cpu"))
    ds = xr.open_dataset(trained["work"] / "data" / "demo.nc")
    hist = np.stack([ds[v].values[-3:] for v in ("u10", "v10", "theta2m", "q2m")], axis=1).astype(np.float32)
    x = torch.from_numpy(normalizer.transform(hist)).unsqueeze(0)
    lat = torch.tensor(blob["grid"]["lat"])
    lon = torch.tensor(blob["grid"]["lon"])
    with torch.inference_mode():
        pred, _ = model(x, lat, lon, steps=40)
    std = pred[0].numpy()
    phys = normalizer.inverse(std)
    warnings, first_bad = plausibility_warnings(std, phys)
    assert first_bad is None, warnings
    assert phys[:, 3].min() >= -1e-6
