import numpy as np
import pytest
import xarray as xr

from fngft.config import DataConfig, load_config
from fngft.io import inspect_dataset, open_weather_dataset
from fngft.preprocess import TemporalWindowDataset, prepare_dataset

VARS = {"u": "u", "v": "v", "theta": "t", "q": "q"}


def make_ds(n_time=24, lat=None, lon=None, levels=None, drop_times=(), nan_at=None, extra_dim=None):
    rng = np.random.default_rng(0)
    time = np.datetime64("2026-01-01", "ns") + np.arange(n_time) * np.timedelta64(6, "h")
    time = np.delete(time, list(drop_times))
    lat = np.linspace(40.0, 0.0, 9) if lat is None else lat  # descending, like ERA5
    lon = np.arange(0.0, 360.0, 22.5) if lon is None else lon
    dims, coords = ["time", "lat", "lon"], {"time": time, "lat": lat, "lon": lon}
    shape = [len(time), len(lat), len(lon)]
    if levels is not None:
        dims.insert(1, "level")
        coords["level"] = levels
        shape.insert(1, len(levels))
    if extra_dim:
        dims.insert(1, extra_dim[0])
        shape.insert(1, extra_dim[1])
    data = {}
    for name in VARS.values():
        arr = rng.normal(size=shape).astype(np.float32)
        if levels is not None:
            arr += np.asarray(levels, dtype=np.float32).reshape(1, -1, 1, 1)
        data[name] = (dims, arr)
    ds = xr.Dataset(data, coords=coords)
    if nan_at is not None:
        ds["t"][nan_at] = np.nan
    return ds


def cfg(**kw):
    return DataConfig(source="unused", variables=VARS, **kw)


def test_descending_latitude_is_sorted_ascending():
    state, times, lat, lon = prepare_dataset(make_ds(), cfg())
    assert np.all(np.diff(lat) > 0)
    assert state.shape == (24, 4, 9, 16)


def test_level_selection():
    ds = make_ds(levels=[500, 850])
    state, *_ = prepare_dataset(ds, cfg(level_dim="level", level_value=850))
    assert state.shape == (24, 4, 9, 16)
    assert state[:, 2].mean() > 800  # level offset added in make_ds


def test_extra_dimension_needs_level_selection():
    with pytest.raises(ValueError, match="extra dimension"):
        prepare_dataset(make_ds(extra_dim=("member", 3)), cfg())
    state, *_ = prepare_dataset(make_ds(extra_dim=("member", 1)), cfg())  # size-1 dims are squeezed
    assert state.shape == (24, 4, 9, 16)


def test_region_and_coarsen():
    ds = make_ds(lat=np.linspace(-40, 40, 17), lon=np.arange(0.0, 360.0, 10.0))
    state, _, lat, lon = prepare_dataset(ds, cfg(region={"lat_min": -10, "lat_max": 30, "lon_min": -40, "lon_max": 40}))
    assert lat.min() >= -10 and lat.max() <= 30
    assert lon.min() >= -40 and lon.max() <= 40 and np.all(np.diff(lon) > 0)
    state2, _, lat2, lon2 = prepare_dataset(ds, cfg(coarsen=2))
    assert state2.shape[2:] == (8, 18)


def test_missing_values_error_and_interpolate():
    ds = make_ds(nan_at={"time": 5, "lat": 2, "lon": 3})
    with pytest.raises(ValueError, match="missing_values"):
        prepare_dataset(ds, cfg())
    state, *_ = prepare_dataset(ds, cfg(missing_values="interpolate"))
    assert np.isfinite(state).all()


def test_time_gap_windows_are_skipped():
    ds = make_ds(n_time=30, drop_times=[12])
    state, times, *_ = prepare_dataset(ds, cfg())
    windows = TemporalWindowDataset(state, times, history=3, horizon=1)
    for s in windows.starts:
        span = times[s + 3] - times[s]
        assert span == np.timedelta64(18, "h")
    assert len(windows) == (12 - 4 + 1) + (17 - 4 + 1)


def test_irregular_time_is_rejected():
    ds = make_ds()
    times = ds.time.values.copy()
    times[5] += np.timedelta64(2, "h")
    with pytest.raises(ValueError, match="not regular"):
        prepare_dataset(ds.assign_coords(time=times), cfg())


def test_directory_of_files_is_combined(tmp_path):
    ds = make_ds(n_time=20)
    ds.isel(time=slice(0, 10)).to_netcdf(tmp_path / "part1.nc")
    ds.isel(time=slice(10, 20)).to_netcdf(tmp_path / "part2.nc")
    config = DataConfig(source=str(tmp_path), variables=VARS)
    assert inspect_dataset(config).time_count == 20
    assert inspect_dataset(config, latest=True).time_count == 10
    with open_weather_dataset(config) as opened:
        state, *_ = prepare_dataset(opened, config)
    assert state.shape[0] == 20


def test_unknown_config_key_is_reported(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  sourse: x\n")
    with pytest.raises(ValueError, match="sourse"):
        load_config(path)


def test_shipped_config_templates_load():
    from pathlib import Path

    for path in sorted(Path(__file__).resolve().parents[1].joinpath("configs").glob("*.yaml")):
        load_config(path)
