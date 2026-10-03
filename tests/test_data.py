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
    assert inspect_dataset(config, latest=True).time_count == 20  # last 24 steps, combined across files
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


def _write(ds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def test_glob_and_per_variable_folders_work_in_latest_mode(tmp_path):
    ds = make_ds(n_time=30)
    _write(ds.isel(time=slice(0, 15)), tmp_path / "glob" / "era5_a.nc")
    _write(ds.isel(time=slice(15, 30)), tmp_path / "glob" / "era5_b.nc")
    for name in VARS.values():
        _write(ds[[name]], tmp_path / "pervar" / f"{name}.nc")
    for source in (str(tmp_path / "glob" / "era5_*.nc"), str(tmp_path / "pervar")):
        config = DataConfig(source=source, variables=VARS)
        assert inspect_dataset(config).time_count == 30
        latest = inspect_dataset(config, latest=True)
        assert latest.time_count == 24 and latest.time_end == str(ds.time.values[-1])


def test_existing_path_with_brackets_is_not_a_glob(tmp_path):
    path = _write(make_ds(), tmp_path / "ERA5 [India]" / "x.nc")
    assert inspect_dataset(DataConfig(source=str(path), variables=VARS)).time_count == 24
    assert inspect_dataset(DataConfig(source=str(path.parent), variables=VARS)).time_count == 24


def test_level_value_must_match_a_level():
    ds = make_ds(levels=[50000.0, 85000.0])
    ds["level"].attrs["units"] = "Pa"
    with pytest.raises(ValueError, match="85000"):
        prepare_dataset(ds, cfg(level_dim="level", level_value=850))
    state, *_ = prepare_dataset(ds, cfg(level_dim="level", level_value=85000))
    assert state[:, 2].mean() > 80000


def test_expver_dimension_has_specific_message():
    with pytest.raises(ValueError, match="expver"):
        prepare_dataset(make_ds(extra_dim=("expver", 2)), cfg())


def test_inspect_reports_the_selected_data(tmp_path):
    ds = make_ds(lat=np.linspace(-40, 40, 17), lon=np.arange(0.0, 360.0, 10.0), levels=[500, 850])
    path = _write(ds, tmp_path / "x.nc")
    config = DataConfig(source=str(path), variables=VARS, level_dim="level", level_value=850,
                        region={"lat_min": -10, "lat_max": 30, "lon_min": 20, "lon_max": 60})
    profile = inspect_dataset(config)
    assert profile.selected_level == 850 and profile.lat_count == 9 and profile.lon_count == 5
    assert profile.longitude_periodic is False and profile.missing_value_count == 0
    ds2 = make_ds(nan_at={"time": 5, "lat": 2, "lon": 3})
    path2 = _write(ds2, tmp_path / "nan.nc")
    with pytest.raises(ValueError, match="missing_values"):
        inspect_dataset(DataConfig(source=str(path2), variables=VARS))
    assert inspect_dataset(DataConfig(source=str(path2), variables=VARS, missing_values="interpolate")).missing_value_count == 1


def test_open_ended_split_is_not_cropped():
    ds = make_ds(n_time=40)
    config = cfg(train_start="2026-01-01", train_end="2026-01-04T23:59", test_start="2026-01-08", test_end=None)
    state, times, *_ = prepare_dataset(ds, config, crop_to_splits=True)
    assert times[-1] == ds.time.values[-1]


def test_interpolation_does_not_cross_split_boundaries():
    ds = make_ds(n_time=12)
    t = ds["t"].values
    t[:] = 280.0
    t[6:, 2, 3] = 400.0
    t[3:6, 2, 3] = np.nan
    ds["t"] = (ds["t"].dims, t)
    config = cfg(missing_values="interpolate", train_start="2026-01-01", train_end="2026-01-02T06:00",
                 val_start="2026-01-02T12:00", val_end="2026-01-03T18:00")
    state, *_ = prepare_dataset(ds, config)
    lat_idx = 8 - 2  # latitude is sorted ascending
    assert np.allclose(state[:6, 2, lat_idx, 3], 280.0)


def _config_file(tmp_path, data_extra):
    import yaml

    data = {"source": "x.nc", "train_start": "2020-01-01", "train_end": "2020-06-30T23:59:59",
            "val_start": "2020-07-01", "val_end": "2020-09-30T23:59:59"}
    data.update(data_extra)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"data": data}), encoding="utf-8")
    return path


def test_split_validation(tmp_path):
    with pytest.raises(ValueError, match="overlaps"):
        load_config(_config_file(tmp_path, {"val_start": "2020-06-01"}))
    with pytest.raises(ValueError, match="train_start"):
        load_config(_config_file(tmp_path, {"train_start": None, "train_end": None}))
    with pytest.raises(ValueError, match="must not be null"):
        load_config(_config_file(tmp_path, {"coarsen": None}))
    cfg_tz = load_config(_config_file(tmp_path, {"train_start": "2020-01-01T05:30:00+05:30"}))
    assert cfg_tz.data.train_start == "2020-01-01T00:00:00"


def test_config_is_read_as_utf8(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_bytes("﻿data:\n  source: data/Météo/era5.nc\n  train_start: '2020-01-01'\n".encode("utf-8"))
    assert load_config(path).data.source == "data/Météo/era5.nc"


def test_date_only_end_covers_the_whole_period(tmp_path):
    cfg_dates = load_config(_config_file(tmp_path, {"train_end": "2020-06-30", "val_start": "2020-07-01", "val_end": "2020-09"}))
    assert cfg_dates.data.train_end.startswith("2020-06-30T23:59:59")
    assert cfg_dates.data.val_end.startswith("2020-09-30T23:59:59")
    assert cfg_dates.data.val_start == "2020-07-01T00:00:00"


def test_inspect_skips_incomplete_edge_steps(tmp_path):
    ds = make_ds(n_time=30)
    for name in VARS.values():
        part = ds[[name]].isel(time=slice(0, 27 if name == "q" else None))
        _write(part, tmp_path / "pervar" / f"{name}.nc")
    config = DataConfig(source=str(tmp_path / "pervar"), variables=VARS)
    for latest in (False, True):
        profile = inspect_dataset(config, latest=latest)
        assert profile.time_end == str(ds.time.values[26]) and profile.incomplete_edge_steps == 3
    with open_weather_dataset(config) as opened:
        state, times, *_ = prepare_dataset(opened, config)
    assert len(times) == 27


def test_glob_inside_folder_with_brackets(tmp_path):
    folder = tmp_path / "ERA5 [2020]"
    _write(make_ds(n_time=10), folder / "a.nc")
    from fngft.io import resolve_sources

    assert len(resolve_sources(str(folder / "*.nc"))) == 1


def test_more_config_validation(tmp_path):
    from dataclasses import replace
    from fngft.config import validate_config

    with pytest.raises(ValueError, match="level_dim"):
        load_config(_config_file(tmp_path, {"level_value": 500}))
    with pytest.raises(ValueError, match="region.lat_min"):
        load_config(_config_file(tmp_path, {"region": {"lat_min": "5", "lat_max": 40}}))
    assert load_config(_config_file(tmp_path, {"region": {"lat_min": None, "lat_max": 40}})).data.region == {"lat_max": 40}
    with pytest.raises(ValueError, match="coarsen"):
        load_config(_config_file(tmp_path, {"coarsen": 2.5}))
    yearly = load_config(_config_file(tmp_path, {"train_start": 2018, "train_end": 2019, "val_start": 2020, "val_end": 2020}))
    assert yearly.data.train_start == "2018-01-01T00:00:00" and yearly.data.val_end.startswith("2020-12-31T23:59:59")
    cfg_ok = load_config(_config_file(tmp_path, {}))
    with pytest.raises(ValueError, match="epochs"):
        validate_config(replace(cfg_ok, training=replace(cfg_ok.training, epochs=0)))
