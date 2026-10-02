from pathlib import Path
import numpy as np
import xarray as xr
import torch

from fngft.config import DataConfig, ModelConfig
from fngft.io import inspect_dataset, open_weather_dataset
from fngft.model import FNGFTWeatherModel
from fngft.preprocess import Standardizer, TemporalWindowDataset, prepare_dataset

path = Path('tmp_demo.nc')
rng = np.random.default_rng(7)
time = np.array(np.arange(20), dtype='timedelta64[h]') + np.datetime64('2026-01-01')
lat = np.linspace(-20, 20, 8)
lon = np.linspace(0, 315, 16)
shape = (len(time), len(lat), len(lon))
data = {
    'u10': ('time lat lon'.split(), rng.normal(size=shape).astype(np.float32)),
    'v10': ('time lat lon'.split(), rng.normal(size=shape).astype(np.float32)),
    'theta2m': ('time lat lon'.split(), rng.normal(size=shape).astype(np.float32)),
    'q2m': ('time lat lon'.split(), np.abs(rng.normal(size=shape)).astype(np.float32)),
}

ds = xr.Dataset(data, coords={'time': time, 'lat': lat, 'lon': lon})
ds.to_netcdf(path, engine='scipy')

cfg = DataConfig(
    source=str(path),
    variables={'u': 'u10', 'v': 'v10', 'theta': 'theta2m', 'q': 'q2m'},
    train_start='2026-01-01',
    train_end='2026-01-01T10:00',
)
profile = inspect_dataset(cfg)
print('time_count', profile.time_count)
with open_weather_dataset(cfg) as ds:
    state, times, lat, lon = prepare_dataset(ds, cfg)
normalizer = Standardizer.fit(state[:10])
standardized = normalizer.transform(state)
windows = TemporalWindowDataset(standardized, times, history=4, horizon=2, start='2026-01-01T06:00', end='2026-01-01T18:00')
print('starts', windows.starts[:3], windows.starts[-3:])
assert windows.starts[0] == 2
assert windows.starts[-1] == 13
history, future = windows[0]
model_cfg = ModelConfig(hidden=8, memory_dim=8, memory_heads=2, memory_layers=1, history=4)
model = FNGFTWeatherModel(model_cfg)
with torch.inference_mode():
    pred, info = model(history.unsqueeze(0), torch.tensor(lat), torch.tensor(lon), steps=2)
print('pred_shape', pred.shape)
print('alpha_range', info['alpha'].min().item(), info['alpha'].max().item())
print('beta_range', info['beta'].min().item(), info['beta'].max().item())
assert pred.shape == (1, 2, 4, len(lat), len(lon))
assert info['alpha'].min().item() >= model_cfg.alpha_min
assert info['alpha'].max().item() <= model_cfg.alpha_max
assert info['beta'].min().item() >= model_cfg.beta_min
assert info['beta'].max().item() <= model_cfg.beta_max
print('PASS')
