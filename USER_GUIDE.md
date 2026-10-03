# What you do on your side

Follow these steps in order on your computer. Every command is run from the repository folder
(the one that contains `pyproject.toml`). Windows commands are shown; on Linux/macOS replace
`.venv\Scripts\activate` with `source .venv/bin/activate`.

## Step 1 — Get the fixed code

If you already have the old copy (`D:\ParidaUser\Claude-Project\fngft_weather_model_v2`), do not
reuse it: the code was moved to the repository root and many files changed. Clone a fresh copy:

```bat
cd D:\ParidaUser\Claude-Project
git clone https://github.com/PSubrat29/fngft-weather-model-v2.git fngft-v3
cd fngft-v3
git checkout claude/focused-hawking-k8bzef
```

(After the pull request is merged into `main`, plain `git clone` is enough.)

## Step 2 — Create a clean Python environment

Python 3.11, 3.12, 3.13 or 3.14.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

NVIDIA GPU: before the line `pip install -r requirements.txt`, install the CUDA build of PyTorch with
the command shown at https://pytorch.org (Get Started → your OS/CUDA version).

## Step 3 — Confirm the installation works (no data needed)

```bat
pytest -q
python -m fngft demo
```

Expected: `38 passed` (or more) and the last demo line `DEMO PASSED`. If either fails, send me the
full output.

## Step 4 — Optional: confirm on real public data (ERA5)

```bat
pip install gcsfs
python scripts\download_era5_sample.py
python -m fngft train --config configs\era5_sample.yaml
python -m fngft evaluate --config configs\era5_sample.yaml --checkpoint artifacts\era5_sample.pt --steps 4
```

Training takes about 10 minutes on a laptop CPU. In every epoch line `val_mse` should be well below
`persistence_val_mse` (about a third of it). The results I obtained are in `AUDIT_REPORT.md`.

## Step 5 — Describe your historical dataset to the model

1. Put your files in one folder, for example `D:\weather\history\`. One file or many files (e.g. one
   per year) are both fine; NetCDF is easiest.
2. Copy the template: `copy configs\real_data.yaml configs\my_data.yaml`
3. Edit `configs\my_data.yaml` — only the `data:` section needs changes:
   - `source:` your folder or file, e.g. `D:/weather/history` (forward slashes are safest).
   - `time_dim`, `lat_dim`, `lon_dim`: the dimension names in your files
     (ERA5 from Copernicus often uses `valid_time`, `latitude`, `longitude`).
   - `variables:` your names for eastward wind (`u`, m/s), northward wind (`v`, m/s),
     temperature (`theta`) and humidity (`q`).
   - multi-level files: `level_dim` and `level_value` (e.g. `pressure_level`, `850`).
   - `train_*`, `val_*`, `test_*`: chronological date ranges inside your data period, e.g. the oldest
     ~70% for training, the next ~15% for validation, the newest ~15% for testing.
   - large data: `region:` (e.g. `{lat_min: 5, lat_max: 40, lon_min: 65, lon_max: 100}`) and/or
     `coarsen: 2` or `4`.
   - data with gaps/NaNs: `missing_values: interpolate`.
   - hourly data: in the `training:` section set `rollout_steps: 6` (slower training, much more stable
     multi-day forecasts); keep the default 2 for 3- or 6-hourly data.

## Step 6 — Validate the dataset

```bat
python -m fngft inspect --config configs\my_data.yaml
```

It prints the time range, time step, grid (after your region/coarsen settings), the selected level,
units and the number of missing values — or an error that names the fix (it lists the available
dimension and variable names, and tells you if a level value or missing data is the problem). Do not train until this succeeds. If you are unsure what your
files contain, run `python -c "import xarray as xr; print(xr.open_dataset(r'D:\weather\history\one_file.nc'))"`
and send me the output.

## Step 7 — Train

```bat
python -m fngft train --config configs\my_data.yaml
```

Watch the epoch lines: `val_mse` must go below `persistence_val_mse`. The best model is saved to
`artifacts\fngft_real.pt` (change `training.checkpoint` to keep several runs). Training stops early when
validation stops improving.

## Step 8 — Evaluate on the held-out test period

```bat
python -m fngft evaluate --config configs\my_data.yaml --checkpoint artifacts\fngft_real.pt --split test --steps 4 --output artifacts\eval_test.json
```

For each variable and lead time the report gives RMSE in physical units, the persistence and
climatology RMSE, `skill_vs_persistence` (positive = better than persistence) and anomaly correlation.

Forecast lead time: on the ERA5 sample the model beats persistence for all variables up to about
3 days. Every forecast is checked; if a step leaves the physically plausible range you get a
`WARNING` line (CLI), a `warnings` field (API/NetCDF) and a banner on the dashboard.

## Step 9 — Forecast and view results

```bat
python -m fngft forecast-latest --config configs\my_data.yaml --checkpoint artifacts\fngft_real.pt --steps 4 --output artifacts\latest_forecast.nc
python -m fngft serve --checkpoint artifacts\fngft_real.pt --config configs\my_data.yaml --port 8080
```

Open http://127.0.0.1:8080 in a browser for the dashboard and press **Run forecast**.

## Step 10 — Merge the fixes on GitHub

Open https://github.com/PSubrat29/fngft-weather-model-v2, create a pull request from branch
`claude/focused-hawking-k8bzef` into `main` (or ask me to open it), wait for the green **tests** check
in the Actions tab, then merge.

## What to send me for the data-analysis phase

- the output of `inspect` for your dataset,
- the training log (epoch lines),
- `artifacts\eval_test.json`.
