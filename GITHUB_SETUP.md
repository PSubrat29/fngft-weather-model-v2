# Repository setup and maintenance

The repository is already on GitHub (`PSubrat29/fngft-weather-model-v2`). This page covers working
with it locally.

## Clone and install

```bash
git clone https://github.com/PSubrat29/fngft-weather-model-v2.git
cd fngft-weather-model-v2
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python -m fngft demo
```

All commands are run from the repository root (the folder containing `pyproject.toml`).

## What is tracked and what is not

`.gitignore` keeps these out of git: `artifacts/` (checkpoints, forecasts), `data/` and all
`*.nc/*.grib/*.zarr` datasets, virtual environments, `__pycache__/`, `*.egg-info/`, `tmp_*` files.
Keep datasets and checkpoints in external storage (or Git LFS if they must be versioned) and record
their version in the experiment log (see `GOVERNANCE.md`).

## Continuous integration

`.github/workflows/tests.yml` runs `pytest` and the synthetic demo on Python 3.11 to 3.14 for
every push and pull request. A red check means the code is broken; look at the failing step's log in
the **Actions** tab.

## Branch workflow

```bash
git checkout -b my-change
# edit, then:
pytest -q
git add -A && git commit -m "Describe the change"
git push -u origin my-change
```

Open a pull request on GitHub and merge it once the checks are green.

## Troubleshooting

| Message | Fix |
|---|---|
| `Required dimension 'lat' not found. Available dimensions: [...]` | set `data.lat_dim` / `lon_dim` / `time_dim` to the listed names |
| `Variable 'x' for 'u' not found ... Available variables: [...]` | fix `data.variables` |
| `extra dimension 'level'` | set `level_dim` and `level_value` |
| `non-finite (NaN/inf) values` | clean the data or set `missing_values: interpolate` |
| `Empty time selection` / `No temporal windows` | the split dates are outside the data period |
| `Dataset time step is X h but model.dt_hours is Y h` | set `model.dt_hours: null` |
| `Non-finite training loss` | lower `training.learning_rate` |
| out of memory | `region`, `coarsen`, a shorter period, smaller `batch_size`/`hidden` |
| `/health` shows `degraded` | `load_error` names the problem (usually a wrong `MODEL_PATH`) |
