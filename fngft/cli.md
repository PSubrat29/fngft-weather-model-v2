# `fngft/cli.py`

## Purpose

Single command-line interface to the real-data workflow. Run it as `python -m fngft …`,
`python -m fngft.cli …` or (after `pip install -e .`) `fngft …`.

## Commands

```bash
python -m fngft demo                                   # synthetic end-to-end installation check
python -m fngft inspect  --config CONFIG [--latest]    # validate and profile the dataset
python -m fngft train    --config CONFIG [--epochs N] [--device cpu|cuda] [--checkpoint PATH]
python -m fngft evaluate --config CONFIG --checkpoint CKPT [--split test] [--steps 4] [--output report.json]
python -m fngft forecast-latest --config CONFIG --checkpoint CKPT [--steps 6] [--output forecast.nc]
python -m fngft serve    --checkpoint CKPT [--config CONFIG] [--host 0.0.0.0] [--port 8080]
```

The CLI keeps the operational path explicit and reproducible. v0.2's CLI could not start at all:
`train.py` imported Python's standard-library `io` module instead of `fngft.io`.
