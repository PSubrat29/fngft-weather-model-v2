# `fngft/cli.py`

## Purpose

Provides a single command-line interface to the real-data workflow.

## Commands

### Inspect

```bash
python -m fngft.cli inspect --config configs/real_data.yaml
```

Validates and profiles the dataset.

### Train

```bash
python -m fngft.cli train --config configs/real_data.yaml
```

Trains and writes the configured checkpoint.

### Evaluate

```bash
python -m fngft.cli evaluate --config configs/real_data.yaml --checkpoint artifacts/fngft_real.pt --split test
```

### Forecast newest data

```bash
python -m fngft.cli forecast-latest --config configs/real_data.yaml --checkpoint artifacts/fngft_real.pt --steps 4
```

The CLI keeps the operational path explicit and reproducible.
