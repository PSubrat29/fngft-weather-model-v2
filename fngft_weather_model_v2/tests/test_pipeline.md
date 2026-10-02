# `tests/test_pipeline.py`

## Purpose

Integration smoke test for the real-data path.

## What it does

1. Creates a small synthetic NetCDF file with realistic xarray dimensions.
2. Uses source-to-canonical variable mapping.
3. Validates the dataset.
4. Extracts and standardizes the four state channels.
5. Builds time windows.
6. Runs the FNGFT-AI model.
7. Checks output shape.
8. Checks that alpha and beta remain inside configured bounds.

## What it does not prove

Passing this test does not establish meteorological forecast skill or validate FNGFT as a physical theory. It proves that the real-data plumbing and model interfaces are internally consistent.
