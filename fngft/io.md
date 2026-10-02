# `fngft/io.py`

## Purpose

Provides the dataset boundary between real weather files and the ML pipeline.

## Supported sources

- NetCDF through xarray
- Zarr through xarray
- GRIB/GRIB2 through an optional GRIB backend such as `cfgrib`/ecCodes

## `discover_latest_source`

If `data.source` points to a directory, the newest supported weather file is selected by filesystem modification time. This supports a simple real-time file-drop workflow.

## `open_weather_dataset`

Opens the configured dataset and returns an xarray `Dataset`.

No neural processing happens here.

## `inspect_dataset`

Runs schema validation and returns a `DatasetProfile` containing:

- time coverage
- grid dimensions
- variable mapping
- time step
- geographic extent

## Real-time behavior

The loader does not continuously poll. A higher-level process calls it when a new forecast cycle is available. This keeps ingestion deterministic and makes replay/testing possible.
