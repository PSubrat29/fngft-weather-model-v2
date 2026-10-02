# `fngft/io.py`

## Purpose

Provides the dataset boundary between real weather files and the ML pipeline.

## Supported sources

- one NetCDF file (`.nc`, `.nc4`), opened with xarray (netCDF4 / h5netcdf / scipy engines)
- a directory or glob pattern of files: combined along time with `xarray.open_mfdataset`
  (yearly or monthly files of a historical archive work directly)
- a Zarr store, local or remote (for example `gs://…` with `gcsfs` installed)
- GRIB/GRIB2 through an optional backend such as `cfgrib` + ecCodes

## `discover_latest_source`

If `data.source` is a directory, returns the newest supported file by modification time. A `.zarr`
directory is treated as one store, not as a folder of files.

## `open_weather_dataset`

Opens the configured dataset. With `latest=True` only the newest file is opened (real-time mode);
otherwise all files are combined. Clear errors are raised for a missing path or an empty directory.

## `inspect_dataset`

Runs schema validation and returns a `DatasetProfile`: time coverage, step, number of missing
timestamps (`time_gaps`), grid size and spacing, whether longitude is global, and variable units.

## Real-time behavior

The loader does not poll. A scheduler calls `forecast-latest` when a new file arrives, which keeps
ingestion deterministic and replayable.
