# `fngft/io.py`

## Purpose

Provides the dataset boundary between real weather files and the ML pipeline.

## Supported sources

- one NetCDF file (`.nc`, `.nc4`), opened with xarray (netCDF4 / h5netcdf / scipy engines)
- a directory or glob pattern of files, combined with `xarray.open_mfdataset` by their coordinates:
  time-split archives (one file per year/month/day), one-variable-per-file archives, or both
- a Zarr store, local or remote (for example `gs://…` with `gcsfs` installed)
- GRIB/GRIB2 through an optional backend such as `cfgrib` + ecCodes

An existing path always wins over glob interpretation, so folder names such as `ERA5 [India]` work.

## `resolve_sources`

Expands `data.source` into the list of files (or the single store) to open.

## `open_weather_dataset`

Opens the configured dataset lazily (no data is read yet). Multiple files are combined with
`join="outer"`, so variables whose files cover slightly different times or levels still combine.

## `inspect_dataset`

Applies every configured selection (level, region, coarsen) through `preprocess.select_dataset` and
returns a `DatasetProfile` of the data the model will actually see: time coverage, step, missing
timestamps (`time_gaps`), grid size/spacing/extent, whether longitude is global and periodic, the
selected level, variable units and the number of missing values. With `missing_values: error` any
NaN fails here instead of later in training. `latest=True` checks only the most recent 24 time steps.

## Real-time behavior

The loader does not poll. `forecast-latest` (a scheduler, or the API) opens the whole source lazily
and uses its most recent complete time steps, so the newest data is used regardless of file names,
file layout or modification times.
