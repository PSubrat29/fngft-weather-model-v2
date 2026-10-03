# Real-data contract

## Minimum schema

One common regular latitude/longitude grid:

```text
(time, lat, lon)        dimension names are configurable (e.g. valid_time, latitude, longitude)
```

and four fields mapped to the canonical channels:

| Channel | Meaning | Units |
|---|---|---|
| `u` | eastward wind | **m/s** (required: the physics advects with it) |
| `v` | northward wind | **m/s** |
| `theta` | temperature or potential temperature | any (K or °C) |
| `q` | humidity (specific or relative) | any consistent unit |

## Example

A dataset containing `u10, v10, t2m, q2m` is mapped with:

```yaml
variables:
  u: u10
  v: v10
  theta: t2m
  q: q2m
```

## Files

`data.source` may be one file, a folder (all `.nc/.nc4/.grib/.grb/.grib2/.grb2` files are combined
by their coordinates), a glob pattern, or a Zarr store. Archives split by time (one file per
year/month/day) and by variable (one file per variable) work directly, for training and for
`forecast-latest`.

## Grid

- regular spacing; latitude ascending or descending (sorted automatically)
- global (0..360 or -180..180) or regional
- `region` crops an area, `coarsen` block-averages to reduce memory
- Gaussian/irregular/curvilinear grids must be regridded to a regular lat/lon grid first

## Time

Time must be strictly increasing with a fixed step. Missing timestamps (gaps that are whole multiples
of the step) are allowed: windows crossing a gap are skipped. The model advances by one data step per
rollout step; `dt_hours` is inferred from the data.

## Missing data

Default (`missing_values: error`): any NaN/inf stops with a message. `missing_values: interpolate`
fills linearly in time per grid point and uses the training-period mean for points that are never
valid. Document the choice for each experiment.

## Vertical data

For a multi-level dataset set `level_dim` and `level_value` to choose one level. The value must match
a level within 1% (in the file's units: pressure in Pa needs `85000` for 850 hPa); the chosen level is
printed and stored in the checkpoint.
Variables without the level dimension (e.g. 2 m fields) are kept as they are. Other extra dimensions
must have size 1 (they are squeezed) or be removed beforehand.

## Memory

Training loads the selected period into memory as float32: `time × 4 × lat × lon × 4 bytes`, and the
peak during loading and standardization is about 2× that (e.g. 5 years hourly at 1° global ≈
43 800 × 4 × 181 × 360 × 4 B ≈ 46 GB state, ~90 GB peak). Use 6-hourly data, `region`, `coarsen`
(applied chunk by chunk, so the full-resolution field is never loaded at once) or shorter periods to
fit your machine. `forecast-latest` reads only the most recent time steps.
