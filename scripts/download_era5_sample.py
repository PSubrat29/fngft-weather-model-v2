"""Download a small real ERA5 sample from the public WeatherBench2 archive (Google Cloud Storage).

Writes data/era5_sample_850hPa_2018_2020.nc (~150 MB): u, v, temperature and specific humidity
at 850 hPa, 6-hourly, 64 x 32 global grid, 2018-2020. Requires:  pip install gcsfs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
VARIABLES = ["u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/era5_sample_850hPa_2018_2020.nc")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2020-12-31T18:00")
    parser.add_argument("--level", type=int, default=850)
    args = parser.parse_args()
    ds = xr.open_zarr(STORE, storage_options={"token": "anon"})
    sub = ds[VARIABLES].sel(time=slice(args.start, args.end), level=[args.level])
    print(f"downloading {dict(sub.sizes)} ...")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.load().to_netcdf(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
