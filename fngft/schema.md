# `fngft/schema.py`

## Purpose

Defines and validates the real-data contract before model training or forecasting.

## Required dataset structure

```text
time, lat, lon dimensions (names configurable)
four variables mapped to u, v, theta, q, each containing time/lat/lon
```

## Validation steps

1. Required dimensions exist (the error lists the available ones).
2. Required variables exist (the error lists the available ones).
3. Each state variable contains time/lat/lon; any other dimension must be the configured
   `level_dim` or have size 1. An `expver` dimension (old-CDS ERA5 + ERA5T mix) gets a specific
   message explaining how to merge it.
4. Time decodes to datetime64, is strictly increasing and has a fixed step. Missing timestamps
   (gaps that are whole multiples of the step) are allowed and counted; other irregular spacing is
   rejected.
5. Latitude/longitude are 1-D, finite, strictly monotonic (ascending or descending) and regular.

The profile also reports whether longitude spans the full circle (`longitude_global`) and the units
attribute of each variable.

## Reason

Weather ML can silently fail when time steps are irregular, dimensions are transposed, or a variable
contains missing values. Validation therefore happens before tensors reach PyTorch.
