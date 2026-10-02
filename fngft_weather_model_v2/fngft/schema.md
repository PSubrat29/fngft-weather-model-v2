# `fngft/schema.py`

## Purpose

Defines and validates the real-data contract before model training or forecasting.

## Required dataset structure

The minimum xarray dataset must contain:

```text
time
lat
lon
u-variable
v-variable
theta-variable
q-variable
```

The source variable names can be different; `DataConfig.variables` maps them to the four canonical channels.

## Validation steps

1. Check required dimensions.
2. Check required variables.
3. Check that every state variable uses time/latitude/longitude dimensions.
4. Check strictly increasing time.
5. Check approximately regular time spacing.
6. Check one-dimensional latitude and longitude coordinates.
7. Check coordinate finiteness.

## Reason

Weather ML can silently fail when time steps are irregular, dimensions are transposed, or a variable contains missing values. Validation therefore happens before tensors reach PyTorch.
