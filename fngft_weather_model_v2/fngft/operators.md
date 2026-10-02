# `fngft/operators.py`

## Purpose

Implements the two mathematical components that make the prototype FNGFT-specific:

1. variable-order spatial fractional response
2. causal temporal memory weighting

## Spatial operator

The prototype computes an FFT representation and applies a spectral multiplier proportional to:

`k^(alpha)`

A small bank of alpha basis operators is combined with positive weights derived from the learned local alpha field.

This is a numerical approximation, not an exact variable-coefficient fractional pseudodifferential operator.

## Temporal memory

For historical lags `tau`, the weighting behaves like:

`w(tau) ∝ tau^(-beta)`

with normalization across the available history.

## Real-data limitation

The FFT formulation is periodic in both spatial directions. Longitude periodicity is natural on a global longitude axis, but latitude is not periodic. Therefore this module is suitable for research on a regular patch/grid, but a production global implementation needs a sphere-aware fractional operator.
