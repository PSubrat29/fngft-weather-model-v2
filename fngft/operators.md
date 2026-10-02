# `fngft/operators.py`

## Purpose

Implements the two mathematical components that make the prototype FNGFT-specific:

1. variable-order spatial fractional response
2. causal temporal memory weighting

## Spatial operator

The field is mirrored across the latitude boundaries (and across the longitude boundaries for
regional grids) before the FFT. The spectral multiplier is

`|k̂|^alpha`,  with `|k̂|² = (k_y² + k_x²) / (2π²) ∈ [0, 1]`

i.e. wavenumbers normalised to the grid's Nyquist range. The response is therefore bounded and does not
grow with grid resolution (v0.2 used integer wavenumbers, giving multipliers up to ~10⁶ on a
0.25° grid). A bank of fixed orders between `alpha_min` and `alpha_max` is combined with positive
Gaussian weights derived from the local alpha field.

This is a numerical approximation, not an exact variable-coefficient fractional pseudodifferential
operator.

## Temporal memory

For historical lags `tau`, `w(tau) ∝ tau^(-beta)`, normalised across the history. All history steps
share one alpha field, so the spatial operator is applied to the whole history in one batched call.

## Limitation

The mirrored-FFT formulation is a regular-grid approximation. A production global implementation
needs a sphere-aware fractional operator.
