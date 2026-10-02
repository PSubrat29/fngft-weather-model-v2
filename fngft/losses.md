# `fngft/losses.py`

## Purpose

Defines the scientific and predictive objectives used during real-data training.

## Current real-data objective

```text
forecast MSE
+ 0.10 · gradient consistency
+ 0.02 · spectral-energy consistency
+ 0.01 · alpha/beta smoothness regularization
```

All terms are computed in standardized units. During multi-step rollout training the loss is averaged
over the rollout steps.

## Why not supervise alpha and beta with arbitrary numbers?

Real observations do not contain ground-truth alpha/beta labels. The real-data model therefore uses
structural regularization until an empirical SGS target is constructed.

## SGS extension

`fractional_operator_consistency_loss` is provided for high-resolution LES/DNS paired data. It
compares the learned closure directly to the empirically inferred subgrid tendency/flux — the route
for testing whether the learned fractional operator corresponds to unresolved atmospheric dynamics.
