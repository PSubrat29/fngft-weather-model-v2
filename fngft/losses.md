# `fngft/losses.py`

## Purpose

Defines the scientific and predictive objectives used during real-data training.

## Current real-data objective

```text
forecast MSE
+ gradient consistency
+ spectral-energy consistency
+ alpha/beta smoothness regularization
```

## Why not supervise alpha and beta with arbitrary numbers?

Real observations normally do not contain ground-truth alpha/beta labels. The previous synthetic prototype could compare them with known teacher values; real-data training cannot do that honestly unless independent closure labels are available.

Therefore the real-data model uses structural regularization until an empirical SGS target is constructed.

## SGS extension

`fractional_operator_consistency_loss` is provided for high-resolution LES/DNS paired data.

That loss compares the learned closure directly to the empirically inferred subgrid tendency/flux. This is the correct route for testing whether the learned fractional operator corresponds to actual unresolved atmospheric dynamics.
