# `fngft/model.py`

## Purpose

Defines the complete neural architecture.

## End-to-end computation

```text
history X(t-h:t)
        │
        ├──> MemoryTransformer ──> H_t
        │
        └──> OrderHead(current X_t, H_t)
                         │
                         └──> alpha, beta, kappa

history + H + alpha/beta/kappa
                │
                ▼
       FractionalClosure
                │
                ├── fractional spatial response
                ├── causal temporal memory
                └── learned residual/flux terms
                │
                ▼
       RealGridPhysicsCore
                │
                ▼
        next forecast state
```

## `MemoryTransformer`

Compresses the temporal history into a learned vector `H_t`.

The current prototype pools each grid field spatially before the Transformer, so it captures global history statistics rather than a full space-time token field.

## `OrderHead`

Predicts:

- `alpha(x,y)` in `[alpha_min, alpha_max]`
- `beta(x,y)` in `[beta_min, beta_max]`
- positive `kappa(x,y)`

The bounds are architectural, not just loss penalties.

## `FractionalClosure`

Applies the explicit fractional-memory operator and adds a learned residual.

For scalar channels, the correction is expressed in flux form and then differentiated, which is preferable to an unconstrained direct scalar increment.

## `FNGFTWeatherModel.step`

Performs one forecast step.

## `FNGFTWeatherModel.forward`

Performs autoregressive multi-step rollout:

```text
X_0 → X_1 → X_2 → ... → X_N
```

The forecast is therefore a trajectory, not a collection of independent one-step predictions.
