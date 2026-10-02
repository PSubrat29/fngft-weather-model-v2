# `fngft/model.py`

## Purpose

Defines the complete neural architecture.

## End-to-end computation

```text
history X(t-h:t)  (standardized)
        │
        ├──> MemoryTransformer ──> H_t
        │
        └──> OrderHead(current X_t, H_t) ──> alpha, beta, kappa

history + H + alpha/beta/kappa ──> FractionalClosure ──> closure increment
                                    ├── fractional spatial response
                                    ├── causal temporal memory
                                    └── learned residual / flux terms

X_t ──> to physical units ──> RealGridPhysicsCore ──> standardize ──> physics forecast

next state = physics forecast + closure_scale · closure
```

## `MemoryTransformer`

Spatially pools each history frame, adds a learned **positional embedding** (without it attention
cannot tell the order of the history), and encodes the sequence into `H_t`.

## `OrderHead`

Predicts `alpha(x,y) ∈ [alpha_min, alpha_max]`, `beta(x,y) ∈ [beta_min, beta_max]` and positive
`kappa(x,y)`. The bounds are architectural, not loss penalties.

## `FractionalClosure`

Applies the fractional-memory operator (a damping term `-fractional_scale · kappa · response`) and a
learned residual. Scalar channels (theta, q) are corrected in flux form: a learned flux is
differentiated (grid-index divergence), which keeps the correction conservative.

The closure is a per-step increment in standardized units. The residual output layers are
initialised to zero, so the untrained model equals physics + fractional damping and training grows
the correction from there.

### Scales

`closure_scale` and `residual_scale` default to 1. In v0.2 they were 0.05 and 0.02 and the theta/q
flux divergence was additionally divided by the Earth radius, so the learned part contributed ~10⁻³
(momentum) and ~10⁻⁹ (scalars) per step — with Adam this cannot be learned in any practical number of
steps. On ERA5 the v0.2 settings gave a validation MSE 14× worse than persistence; the current
defaults beat persistence within the first epoch.

## Normalization buffers

`state_mean` / `state_std` are saved in the state dict. `set_normalization` must be called before
training (done by `train.py`); `build_model` restores them from a checkpoint.

## `FNGFTWeatherModel.forward`

Autoregressive rollout `X_0 → X_1 → … → X_N`; the forecast is a trajectory.
