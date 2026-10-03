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

## Convolutions on the sphere

The 3×3 convolutions (`GridConv2d`) wrap around in longitude on global grids and zero-pad in latitude,
so a forecast does not depend on where the 0/360° seam lies (in 0.3.0 a temperature discontinuity
grew along the prime meridian). On regional grids both directions are zero-padded.

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

## Boundaries, humidity floor and multi-day stability

Multi-day forecasts feed each prediction back as input. On real ERA5 the learned closure fed back
unstably from the outermost latitude rows (where convolutions see padding and the physics is least
accurate) and v0.3.0 forecasts reached thousands of m/s within 2–5 days. Three safeguards fix this:

- `closure_boundary_rows` (default 1): the closure is switched off on the outermost latitude rows, and
  on the outermost columns of non-periodic (regional) grids.
- `boundary_mode` (default `persistence`): those rows keep their last state, i.e. a fixed boundary
  condition; `physics` applies the physics branch alone there.
- `q_min` (default 0): q is floored at this physical value after every step (set `null` if q is, for
  example, a dewpoint in °C).

Training with `rollout_steps: 2` (the default; about 6 for hourly data) further reduces drift. Measured on four real ERA5
grids (64×32 with and without poles, 240×121 with poles at 6-hourly and hourly steps), 10-day
(6-hourly) and 4-day (hourly) forecasts stay physically plausible (max wind 35–80 m/s, q ≥ 0); without
the boundary mask the same models exceed 150 m/s after 2–9 days. See `AUDIT_REPORT.md`.

`realtime.run_forecast` additionally flags any step that leaves the plausible range.

## Normalization buffers

`state_mean` / `state_std` are saved in the state dict. `set_normalization` must be called before
training (done by `train.py`); `build_model` restores them from a checkpoint.

## `FNGFTWeatherModel.forward`

Autoregressive rollout `X_0 → X_1 → … → X_N`; the forecast is a trajectory.
