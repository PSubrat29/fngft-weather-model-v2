# `fngft/physics.py`

## Purpose

Provides the explicit known-physics branch used by the hybrid model.

## Units

The physics runs in **physical units**: the model converts its standardized state back with the
training mean/std, applies the physics, and standardizes again. Winds must therefore be in m/s.
(v0.2 applied advection to z-scores, which moved fields with the wrong speeds and even the wrong
direction.)

## One step

1. **Semi-Lagrangian advection** of u, v, theta and q: departure points are traced back along the
   local wind and fields are interpolated there. It is stable for any Courant number, so 0.25° hourly
   data or coarse 6-hourly data with fast jets do not blow up (the v0.2 forward-Euler centred scheme
   is unconditionally unstable for advection).
2. **Coriolis** (optional, `coriolis_scale`): exact rotation of the wind by the angle `f·dt`
   (energy conserving).
3. **Stratification** `dv -= s · dθ/dy` and **scalar diffusion**, explicit.

Longitude is periodic for global grids and one-sided/clamped for regional grids; latitude edges are
clamped.

## Why Coriolis is off by default

The state contains no pressure or geopotential, so there is no pressure-gradient force to balance
Coriolis. Real winds are close to geostrophic balance, so rotating them by `f·dt` every step rotates
balanced flow by ~100° per 6 h at mid-latitudes. On real ERA5 data (850 hPa, 6-hourly) this made the
model 3× worse than persistence (validation MSE 0.25 vs 0.08), while the same model without the
Coriolis term was 32% better than persistence after 3 short epochs. Enable it only together with a
pressure/geopotential channel.

## Scope boundary

This is **not** a numerical weather prediction dynamical core. It does not solve the primitive
equations, moist thermodynamics, radiation, boundary layer, cloud microphysics, land surface, ocean
coupling, or data assimilation. Interpretation: `real-data hybrid research core`, not
`operational NWP solver`.

## Upgrade path

Replace `RealGridPhysicsCore` with a validated atmospheric dynamical core while preserving the
fractional closure interface.
