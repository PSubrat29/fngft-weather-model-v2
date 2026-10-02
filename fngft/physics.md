# `fngft/physics.py`

## Purpose

Provides the explicit known-physics branch used by the hybrid model.

## Why it exists

The architecture is intended to learn unresolved physics rather than replace every equation with a black box.

## Current real-grid proxy

The physics branch currently includes:

- spherical-coordinate horizontal gradients on a regular lat/lon grid
- advection by `u` and `v`
- latitude-dependent Coriolis parameter
- a simple stratification term
- diffusion regularization

## Important scope boundary

This is **not** a complete numerical weather prediction dynamical core. It does not solve the full compressible/hydrostatic primitive equations, moist thermodynamics, radiation, boundary layer, cloud microphysics, land surface, ocean coupling, or data assimilation.

The correct interpretation is:

`real-data hybrid research core`

not:

`operational NWP solver`.

## Upgrade path

The final global architecture should replace `RealGridPhysicsCore` with a validated atmospheric dynamical core while preserving the fractional closure interface.
