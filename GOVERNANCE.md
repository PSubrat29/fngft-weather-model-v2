# FNGFT-AI prototype governance

## 1. Data provenance

Every experiment should identify:

- source provider
- dataset name/version
- retrieval time
- time coverage
- grid resolution
- variables and units
- preprocessing/QC version
- missing-data policy

## 2. Model provenance

Every checkpoint should be tied to:

- source code revision
- configuration file
- model configuration
- normalization statistics
- random seed
- training hardware
- dependency environment

Every checkpoint already stores the model configuration (including the inferred time step), data mapping and units, normalization statistics, grid, training configuration and per-epoch history, plus provenance: creation time, package and PyTorch versions, git revision, data period and device. Store the `evaluate` JSON report next to it.

## 3. Scientific promotion gates

A checkpoint should not be promoted because RMSE improved alone.

Minimum evidence should include:

- held-out temporal skill that beats persistence and climatology (`python -m fngft evaluate --steps N`)
- spatial/gradient skill
- spectral behavior
- long-rollout stability
- alpha/beta reproducibility
- cross-regime results
- cross-resolution results
- SGS closure consistency when available

## 4. Operational release gate

Before a forecast is consumed downstream, validate:

- finite outputs
- expected grid dimensions
- expected time horizon
- valid variable ranges after inverse normalization
- model/checkpoint identity
- input provenance
- uncertainty/quality status

## 5. Human accountability

The research prototype must not be treated as an autonomous replacement for meteorological review in high-impact weather decisions.
