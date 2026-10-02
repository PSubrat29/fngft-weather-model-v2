# Real-data workflow

## Phase A — ingest

```text
data source
  ↓
xarray
  ↓
schema validation
```

## Phase B — construct the learning state

```text
source variable names
      ↓
canonical names
      ↓
[u,v,theta,q]
      ↓
[time,4,lat,lon]
```

## Phase C — remove leakage

Fit normalization statistics from training dates only.

## Phase D — learn temporal dependence

Construct history windows such as:

```text
X(t-7), X(t-6), ..., X(t)
```

## Phase E — forecast

The model computes:

```text
H_t
alpha_t, beta_t, kappa_t
fractional-memory response
explicit physics
closure
next state
```

## Phase F — rollout

The prediction becomes the next input:

```text
X(t) → X(t+1) → X(t+2) → ...
```

## Phase G — evaluate

Evaluate both forecast skill and structural behavior.

## Phase H — only then add SGS data

For theory validation, pair coarse and high-resolution simulations and supervise the unresolved response.
