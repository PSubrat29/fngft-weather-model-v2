# Real-data workflow

## Phase A — ingest

```text
data source (file, folder, glob, Zarr)
  ↓
xarray
  ↓
schema validation      python -m fngft inspect --config CONFIG
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

Evaluate both forecast skill and structural behavior:

```bash
python -m fngft evaluate --config CONFIG --checkpoint CKPT --split test --steps 4 --output artifacts/eval_test.json
```

The report compares every variable and lead time with persistence and climatology. A model that does not beat
persistence has not learned anything useful yet.

## Phase H — only then add SGS data

For theory validation, pair coarse and high-resolution simulations and supervise the unresolved response.
