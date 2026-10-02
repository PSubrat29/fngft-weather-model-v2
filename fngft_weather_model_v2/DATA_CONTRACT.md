# Real-data contract

## Minimum schema

The current prototype requires one common regular grid:

```text
(time, lat, lon)
```

and four fields:

```text
u
v
theta
q
```

## Example source dataset

A dataset may contain:

```text
u10
v10
t2m
q2m
```

and the configuration maps:

```yaml
variables:
  u: u10
  v: v10
  theta: t2m
  q: q2m
```

## Missing data

The current implementation requires finite values after loading. Missing-value handling must therefore happen before training, with the imputation or quality-control method documented.

## Time

Time must be strictly increasing and approximately regular. The model advances by one configured fixed time step per rollout step.

## Vertical data

For a 3-D dataset, configure `level_dim` and `level_value` to choose a single vertical level for this prototype. Full multi-level learning is a future extension.
