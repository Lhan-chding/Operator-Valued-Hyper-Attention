# OVHA Benchmark Protocol

## Evidence Roles

Synthetic controlled data is for mechanism identification, not final evidence. It can show whether the router, memory and hyper-adapter respond to known hidden structure, because the generator can emit `true_component_weight_by_q` as an offline diagnostic.

Public/real benchmark data is required for external validity. PDEBench, FNO classic and mechanics/material datasets are early paths to claims about neural-operator performance outside the synthetic generator; they are not the boundary of the method.

DeepONet, FNO, PDEBench, material benchmarks, CV dense-query tasks and trajectory tasks should be treated as evidence surfaces. The method-level object is broader: a context-conditioned operator-valued transformer that can infer and compose operators across input/query/output spaces.

## Dataset Levels

| Level | Use | Required Treatment |
|---|---|---|
| L0 tiny CPU synthetic | smoke only | never report as method evidence |
| L1 controlled stress | component necessity | report true-vs-learned router diagnostics |
| L2 PDEBench/FNO | standard operator benchmark | report iid/OOD/resolution/context splits |
| L3 mechanics/material | cross-material evidence | report material/parameter holdout |
| L4/L5 large/cross-modal | future extension | do not use for Phase 1.6 claims, but keep interface-compatible |

## Splits

Each benchmark should define:

- `iid`
- `parameter_holdout`
- `resolution_transfer`
- `context_size_sweep`
- `num_demos_sweep`
- `noisy_context`
- `confusable_context` or `sparse_context`

If a dataset cannot support a split, the report must mark it as `not_supported`.

## Episode Construction

Full field pairs are adapted into OVHA episodes:

```text
context demonstrations D = {(u_i, q_i, y_i)}
target input u*
target query q*
prediction y*(q*)
```

The adapter may sample uniformly, stratified, boundary-focused or randomly. Evaluation episodes must be deterministic/fixed so every model sees the same context and target query points.

Dataset labels, PDE parameters, boundary IDs and hidden latents are forbidden as metadata-free model inputs. They may appear only in oracle diagnostics or offline reports.

## Public Cache Format

Public benchmark training/eval reads local cache files only. It does not download raw benchmark archives and it must not silently fall back to synthetic data when a public cache is configured.

Set either `public_data_root` in the config or `OVHA_PUBLIC_BENCHMARK_ROOT` in the shell. The reader looks for per-family split files under one of:

```text
<public_data_root>/<family>/<split>.npz
<public_data_root>/<dataset>/<family>/<split>.npz
<public_data_root>/<dataset>/<split>.npz
```

Supported suffixes are `.npz`, `.npy`, `.h5` and `.hdf5`. The preferred cache schema is:

| Key | Shape | Meaning |
|---|---|---|
| `input_field` | `[samples, points, channels]` or flattenable grid | input/operator condition field |
| `output_field` | `[samples, points, channels]` or flattenable grid | target solution/response field |
| `coordinates` | `[points, coord_dim]` | query coordinates for the flattened field |
| `operator_group_id` | `[samples]`, optional | groups samples that share an operator for few-shot demos |

Accepted aliases include `u`/`input`/`inputs` for `input_field`, `y`/`output`/`solution` for `output_field`, and `coords`/`grid` for `coordinates`.

For the first public pilot, prepare at least:

```text
data/public_benchmark_cache/pdebench_burgers_1d/train.npz
data/public_benchmark_cache/pdebench_burgers_1d/iid.npz
data/public_benchmark_cache/pdebench_advection_1d/train.npz
data/public_benchmark_cache/pdebench_advection_1d/iid.npz
data/public_benchmark_cache/pdebench_darcy_2d/train.npz
data/public_benchmark_cache/pdebench_darcy_2d/iid.npz
data/public_benchmark_cache/pdebench_shallow_water_2d/train.npz
data/public_benchmark_cache/pdebench_shallow_water_2d/iid.npz
```

Additional split files such as `parameter_holdout.npz`, `resolution_transfer.npz`, `sparse_context.npz` and `confusable_context.npz` should be added before treating those rows as scientific evidence. If a split is aliased to `test`/`val`/`iid`, report that source split explicitly.

The same episode contract should later cover non-PDE modalities:

- image-to-field or image-to-density prediction: `u` is an image or latent field, `q` is a pixel/point/query coordinate, `y` is dense output
- trajectory-query prediction: `u` is history/context state, `q` is time/action/state query, `y` is future state or response
- sequence-to-field tasks: `u` is sequence or parameter object, `q` indexes output domain, `y` is structured response

These are examples, not separate method definitions. The benchmark protocol should keep the shared operator-memory and operator-valued attention abstraction visible.

## Metrics

Report by model, split, family/dataset and seed:

- relative L2
- MSE/RMSE
- median and p90 relative L2
- mean +/- std across seeds
- paired delta vs `ovha_full`
- win rate by split/family

OVHA-specific diagnostics:

- primitive entropy
- effective number of primitives
- router weight std over query and batch
- router true weight MAE/KL/CE for controlled stress only
- oracle-router and oracle-adapter upper-bound relative L2 for controlled stress only
- context/memory swap sensitivity
- memory and adapter norms

Every controlled eval row must include deterministic `episode_id`, `batch_hash`, `context_hash` and `target_hash` so train/eval repetition and leakage bugs are auditable.

## Reporting Boundary

Do not mix:

1. checkpoint-loaded main/ablation results
2. controlled stress diagnostics
3. public benchmark pilot results
4. oracle/untrained diagnostics

No synthetic toy result should be written as final proof of algorithmic effectiveness.

No public benchmark should be written as the only intended application domain. Reports should separate "evidence on this domain" from "definition of the OVHA architecture."
