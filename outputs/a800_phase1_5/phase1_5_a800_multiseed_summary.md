# Phase 1.5 A800 Multi-Seed Summary

## Run Set

- Host: Ubuntu 22.04 / Linux 5.15.0-105-generic x86_64
- GPU target: NVIDIA A800 80GB PCIe via `CUDA_VISIBLE_DEVICES=4`
- Python: 3.10.12
- Torch: 2.4.1+cu118
- Config family: `configs/phase1_5_gpu_main.json`
- Seeds: 31, 32, 33, 34
- Per-seed training: 10,000 steps
- Per-seed evaluation rows: 640
- Per-seed diagnostics rows: 640
- Model parameters for `ovha_full`: 489,429

## Aggregate Relative L2

Mean `relative_l2` across all splits, families and four seeds:

| model | mean relL2 |
|---|---:|
| local_only | 1.137794 |
| separable_only | 1.188397 |
| ovha_no_query_router | 1.318062 |
| ovha_no_memory | 1.318159 |
| ovha_no_query_adapter | 1.321385 |
| ovha_full | 1.324814 |
| ovha_random_router | 1.412323 |
| oracle_metadata_upper_bound | 1.610114 |
| spectral_only | 2.191473 |
| perceiver_io_style | 2.281816 |
| transformer_only | 2.379144 |
| ovha_vector_value_only | 2.382044 |
| ovha_no_hyper_adapter | 2.470776 |
| simple_stack | 2.480682 |
| mlp_expert_moe | 3.046375 |
| icon_style | 3.100777 |

## Primary Comparator Result

`ovha_full` beats the primary neural comparators in every downloaded seed:

| seed | ovha_full | transformer_only | ovha_vector_value_only | simple_stack |
|---:|---:|---:|---:|---:|
| 31 | 1.264604 | 2.071187 | 2.079338 | 2.273187 |
| 32 | 1.312444 | 2.256934 | 2.258483 | 2.384791 |
| 33 | 1.410603 | 2.877837 | 2.883364 | 2.824243 |
| 34 | 1.311604 | 2.310616 | 2.306992 | 2.440506 |

Conclusion: provisional multi-seed Go for the metadata-free OVHA-vs-vector/simple-stack comparison.

## Split-Level Pattern

| split | ovha_full | transformer_only | ovha_vector_value_only | simple_stack |
|---|---:|---:|---:|---:|
| confusable_context | 1.271525 | 2.063092 | 2.063749 | 2.293825 |
| context_size_sweep | 1.358843 | 2.444992 | 2.443838 | 2.536942 |
| family_holdout | 1.506526 | 3.175015 | 3.182278 | 3.058123 |
| iid | 1.318135 | 2.383235 | 2.398739 | 2.495218 |
| noisy_context | 1.345728 | 2.488675 | 2.486027 | 2.599547 |
| num_demos_sweep | 1.297356 | 2.061406 | 2.060758 | 2.289535 |
| parameter_holdout | 1.169632 | 2.013357 | 2.015485 | 2.002875 |
| resolution_transfer | 1.330766 | 2.403375 | 2.405481 | 2.569388 |

## Family-Level Pattern

| family | ovha_full | transformer_only | ovha_vector_value_only | simple_stack |
|---|---:|---:|---:|---:|
| compositional_mixed_family | 0.937724 | 0.980507 | 0.982567 | 1.130436 |
| local_green_family | 0.938677 | 0.712781 | 0.712107 | 0.949355 |
| nonlinear_family | 0.932799 | 0.727539 | 0.727072 | 0.877733 |
| separable_lowrank_family | 2.045927 | 4.518013 | 4.524624 | 5.273152 |
| spectral_family | 1.768943 | 4.956876 | 4.963851 | 4.172732 |

The strongest OVHA advantage is on the harder separable-lowrank and spectral families. Vector baselines remain better on local-green and nonlinear families under this compact config.

## Caveats

- The current A800 config is a compact validation run, not an A800-saturating large run.
- `local_only`, `separable_only`, and several no-query/no-memory variants are close to or slightly better than `ovha_full` in aggregate. This means the run supports the operator-valued attention comparison, but it does not yet prove every architectural component is necessary.
- `oracle_metadata_upper_bound` is a diagnostic comparator only and remains excluded from the metadata-free claim.
- These runs are still synthetic operator-zoo evidence, not a real PDE/material benchmark.

## Decision

Phase 1.5 status should be recorded as:

Provisional Go for multi-seed metadata-free GPU validation against vector attention and simple-stack baselines. Continue to Phase 2 planning, while keeping component-necessity ablations and real benchmark validation as open requirements.
