# Phase 1.6 Report

## Environment

- torch_available: True
- device: cpu
- config_hash: 9f0a69dfeb8a323f

## Checkpoint Integrity

| model | seed | checkpoint_loaded | train_steps | final_train_relL2 | eval_relL2 |
|---|---:|---|---:|---:|---:|
| ovha_full | 41 | True | 20 | 1.242307 | 1.006336 |
| transformer_only | 41 | True | 20 | 1.473022 | 1.118481 |

## Controlled Stress Tasks

| family | ovha_full | best_single | vector_big | no_memory | no_router | no_adapter | conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| query_piecewise_composition_family | 1.006336 |  |  |  |  |  | incomplete; baseline missing |

## Diagnostic Signals

| model | family | entropy | memory_norm | adapter_norm | primitive_load |
|---|---|---:|---:|---:|---|
| ovha_full | query_piecewise_composition_family | 1.096400 | 2.109006 | 12.637806 | local=0.341890, separable=0.354851, spectral=0.303258 |
| transformer_only | query_piecewise_composition_family | 0.000000 | 2.631976 | 11.915462 | vector_value=1.000000 |

## Public Benchmark Pilot

| dataset | split | ovha_full | best_baseline | delta | win? |
|---|---|---:|---:|---:|---|
| not_available | not_available |  |  |  | public benchmark loaders prepared; full data run pending |

## Oracle / Untrained Diagnostics

| model | rows | checkpoint_loaded_rows | note |
|---|---:|---:|---|
| none | 0 | 0 | all eval rows were checkpoint-loaded main/ablation rows |

## Dataset Card Appendix

- PDEBench subset: requires external download/cache; planned splits are iid, parameter_holdout, resolution_transfer, context sweeps, noisy/confusable context where supported.
- FNO classic subset: requires external canonical Burgers/Darcy/Navier-Stokes cache; aligned to neural-operator baselines.
- Mechanical MNIST small: requires external mechanics/material cache; used for parameter/material holdout evidence.

## Go / No-Go

Protocol Go only: checkpoint path is wired, but controlled-stress evidence is incomplete or mixed; diagnose before treating this as a scientific win.
