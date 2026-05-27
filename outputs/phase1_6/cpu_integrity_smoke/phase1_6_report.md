# Phase 1.6 Report

## Environment

- torch_available: True
- device: cpu
- config_hash: 29610294f65a3314

## Checkpoint Integrity

| model | seed | checkpoint_loaded | train_steps | final_train_relL2 | eval_relL2 |
|---|---:|---|---:|---:|---:|
| ovha_full | 41 | True | 20 | 0.895557 | 1.123621 |
| transformer_only | 41 | True | 20 | 1.027001 | 1.382477 |

## Controlled Stress Tasks

| family | ovha_full | best_single | vector_big | no_memory | no_router | no_adapter | conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| query_piecewise_composition_family | 1.123621 |  |  |  |  |  | no claim; needs A800 multi-seed evidence |

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

Protocol Go for A800 execution: checkpoint-loaded CPU/integrity path is wired; Phase 2 scientific Go still requires GPU/public benchmark evidence.
