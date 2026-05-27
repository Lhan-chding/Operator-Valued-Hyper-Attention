# Phase 1.6: Learned-Evaluation Integrity + Benchmark Protocol Foundation

## Goal

Phase 1.6 fixes the learned-evaluation integrity gap before any Phase 2 claim:

- evaluation must load trained checkpoints by default
- every reported main/ablation model must be trained under the same budget
- synthetic data is only a controlled mechanism stress test
- public/real benchmark protocol is required for external validity

Phase 2 is blocked until Phase 1.6 passes the Go criteria below.

## Direction Guardrail

Phase 1.6 is not redefining OVHA as a PDE/FNO/DeepONet project. PDEBench, FNO classic and Mechanical MNIST are used because they are credible early evidence surfaces for operator learning, not because the method is limited to those domains.

The long-term target remains:

```text
Operator-Valued Attention
+ Hyper-Operator Transformer
+ Operator Memory Transformer
= context-conditioned composition of multiple operators across domains/modalities
```

This makes the work a candidate multi-domain / multimodal meta-operator framework: different modalities become different input objects, query spaces and output spaces under the same episode protocol. Phase 1.6 only verifies training integrity and mechanism stress before expensive A800 runs; it does not claim that multimodal capability is already proven.

## Scope

Included:

- checkpoint-loaded training/evaluation artifacts
- per-model/per-seed train metrics, eval metrics, diagnostics and checkpoints
- controlled analytical stress families for component necessity
- `BenchmarkRegistry` and public benchmark loader API
- `FieldToEpisodeAdapter` for full-field-to-episode conversion
- CPU integrity smoke config and A800 scripts/configs
- Phase 1.6 summarizer

Excluded from local Codex/Mac execution:

- long multi-model training
- multi-seed GPU benchmark matrices
- full PDEBench/FNO/Mechanical MNIST training
- high-resolution or large-batch experiments

## Artifact Contract

Training writes:

```text
outputs/<run>/checkpoints/<model_name>/seed_<seed>/model.pt
outputs/<run>/train_metrics/<model_name>/seed_<seed>.jsonl
```

Evaluation writes:

```text
outputs/<run>/eval_metrics/<model_name>/seed_<seed>.jsonl
outputs/<run>/diagnostics/<model_name>/seed_<seed>.jsonl
```

Every checkpoint contains `model_name`, `model_state_dict`, `config`, `seed`, `train_steps`, `config_hash`, `parameter_count` and optional `git_commit`.

Every eval row must include `checkpoint_loaded`, `checkpoint_path`, `checkpoint_train_steps`, `model_name`, `seed` and `eval_seed`.

## Benchmark Ladder

| Level | Purpose | Data | Role |
|---|---|---|---|
| L0 | pipeline correctness | tiny CPU synthetic | code validation only |
| L1 | mechanism stress | controlled analytical synthetic | component necessity |
| L2 | standard operator benchmark | PDEBench/FNO subset | neural-operator comparison |
| L3 | material/mechanics benchmark | Mechanical MNIST-like fields | cross-material evidence |
| L4 | large-scale | high-res PDE/irregular geometry | later main/appendix |
| L5 | cross-modal dense-query | image/trajectory query | later Phase 2/3 extension |

Benchmark names are evidence labels, not method boundaries. A later image-to-field or trajectory-query benchmark should reuse the same context/query/operator interface rather than becoming a separate CV or robotics demo track.

## Required Model Groups

Main:

```text
ovha_full
transformer_only
ovha_vector_value_big
simple_stack
mlp_expert_moe
best single primitive among local/separable/spectral
```

Ablation:

```text
ovha_no_memory
ovha_no_hyper_adapter
ovha_no_query_router
ovha_no_query_adapter
ovha_random_router
```

Oracle diagnostics are separate and never part of the metadata-free main table.

## Local Validation

Mac/Codex should run only:

```bash
.venv/bin/python -m unittest tests.test_phase1_6_eval_loads_checkpoint
.venv/bin/python -m unittest tests.test_phase1_6_benchmark_protocol
PYTHON=.venv/bin/python bash scripts/run_phase1_6_cpu_integrity_smoke.sh
```

## A800 Handoff

After pulling this branch on Ubuntu/A800:

```bash
cd <repo>
git pull
bash scripts/run_phase1_6_gpu_integrity_short.sh
bash scripts/run_phase1_6_gpu_component_main.sh
bash scripts/run_phase1_6_gpu_public_pilot.sh
python scripts/summarize_phase1_6.py --root outputs/phase1_6
```

## Go / No-Go

Phase 1.6 Go requires:

- missing checkpoints fail when `require_checkpoint=true`
- all main/ablation models have trained checkpoint-loaded eval rows
- controlled stress shows OVHA-full beating best single primitive and vector baseline
- `no_memory`, `no_query_router` and `no_hyper_adapter` degrade on their target stress families
- router diagnostics show query-dependent specialization on controlled stress
- public benchmark pilot runs at least two PDEBench/FNO-style datasets
- reports separate controlled stress, public benchmark, oracle and untrained diagnostics

No-Go if evaluator returns to random initialization, single primitives beat OVHA-full without explanation, ablations do not degrade, metadata leaks into model input, fixed eval episodes are not reproducible, or only aggregate means are reported.
