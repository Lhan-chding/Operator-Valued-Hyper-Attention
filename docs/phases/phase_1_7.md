# Phase 1.7: OVHA-Controlled-v2 Gate

## Goal

Phase 1.7 follows the GPT Pro next-step report and blocks public benchmark work until the controlled data, sampling protocol and diagnostics are fixed.

## Implemented P0 Contract

- Training calls `sample_batch(..., episode_id=step - 1)` through `train_episode_base`.
- Evaluation uses fixed paired episode ids through `eval_episode_base + offset`; GPU configs use 128 to 512 eval episodes per split/family.
- Train and eval rows write `episode_id`, `batch_hash`, `context_hash` and `target_hash`.
- Controlled-v2 exposes `true_primitive_outputs_by_q`, `true_component_weight_by_q`, `true_operator_params`, modality reliability and history preference only through hidden diagnostics.
- Primitive order is aligned as `spectral`, `local`, `separable`.
- Router oracle MAE/KL/CE, oracle-router upper bound, oracle-adapter upper bound and memory-swap delta are written into eval rows.
- Public field `operator_transfer` sampling uses non-target demos from the same `operator_group_id` when groups are available.

## Controlled-v2 Families

```text
single_primitive_representable
single_primitive_spectral
single_primitive_local
single_primitive_separable
query_piecewise_router
context_identifiable_mixture
hyper_parameter_family
same_target_counterfactual
modality_reliability_conflict
history_session_preference
```

## Local Validation Boundary

Codex/Mac should run unit and tiny smoke tests only:

```bash
.venv/bin/python -m unittest tests.test_phase1_7_controlled_v2_protocol
```

The 5k sanity run is intentionally an Ubuntu/GPU handoff:

```bash
cd <repo>
git pull
PYTHON=.venv/bin/python bash scripts/run_phase1_7_controlled_v2_sanity.sh
```

The script defaults to GPU 3, while still allowing an explicit override:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHON=.venv/bin/python bash scripts/run_phase1_7_controlled_v2_sanity.sh
```

It uses unbuffered Python output and prints GPU/process status every 10 seconds. The same status is saved to:

```text
outputs/phase1_7/controlled_v2_sanity/process_monitor.log
```

It also prints code-level progress from inside the training/evaluation loops. Defaults:

```text
OVHA_PROGRESS_INTERVAL=25
OVHA_EVAL_PROGRESS_INTERVAL=128
```

For maximum verbosity during debugging:

```bash
OVHA_PROGRESS_INTERVAL=1 OVHA_EVAL_PROGRESS_INTERVAL=1 PYTHON=.venv/bin/python bash scripts/run_phase1_7_controlled_v2_sanity.sh
```

If only GPU 3 is available, use the single-GPU parallel runner to keep the A800 busier without touching other cards:

```bash
PYTHON=.venv/bin/python JOBS_PER_GPU=2 bash scripts/run_phase1_7_controlled_v2_sanity_gpu3_parallel.sh
```

This shards by `(seed, model)` into independent output directories under:

```text
outputs/phase1_7/controlled_v2_sanity_gpu3_parallel/
```

Default `JOBS_PER_GPU=2` is conservative for an 80GB A800. If `nvidia-smi` shows GPU memory and utilization remain low, try `JOBS_PER_GPU=3` or `JOBS_PER_GPU=4`; if host CPU load, dataloader time or other users become a problem, drop back to `1`.

To stop a foreground run, press `Ctrl-C`. If the Python process does not exit, run:

```bash
pkill -TERM -f 'train_torch_meta_operator.py|eval_torch_meta_operator.py'
pkill -KILL -f 'train_torch_meta_operator.py|eval_torch_meta_operator.py'  # only if TERM fails
```

Do not start PDEBench-mini, Mechanical-MNIST-mini, OpenFWI-mini or any larger public benchmark until the sanity run shows the expected component pattern:

- `ovha_full` wins on `query_piecewise_router` over no-router/simple-stack.
- `ovha_full` wins on `context_identifiable_mixture` over zero/global/shuffled/target-only memory ablations.
- `ovha_full` wins on `hyper_parameter_family` over `ovha_no_hyper_adapter`.
- `same_target_counterfactual` fails for target-only/no-memory and succeeds for full.

## Next Gate

If the 5k sanity run fails, inspect the oracle metrics before adding public data:

| Signal | Interpretation |
|---|---|
| oracle router succeeds, learned router fails | router training or router architecture issue |
| oracle adapter succeeds, learned adapter fails | hyper-adapter issue |
| oracle router and adapter fail | primitive/generator mismatch |
| train low but eval high | sampling/split/overfit issue |
| no-memory wins | context is not necessary or memory encoder is harmful |
