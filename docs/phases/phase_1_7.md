# Phase 1.7: OVHA-Controlled-v2 Gate

## Goal

Phase 1.7 follows the GPT Pro next-step report and blocks public benchmark work until the controlled data, sampling protocol and diagnostics are fixed.

## Implemented P0 Contract

- Training calls `sample_batch(..., episode_id=step - 1)` through `train_episode_base`.
- Evaluation uses fixed paired episode ids through `eval_episode_base + offset`; GPU configs use 128 to 512 eval episodes per split/family.
- Train and eval rows write `episode_id`, `batch_hash`, `context_hash` and `target_hash`.
- Controlled-v2 exposes `true_primitive_outputs_by_q`, `true_component_weight_by_q`, `true_operator_params`, modality reliability and history preference only through hidden diagnostics.
- Controlled-v2 also keeps full hidden `true_operator_tensors` for non-public oracle diagnostics; these tensors are not part of model inputs.
- Primitive order is aligned as `spectral`, `local`, `separable`.
- `model_aligned` iid generation is constrained to the short-term sanity adapter domain: gain in `[0.9, 1.1]` and local lengthscale in `[0.08, 0.20]`.
- The separable model primitive uses the same fourth basis as the generator: `cos(2*pi*s) * cos(2*pi*q)`.
- Router oracle MAE/KL/CE, oracle-router upper bound, oracle-adapter upper bound, model true-param primitive oracle and memory-swap delta are written into eval rows.
- Eval rows now also expose the full oracle matrix: learned router + learned adapter, true router + learned adapter, learned router + true adapter, and true router + true adapter.
- Context memory tokens include primitive-aligned spectral, local and separable candidate outputs, residual features and sufficient-statistics evidence features.
- OVHA keeps primitive-specific memory slots and routes/adapts from shared primitive-conditioned memory instead of independent mean-pooled router/adapter heads.
- Query-conditioned router residuals are gated by context-prior entropy, so single-primitive episodes with a confident context route do not receive unnecessary per-query routing perturbations while query-piecewise episodes keep the residual path.
- Oracle route overrides are propagated into the hyper-adapter posterior features, not only into the final mixture weights. Stage-C sanity/main configs force oracle routing on single-primitive batches so mixed training cannot keep corrupting the specialist gates; router CE is computed from the learned router output, not the teacher-forced mixture weights.
- Hyper-adapter heads receive a direct per-primitive evidence channel (`ls_coeff`, residual energy and uncertainty) alongside pooled memory and router posterior features, so separable/spectral parameter inference can use the sufficient statistics already computed from public context triples. The local adapter also derives a `model_aligned`-only public-evidence lengthscale prior from amplitude-normalized local candidate Gram/correlation scores, gated by local router posterior, before applying learned residual corrections.
- `model_aligned` adapter params are episode-global by default. Spectral frequency/phase and local shift are disabled for `model_aligned`; spectral mode logits, separable rank logits, gain, bias and local lengthscale are expanded from episode-level predictions.
- Controlled-v2 adapter auxiliary losses supervise spectral mode KL, separable rank KL, gain/bias Huber, local lengthscale log Huber, primitive output gap, oracle-routed prediction and q-variance scope. Parameter and primitive-output auxiliary terms are weighted by hidden true component activity, so single-primitive batches do not train inactive adapters on unidentifiable nuisance params. Mixed Stage-C configs rebalance router-vs-adapter supervision with router CE at `0.25` and adapter, primitive-output and oracle-routed prediction supervision at `0.4`.
- `OVHAOutput` includes `adapter_params`, `router_logits`, `router_output` and `memory_bank`; diagnostics include adapter parameter stats, router prior entropy, query residual norm, per-primitive output gaps, and local scale/bias/lengthscale oracle errors.
- Controlled-v2 training configs use router auxiliary CE with `router_auxiliary_loss_weight = 0.25` when hidden true router weights are available.
- Single-primitive controlled-stress conclusions use the specialist-collapse tolerance (`full_true_router_learned_adapter <= 1.05 * matched_single + 0.01`) instead of requiring OVHA-full to strictly beat a dedicated single-family specialist.
- G1 single-primitive configs use oracle-routed adapter warmup for the full run: router frozen, true route override active, non-active primitive gradients masked, adapter/primitive/oracle-routed auxiliary losses enabled.
- `mlp_expert_moe` uses distinct expert keys (`mlp_expert_0`, `mlp_expert_1`) so the ModuleDict no longer collapses duplicate names.
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
.venv/bin/python -m unittest tests.test_phase1_7_adapter_router_patch
.venv/bin/python -m unittest discover -s tests -p 'test*.py'
```

Do not run the old all-in-one 5k sanity first. The Ubuntu/GPU handoff order is:

```bash
cd <repo>
git pull

# G1 single-primitive iid gates
PYTHON=.venv/bin/python python scripts/run_phase1_7_adapter_collapse_gate.py --family all --device cuda

# Equivalent manual commands
PYTHON=.venv/bin/python python train_torch_meta_operator.py --config configs/phase1_7_g1_single_iid_spectral.json
PYTHON=.venv/bin/python python eval_torch_meta_operator.py --config configs/phase1_7_g1_single_iid_spectral.json
PYTHON=.venv/bin/python python scripts/summarize_phase1_6.py --root outputs/phase1_7/g1_single_iid_spectral

PYTHON=.venv/bin/python python train_torch_meta_operator.py --config configs/phase1_7_g1_single_iid_local.json
PYTHON=.venv/bin/python python eval_torch_meta_operator.py --config configs/phase1_7_g1_single_iid_local.json
PYTHON=.venv/bin/python python scripts/summarize_phase1_6.py --root outputs/phase1_7/g1_single_iid_local

PYTHON=.venv/bin/python python train_torch_meta_operator.py --config configs/phase1_7_g1_single_iid_separable.json
PYTHON=.venv/bin/python python eval_torch_meta_operator.py --config configs/phase1_7_g1_single_iid_separable.json
PYTHON=.venv/bin/python python scripts/summarize_phase1_6.py --root outputs/phase1_7/g1_single_iid_separable

# G2/G3 router and memory iid gates
PYTHON=.venv/bin/python python train_torch_meta_operator.py --config configs/phase1_7_g2_g3_component_iid.json
PYTHON=.venv/bin/python python eval_torch_meta_operator.py --config configs/phase1_7_g2_g3_component_iid.json
PYTHON=.venv/bin/python python scripts/summarize_phase1_6.py --root outputs/phase1_7/g2_g3_component_iid
```

Only after those gates pass should the 5k sanity run be started:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHON=.venv/bin/python bash scripts/run_phase1_7_controlled_v2_sanity.sh
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

If only GPU 4 is available, use the single-GPU parallel runner to keep the A800 busier without touching other cards:

```bash
PYTHON=.venv/bin/python JOBS_PER_GPU=2 bash scripts/run_phase1_7_controlled_v2_sanity_gpu4_parallel.sh
```

This shards by `(seed, model)` into independent output directories under:

```text
outputs/phase1_7/controlled_v2_sanity_gpu4_parallel/
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

The 2026-05-29 `controlled_v2_sanity_context_gated_gpu4` run cleared this protocol gate with all single-primitive collapse rows passing and both compositional rows retaining positive component signal. The next step is a local-cache public pilot, not more controlled-v2 tuning. Public raw archives still need to be downloaded/converted on the server into the cache format in `docs/benchmark_protocol.md`; the training/eval path now fails clearly if the configured public cache is missing.

Start with the smallest real-data smoke config before the multi-family public mini:

```bash
CUDA_VISIBLE_DEVICES=4 \
OVHA_PUBLIC_BENCHMARK_ROOT=data/public_benchmark_cache \
bash scripts/run_phase1_6_gpu_public_pilot.sh \
  configs/phase1_7_public_pdebench_iid_smoke.json \
  outputs/phase1_7/public_pdebench_iid_smoke_gpu4
```

Only move to `configs/phase1_7_public_pdebench_mini.json` after the smoke run proves the public cache is being read (`dataset` and `source_split` populated in train/eval JSONL).

The 2026-05-29 Burgers public smoke first passed on one seed and then passed a 3-seed v2 check using `configs/phase1_7_public_pdebench_iid_smoke_v2.json` on GPU 4. The run used a local cache converted from `1D_Burgers_Sols_Nu0.01.hdf5` into `data/public_benchmark_cache/pdebench_burgers_1d/{train,iid}.npz`, with `tensor[:, 0, :]` as `input_field`, `tensor[:, -1, :]` as `output_field`, and `x-coordinate` as `coordinates`.

Public smoke v2 summary:

| dataset | split | seeds | ovha_full | transformer_only | simple_stack | best_baseline_delta | per-seed wins |
|---|---|---:|---:|---:|---:|---:|---:|
| pdebench_burgers_1d | iid | 3 | 0.307594 +/- 0.002463 | 0.381535 +/- 0.003310 | 0.536556 +/- 0.001503 | 0.073940 | 3/3 |

Per-seed v2 means:

| seed | ovha_full | best baseline | delta | win |
|---:|---:|---:|---:|---|
| 81 | 0.308088 | transformer_only 0.386120 | 0.078032 | true |
| 82 | 0.304362 | transformer_only 0.378425 | 0.074063 | true |
| 83 | 0.310333 | transformer_only 0.380060 | 0.069727 | true |

This is the first external-validity signal on a public PDEBench field cache and proves the public data path end to end: raw HDF5 -> local public cache -> metadata-free episode source -> train checkpoint -> checkpoint-loaded eval -> report. It is still not Phase 2 scientific Go; the next evidence step is to add at least a second PDEBench family/split before making broader method claims.

## Next Gate

If the 5k sanity run fails, inspect the oracle metrics before adding public data:

| Signal | Interpretation |
|---|---|
| oracle router succeeds, learned router fails | router training or router architecture issue |
| oracle adapter succeeds, learned adapter fails | hyper-adapter issue |
| oracle router and adapter fail | primitive/generator mismatch |
| train low but eval high | sampling/split/overfit issue |
| no-memory wins | context is not necessary or memory encoder is harmful |
