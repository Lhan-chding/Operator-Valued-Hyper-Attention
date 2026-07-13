# OVHA-ROD Phase 0/1

This project is the end-to-end Referring Object Detection branch of OVHA. It
extends the pinned MM-Grounding-DINO Swin-T parent at encoder query selection;
it does not read the repository's fixed-candidate feature artifacts.

The current implementation intentionally stops at the RQGO server gate:

1. reproduce the parent under the official five-epoch RefCOCO-family budget;
2. compare parent, parent plus referent head, generic dense seed, and RQGO;
3. verify zero-init equivalence, dense-query coverage, accuracy, and leakage;
4. implement no decoder operators until that gate passes.

## Layout

```text
configs/                 three training plus validation-only configs
environment/             exact MMDetection and Python dependency contract
ovha_rod/models/         Phase 1 role, seed, detector, and head modules
scripts/                 environment setup, preflight, and server runner
tests/unit/              pure Torch and static integration contracts
RUNBOOK.md               server procedure and acceptance checklist
```

## Locked parent

- Repository: `https://github.com/open-mmlab/mmdetection.git`
- Commit: `cfd5d3a985b0249de009b67d04f37263e11cdf3d`
- MM-Grounding-DINO Swin-T pretrained checkpoint: recorded in
  `environment/mmdetection.lock`
- Main protocol: five epochs, validation-only model selection, global batch 32,
  500 optimizer-step learning-rate and seed-loss warmup, AdamW, AMP, and no
  separate freeze stage
- Compatibility profile: `cu121-wheel`, using Python 3.10, PyTorch 2.1.0,
  torchvision 0.16.0, and the SHA-256-pinned official prebuilt MMCV 2.1.0
  wheel; it neither invokes `nvcc` nor changes the host CUDA toolkit
- Security exception: PyTorch 2.1.0 is an older compatibility release, so only
  the exact user-private official checkpoint may be loaded after its locked
  byte size and SHA-256 match; resume additionally accepts only an epoch
  checkpoint created inside the same private run directory with an exactly
  matching run identity; unknown `.pth` files are prohibited

The three configs inherit the official dataset-specific 5e recipes and replace
their evaluation surface with validation only. Held-out evaluation is not part
of this Phase 1 runner.

## Phase 1 variants

| Variant | `seed_operator` | Seed loss | Referent loss | Role diversity |
|---|---|---:|---:|---:|
| `phase0_parent` | official detector/head/schedule | 0 | 0 | 0 |
| `parent_ref` | `none` | 0 | 0.5 | 0 |
| `generic` | `generic` | 0.5 | 0.5 | 0 |
| `rqgo` | `rqgo` | 0.5 | 0.5 | 0.005 |

`phase0_parent` uses a separate val-only config that does not override the
official optimizer or scheduler. The other three models form the strict
same-custom-schedule Phase 1 control set. Generic and RQGO consume the same
dense IoU-quality targets and use the same bounded seed-bias contract. The
parent token maximum remains the base Top-K score in every Phase 1 variant.

`SeedLossWarmupHook` raises the configured seed-loss weight linearly from zero
over the first 500 optimizer updates. `OperatorDiagnosticsHook` writes finite
seed/loss scalars every 50 iterations to `operator_diagnostics.jsonl` inside
each work directory. When gradient accumulation is required, the server runner
scales the hook's iteration count so the warmup still spans 500 updates.

## Local checks

Pure Torch tests can run without MMDetection when the project package is on
`PYTHONPATH`:

```bash
cd projects/ovha_rod
PYTHONPATH="$PWD" ../../.venv/bin/python -m unittest discover -s tests/unit -v
```

Static checks for the server-facing files:

```bash
python3 -m compileall -q configs scripts/server_preflight.py
bash -n scripts/setup_mmdetection.sh
bash -n scripts/run_phase1_server.sh
python3 scripts/server_preflight.py --help
bash scripts/setup_mmdetection.sh --help
bash scripts/run_phase1_server.sh --help
```

## Server entry point

See [RUNBOOK.md](RUNBOOK.md). The short path is:

```bash
PRIVATE_ROOT="$HOME/.local/share/ovha-rod"
umask 077
install -d -m 700 "$PRIVATE_ROOT"
git clone --depth 1 --single-branch \
  --branch codex/ovha-rod-implementation \
  https://github.com/Lhan-chding/Operator-Valued-Hyper-Attention.git \
  "$PRIVATE_ROOT/project"
cd "$PRIVATE_ROOT/project/projects/ovha_rod"

bash scripts/setup_mmdetection.sh \
  --venv "$PRIVATE_ROOT/env-cu121" \
  --mmdet-dir "$PRIVATE_ROOT/mmdetection" \
  --python /usr/bin/python3

source "$PRIVATE_ROOT/env-cu121/bin/activate"

DATA_ROOT="$HOME/work/Operator-Valued-Hyper-Attention/data/raw_public/multimodal/COCO2014"
BERT_ROOT="$PRIVATE_ROOT/bert-base-uncased"
CHECKPOINT="$PRIVATE_ROOT/inputs/checkpoints/mm_grounding_dino_swin_t.pth"
WORK_ROOT="$PRIVATE_ROOT/runs"

# Set this only after verifying that the selected physical GPUs are idle.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

bash scripts/run_phase1_server.sh \
  --dataset refcoco \
  --variant rqgo \
  --mmdet-root "$PRIVATE_ROOT/mmdetection" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a \
  --bert-root "$BERT_ROOT" \
  --work-root "$WORK_ROOT" \
  --gpus 1 \
  --per-device-batch 8 \
  --master-port 29626
```

Before a full run, execute the dedicated two-batch train/backward smoke with
the same `--cfg-options` printed by the runner:

```bash
# The smoke entrypoint requires exactly one explicitly selected idle GPU.
SMOKE_DIR="$WORK_ROOT/refcoco/rqgo-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_VISIBLE_DEVICES=4 python scripts/two_batch_smoke.py \
  configs/ovha_rod_swin_t_5e_refcoco.py \
  --work-dir "$SMOKE_DIR" \
  --batch-size 8 \
  --cfg-options \
    model.seed_operator=rqgo \
    load_from="$CHECKPOINT" \
    model.language_model.name="$BERT_ROOT" \
    train_dataloader.dataset.data_root="$DATA_ROOT" \
    train_dataloader.dataset.pipeline.5.tokenizer_name="$BERT_ROOT"
```

MMEngine writes a complete `epoch_N.pth` after every epoch and keeps the two
latest checkpoints. An SSH disconnect is harmless when the command runs in
`tmux`. To perform a guarded epoch-boundary resume, rerun the identical command
with `--resume`; the runner rejects a changed GPU count, batch size,
accumulation, seed, variant, source commit, or environment identity. Mid-epoch
resume is intentionally rejected because the standard epoch loop does not
persist the dataloader cursor or worker RNG state.

Do not proceed to TQ-CATO, Q-SRO, or MS-TLEO until the RQGO gate in the
runbook passes.
