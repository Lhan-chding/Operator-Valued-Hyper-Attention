# OVHA-ROD Phase 0/1

This project is the end-to-end Referring Object Detection branch of OVHA. It
extends the pinned MM-Grounding-DINO Swin-T parent at encoder query selection;
it does not read the repository's fixed-candidate feature artifacts.

Formal experiment activation intentionally stops at the RQGO server gate:

1. reproduce the parent under the official five-epoch RefCOCO-family budget;
2. compare parent, parent plus referent head, generic dense seed, and RQGO;
3. verify zero-init equivalence, dense-query coverage, accuracy, and leakage;
4. keep the staged decoder operator bank disabled until that gate passes.

The post-RQGO implementation is already staged behind an explicit, default-off
configuration boundary. This lets operator engineering and CPU contract tests
finish while the RQGO run is in progress without changing that run's model,
optimizer, numerical protocol, or checkpoint identity.

## Layout

```text
configs/                 Phase 1 configs and post-RQGO experiment configs
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
  500 optimizer-step learning-rate and seed-loss warmup, AdamW, full FP32, and no
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

## Post-RQGO decoder operator bank

The staged bank applies structured residuals after each decoder layer to only
the trailing matching-query block; the denoising prefix is preserved. It
contains:

- `Q-SRO`: query-to-query spatial and semantic relation reasoning;
- `TQ-CATO`: masked token-to-query transport with row-normalized softmax;
- `MS-TLEO`: valid-ratio-aware multi-scale local and context sampling;
- Router, recurrent operator memory, low-rank hyper-adaptation, and RCEO
  reliability modulation;
- independent query, box-logit, and referent-score residual gates initialized
  to zero so the enabled bank begins at the parent path.

`configs/post_rqgo/` contains one full-bank config, three single-operator
configs, and four component ablations (`no_router`, `no_memory`,
`no_hyper_adapter`, and `no_rceo`). The original three formal Phase 1 configs
contain no `decoder_operator_cfg`, so they remain byte-for-byte on the disabled
path. `OperatorDiagnosticsHook` records seed and decoder-bank gradient norms
plus finite scalar diagnostics.

The implementation and pure-Torch contracts are complete locally, but CUDA
acceptance is pending. Before treating any post-RQGO variant as runnable or
scientifically accepted, execute the locked A800 two-batch forward/backward
smoke, verify parent equivalence at zero gates, finite non-zero active gradients,
DN-prefix preservation, peak memory, and end-to-end MMDetection integration.

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

Phase 1 uses full FP32. The locked MMCV 2.1.0 CUDA extension does not implement
multi-scale deformable attention for BF16, while dynamic FP16 scaling produced
non-finite gradients and skipped optimizer steps on the server. Gradient
clipping is fail-fast: any non-finite full-model gradient aborts the run instead
of logging `grad_norm: nan` and continuing. Treat such an abort as a failed
smoke/run, not as a warning to ignore.

MMEngine writes a complete `epoch_N.pth` after every epoch and keeps the two
latest checkpoints. An SSH disconnect is harmless when the command runs in
`tmux`. To perform a guarded epoch-boundary resume, rerun the identical command
with `--resume`; the runner rejects a changed GPU count, batch size,
accumulation, AMP dtype, seed, variant, source commit, or environment identity.
Mid-epoch resume is intentionally rejected because the standard epoch loop
does not persist the dataloader cursor or worker RNG state.

Do not enable the staged post-RQGO bank in a formal experiment until the RQGO
gate and the CUDA acceptance checklist in the runbook both pass.
