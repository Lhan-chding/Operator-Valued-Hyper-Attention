# OVHA-ROD Phase 0/1 Server Runbook

## 1. Scope and stop condition

This runbook covers parent reproduction and the RQGO-only decision gate. It
does not authorize later decoder operators. Every command evaluates validation
data only; held-out splits remain untouched until a final model is frozen.

The Phase 1 protocol trains all configured parameter groups from the first
step. There is no separate freeze/unfreeze stage. Parent+referent, generic
seed, and RQGO share the custom optimizer groups, global batch, optimizer
updates, augmentation, warmup, and checkpoint. `phase0_parent` is deliberately
separate: it preserves the official optimizer/scheduler for reproduction and
is not the same-schedule causal control.

## 2. Required server state

- Linux server with NVIDIA CUDA and enough memory for MM-Grounding-DINO Swin-T.
- Python 3.9, 3.10, or 3.11.
- Git and network access during setup.
- COCO `train2014/` images.
- MDETR-style training and validation annotations for the selected dataset.
- Local `bert-base-uncased` directory.
- Official pretrained MM-Grounding-DINO Swin-T checkpoint.

Expected data layout for RefCOCO:

```text
/srv/data/coco/
├── train2014/
└── mdetr_annotations/
    ├── finetune_refcoco_train_vg.json
    └── finetune_refcoco_val.json
```

The `refcoco_plus` and `refcocog` modes use the corresponding official file
names checked by `server_preflight.py`.

## 3. Create the locked environment

The default setup targets PyTorch 2.6.0 with CUDA 12.4, the first patched
release for CVE-2025-32434. If the server requires CUDA 11.8 or 12.6, pass the
matching allowlisted official PyTorch wheel index while keeping the locked
PyTorch/torchvision versions fixed. Do not downgrade below 2.6.0.

```bash
cd /path/to/Operator-Valued-Hyper-Attention/projects/ovha_rod

bash scripts/setup_mmdetection.sh \
  --venv /srv/envs/ovha-rod \
  --mmdet-dir /srv/src/mmdetection-cfd5d3a

source /srv/envs/ovha-rod/bin/activate
export PYTHONPATH="$PWD:/srv/src/mmdetection-cfd5d3a${PYTHONPATH:+:$PYTHONPATH}"
```

The setup script refuses a non-Git destination, verifies the official origin,
checks out the exact locked commit, installs it editable, and verifies the
compiled deformable-attention import.

## 4. Download immutable model inputs

Checkpoint URL and model identity are recorded in
`environment/mmdetection.lock`. Store both the checkpoint and BERT files at
stable local paths. Verify the checkpoint against the version-controlled
digest:

```bash
sha256sum /srv/checkpoints/mm_grounding_dino_swin_t.pth
grep checkpoint_sha256 environment/mmdetection.lock
```

The values must match exactly, and passing the locked digest to preflight and
training is mandatory. Python configs and PyTorch checkpoints are executable
trust-boundary inputs: use only the reviewed project configs and official
checkpoint, never an untrusted file. Perform the first load in a
least-privilege environment without secrets or host-sensitive mounts.

## 5. Run fail-fast preflight

```bash
python scripts/server_preflight.py \
  --dataset refcoco \
  --mmdet-root /srv/src/mmdetection-cfd5d3a \
  --data-root /srv/data/coco \
  --checkpoint /srv/checkpoints/mm_grounding_dino_swin_t.pth \
  --checkpoint-sha256 b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a \
  --bert-root /srv/models/bert-base-uncased \
  --output /srv/runs/ovha_rod/refcoco/preflight.json
```

Exit code `0` is mandatory. The report checks:

- Python and package versions;
- compiled MMCV deformable attention;
- official origin and exact MMDetection commit;
- project registry import and resolved config;
- validation-only config policy;
- train/validation annotations and images;
- checkpoint existence/digest and local BERT files;
- visible CUDA devices.

## 6. Dry-run command generation

```bash
bash scripts/run_phase1_server.sh \
  --dataset refcoco \
  --variant all \
  --mmdet-root /srv/src/mmdetection-cfd5d3a \
  --data-root /srv/data/coco \
  --checkpoint /srv/checkpoints/mm_grounding_dino_swin_t.pth \
  --checkpoint-sha256 b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a \
  --bert-root /srv/models/bert-base-uncased \
  --gpus 8 \
  --dry-run
```

Inspect the four printed commands before launching. The runner enforces a
global batch of 32. For fewer GPUs it adds gradient accumulation and scales the
iteration-based learning-rate and seed-loss warmups so that both still cover
500 optimizer updates.

## 7. Run the four Phase 1 controls

```bash
bash scripts/run_phase1_server.sh \
  --dataset refcoco \
  --variant all \
  --mmdet-root /srv/src/mmdetection-cfd5d3a \
  --data-root /srv/data/coco \
  --checkpoint /srv/checkpoints/mm_grounding_dino_swin_t.pth \
  --checkpoint-sha256 b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a \
  --bert-root /srv/models/bert-base-uncased \
  --work-root /srv/runs/ovha_rod \
  --gpus 8 \
  --per-device-batch 4 \
  --seed 2026
```

Repeat with the preregistered seeds only after the first seed passes smoke and
artifact checks. Each work directory contains its exact command plus the
standard MMEngine config snapshot, logs, checkpoints, and validation metrics.
`OperatorDiagnosticsHook` additionally appends finite diagnostics every 50
iterations to `operator_diagnostics.jsonl`.

## 8. Required smoke checks

Before accepting a full five-epoch run, execute the dedicated two-batch
backward/checkpoint smoke (the full runner's dry-run prints the equivalent
overrides for each dataset):

```bash
python scripts/two_batch_smoke.py configs/ovha_rod_swin_t_5e_refcoco.py \
  --work-dir /srv/runs/ovha_rod/refcoco/rqgo-smoke \
  --cfg-options \
    model.seed_operator=rqgo \
    load_from=/srv/checkpoints/mm_grounding_dino_swin_t.pth \
    model.language_model.name=/srv/models/bert-base-uncased \
    train_dataloader.dataset.data_root=/srv/data/coco \
    train_dataloader.dataset.pipeline.5.tokenizer_name=/srv/models/bert-base-uncased
```

Then inspect the two batches:

- all parent and auxiliary losses are finite;
- RQGO/generic seed gradients are finite and non-zero when active;
- `seed_bias` is bounded by 2.0;
- invalid encoder positions have zero seed bias;
- parent selection uses the unchanged token maximum;
- peak memory leaves a safety margin;
- no ground-truth field enters `predict()`.

Abort immediately on NaN/Inf, missing dense tensors, checkpoint base-key
coverage below 99%, or any model-input leakage.

## 9. RQGO Go/No-Go gate

All conditions are required before later operators are considered:

1. Zero-init parent versus RQGO logits and boxes differ by less than `1e-5`.
2. Removing GT fields from prediction leaves outputs exactly unchanged.
3. Selected Top-K Query Oracle@0.5 improves by at least 0.5 percentage point
   on validation under the preregistered comparison.
4. Final validation Acc@0.5 is no more than 0.2 percentage point below parent.
5. RQGO is not weaker than the identically supervised generic dense seed
   control within the preregistered uncertainty rule.
6. The parent reproduction remains within 0.5 percentage point of the saved
   same-environment reference.
7. All run artifacts include config, command, environment/preflight report,
   unrounded metrics, diagnostics, and checkpoint identity.

If Query Oracle does not improve, stop and audit seed targets, relation-field
orientation, masks, and the Top-K insertion point. Do not compensate by adding
decoder modules.

## 10. Artifact handoff

Return these paths after each server run:

- `preflight.json`
- each variant's `command.txt`
- each variant's `operator_diagnostics.jsonl`
- resolved config dump
- full training log
- validation metric JSON/log
- seed/query diagnostics JSON
- checkpoint filename and SHA-256
- GPU peak-memory and batch-1 latency record

The next implementation phase starts only after those artifacts satisfy the
gate above.
