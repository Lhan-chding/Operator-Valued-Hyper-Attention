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
- Python 3.10 on Linux x86_64 for the exact prebuilt MMCV wheel.
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

The selected `cu121-wheel` compatibility profile pins PyTorch 2.1.0,
torchvision 0.16.0, and the exact official MMCV 2.1.0 cp310 Linux x86_64 wheel.
It downloads CUDA 12.1 runtime wheels into a user venv; it does not invoke
`nvcc`, install a CUDA toolkit, change the NVIDIA driver, or use a GPU during
setup. The MMCV URL and SHA-256 are version controlled, and binary-only
installation fails closed instead of falling back to a source build.

This profile is a compatibility exception, not the general security baseline:
PyTorch 2.1.0 predates the patched checkpoint loader. Never load an unknown
`.pth`. The only allowed initial checkpoint is the exact OpenMMLab artifact
whose byte size and SHA-256 are locked in `environment/mmdetection.lock`.

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
export PYTHONPATH="$PWD:$PRIVATE_ROOT/mmdetection${PYTHONPATH:+:$PYTHONPATH}"
chmod -R go-w "$PRIVATE_ROOT/project" "$PRIVATE_ROOT/mmdetection"
DATA_ROOT="$HOME/work/Operator-Valued-Hyper-Attention/data/raw_public/multimodal/COCO2014"
BERT_ROOT="$PRIVATE_ROOT/bert-base-uncased"
CHECKPOINT="$PRIVATE_ROOT/inputs/checkpoints/mm_grounding_dino_swin_t.pth"
```

The setup script refuses any existing or symlinked venv, refuses a non-Git or
symlinked MMDetection destination, verifies the official origin, checks out the
exact locked commit, installs it editable, and verifies the prebuilt
deformable-attention import plus the CUDA 12.1 runtime. It also writes a
mode-600 environment manifest inside the venv.

Unknown or cross-run resume remains disabled for this older-Torch compatibility
profile. A guarded epoch-boundary resume is allowed only from the same private
work directory and exact recorded run identity. The guard freezes the selected
checkpoint into a private read-only copy before MMEngine deserializes it.

## 4. Download immutable model inputs

Checkpoint URL and model identity are recorded in
`environment/mmdetection.lock`. Store both the checkpoint and BERT files at
stable local paths. Verify the checkpoint against the version-controlled
digest:

```bash
PRIVATE_ROOT="$HOME/.local/share/ovha-rod"
INPUT_DIR="$PRIVATE_ROOT/inputs/checkpoints"
CHECKPOINT="$INPUT_DIR/mm_grounding_dino_swin_t.pth"
CHECKPOINT_URL="https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"
EXPECTED_SHA="b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a"
install -d -m 700 "$INPUT_DIR"
curl -fL --retry 5 -o "$CHECKPOINT.partial" "$CHECKPOINT_URL"
test "$(stat -c %s "$CHECKPOINT.partial")" = "1093815743"
echo "$EXPECTED_SHA  $CHECKPOINT.partial" | sha256sum -c -
chmod 400 "$CHECKPOINT.partial"
mv "$CHECKPOINT.partial" "$CHECKPOINT"
```

Use the pinned safetensors BERT snapshot. `pytorch_model.bin` is prohibited in
this profile because it would cross the older-Torch pickle boundary:

```bash
BERT_ROOT="$PRIVATE_ROOT/bert-base-uncased"
BERT_REVISION="86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_BASE="https://huggingface.co/google-bert/bert-base-uncased/resolve/$BERT_REVISION"
install -d -m 700 "$BERT_ROOT"
for name in config.json tokenizer_config.json tokenizer.json vocab.txt; do
  curl -fL --retry 5 -o "$BERT_ROOT/$name" "$BERT_BASE/$name"
  chmod 400 "$BERT_ROOT/$name"
done
curl -fL --retry 5 -o "$BERT_ROOT/model.safetensors.partial" \
  "$BERT_BASE/model.safetensors"
test "$(stat -c %s "$BERT_ROOT/model.safetensors.partial")" = "440449768"
echo "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3  $BERT_ROOT/model.safetensors.partial" | sha256sum -c -
chmod 400 "$BERT_ROOT/model.safetensors.partial"
mv "$BERT_ROOT/model.safetensors.partial" "$BERT_ROOT/model.safetensors"
test ! -e "$BERT_ROOT/pytorch_model.bin"
```

The size and digest must match exactly, the file must be user-owned with no
group/other permissions or symlinked path component, and passing the locked
digest to preflight and training is mandatory. Python configs and PyTorch
checkpoints are executable trust-boundary inputs: use only the reviewed project
configs and official checkpoint, never an untrusted file. Perform the first
load in a least-privilege environment without secrets or host-sensitive mounts.

## 5. Run fail-fast preflight

```bash
python scripts/server_preflight.py \
  --dataset refcoco \
  --mmdet-root "$PRIVATE_ROOT/mmdetection" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a \
  --bert-root "$BERT_ROOT" \
  --work-root "$PRIVATE_ROOT/runs" \
  --output "$PRIVATE_ROOT/runs/refcoco/preflight.json"
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

For any real launch, first reserve or otherwise coordinate the physical GPUs,
then set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and an explicit
`CUDA_VISIBLE_DEVICES` list whose length equals `--gpus`. The runner refuses
busy cards and rechecks them immediately before every variant. Also choose a
free localhost `--master-port`; do not share the default torchrun port.

```bash
bash scripts/run_phase1_server.sh \
  --dataset refcoco \
  --variant all \
  --mmdet-root "$PRIVATE_ROOT/mmdetection" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a \
  --bert-root "$BERT_ROOT" \
  --work-root "$PRIVATE_ROOT/runs" \
  --gpus 8 \
  --master-port 29626 \
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
  --mmdet-root "$PRIVATE_ROOT/mmdetection" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a \
  --bert-root "$BERT_ROOT" \
  --work-root "$PRIVATE_ROOT/runs" \
  --gpus 8 \
  --per-device-batch 4 \
  --master-port 29626 \
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
SMOKE_DIR="$PRIVATE_ROOT/runs/refcoco/rqgo-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
python scripts/two_batch_smoke.py configs/ovha_rod_swin_t_5e_refcoco.py \
  --work-dir "$SMOKE_DIR" \
  --batch-size 8 \
  --cfg-options \
    model.seed_operator=rqgo \
    load_from="$CHECKPOINT" \
    model.language_model.name="$BERT_ROOT" \
    train_dataloader.dataset.data_root="$DATA_ROOT" \
    train_dataloader.dataset.pipeline.5.tokenizer_name="$BERT_ROOT"
```

Then inspect the two batches:

- all parent and auxiliary losses are finite;
- `grad_norm` is finite; a non-finite full-model gradient is a hard failure;
- RQGO/generic seed gradients are finite and non-zero when active;
- `seed_bias` is bounded by 2.0;
- invalid encoder positions have zero seed bias;
- parent selection uses the unchanged token maximum;
- peak memory leaves a safety margin;
- no ground-truth field enters `predict()`.

Abort immediately on NaN/Inf, missing dense tensors, checkpoint base-key
coverage below 99%, or any model-input leakage.

The locked Phase 1 AMP profile is BF16 autocast with loss scale `1.0`. On the
A800 this retains the FP32 exponent range while using tensor-core mixed
precision. Dynamic FP16 scaling is not allowed for the formal run because an
overflow can make `GradScaler` skip `optimizer.step()` while the scheduler
continues. The optimizer wrapper also sets
`clip_grad.error_if_nonfinite=True`, so the job fails at the first invalid
full-model gradient instead of continuing with `grad_norm: nan`.

## 9. Checkpoints, disconnects, and safe resume

Run formal training inside `tmux`. Closing the laptop, losing SSH, or detaching
from tmux does not stop the server process and therefore does not require a
resume. The formal runner saves model, optimizer, scheduler, runner state, and
AMP scaler as `epoch_N.pth` after every epoch and keeps the latest two.

The original run command can be resumed by adding one flag:

```bash
bash scripts/run_phase1_server.sh \
  --dataset refcoco \
  --variant rqgo \
  --mmdet-root "$PRIVATE_ROOT/mmdetection" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a \
  --bert-root "$BERT_ROOT" \
  --work-root "$PRIVATE_ROOT/runs" \
  --gpus 1 \
  --per-device-batch 8 \
  --master-port 29626 \
  --seed 2026 \
  --resume
```

Resume fails closed unless `last_checkpoint` names a non-empty private
`epoch_N.pth` inside the same work directory and `run_identity.json` exactly
matches dataset, variant, seed, GPU count, per-device batch, accumulation,
global batch, AMP mode and dtype, project commit, MMDetection commit, and
environment profile. Never edit `last_checkpoint`, copy a `.pth` from another
run, or use a downloaded checkpoint as resume input. PyTorch checkpoints are
pickle trust boundaries.

Mid-epoch resume is deliberately rejected. MMEngine's standard epoch loop does
not persist the dataloader cursor, worker RNG, or prefetch queue, so treating an
`iter_N.pth` as exact formal continuation can repeat or omit samples. If a job
is intentionally stopped before the epoch checkpoint is complete, restart from
the preceding complete epoch.

## 10. RQGO Go/No-Go gate

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

## 11. Artifact handoff

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
