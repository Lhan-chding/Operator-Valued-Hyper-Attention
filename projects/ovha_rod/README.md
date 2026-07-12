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
- Security baseline: PyTorch 2.6.0 patched line; load only the locked official
  checkpoint after its version-controlled SHA-256 matches exactly

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
bash scripts/setup_mmdetection.sh \
  --venv /srv/envs/ovha-rod \
  --mmdet-dir /srv/src/mmdetection-cfd5d3a

source /srv/envs/ovha-rod/bin/activate

bash scripts/run_phase1_server.sh \
  --dataset refcoco \
  --variant all \
  --mmdet-root /srv/src/mmdetection-cfd5d3a \
  --data-root /srv/data/coco \
  --checkpoint /srv/checkpoints/mm_grounding_dino_swin_t.pth \
  --checkpoint-sha256 b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a \
  --bert-root /srv/models/bert-base-uncased \
  --gpus 8
```

Before a full run, execute the dedicated two-batch train/backward smoke with
the same `--cfg-options` printed by the runner:

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

Do not proceed to TQ-CATO, Q-SRO, or MS-TLEO until the RQGO gate in the
runbook passes.
