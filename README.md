# Operator-Valued Hyper-Attention Phase 1

This repository contains the Phase-1 OVHA prototype described in `ovha_phase1_theory_and_codex_plan_cn_v1.docx`.

The implementation is intentionally zero-dependency so it can run in a fresh Python 3.9+ environment:

- `theory_notes/` formalizes the OVHA math object, special-case reductions, and proof sketches.
- `moat_ovha/models/` implements memory, hyper-adapter, router, primitives, and the OVHA layer.
- `moat_ovha/data/` provides a deterministic 1D/2D-ready operator zoo interface.
- `moat_ovha/baselines/` provides Transformer-only, Perceiver IO-style, ICON-style, DeepONet-only, FNO-only, simple-stack, and ablation predictors.
- `tests/` contains the Phase-1 contract tests.
- `scripts/` runs and summarizes the deterministic Phase-1 sweep.
- `docs/project_direction.md` is the persistent project-level handoff file for future windows.
- `docs/phases/phase_1.md` is the stable Phase-1 stage summary.
- `docs/phases/phase_1_5.md` is the Phase-1.5 metadata-free PyTorch plan and status file.

## Run

```bash
python3 -m unittest discover -s tests
sh scripts/run_phase1_sweep.sh configs/phase1_minimal.json
```

Outputs are written to `outputs/phase1/`:

- `train_metrics.jsonl`
- `eval_metrics.jsonl`
- `phase1_report.md`
- `phase1_gpt_pro_summary.md`

## Phase-1 Scope

The prototype is a minimal executable scaffold, not a large neural training stack. Once torch/numpy are available, the same interfaces can be upgraded to learned training loops without changing the Phase-1 acceptance surface.

## Continuity Docs

Future work should update `docs/project_direction.md` and the relevant `docs/phases/phase_N.md` file at the end of every phase or major decision point.

## Phase 1.5

Phase 1.5 adds `moat_ovha_torch/`, a metadata-free PyTorch implementation surface. Torch is optional for repository import: if it is not installed, torch-specific tests skip and the CPU smoke scripts write a clear skip report.
