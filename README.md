# Operator-Valued Hyper-Attention Phase 1

This repository contains the Phase-1 OVHA prototype described in `ovha_phase1_theory_and_codex_plan_cn_v1.docx`.

The implementation is intentionally zero-dependency so it can run in a fresh Python 3.9+ environment:

- `theory_notes/` formalizes the OVHA math object, special-case reductions, and proof sketches.
- `moat_ovha/models/` implements memory, hyper-adapter, router, primitives, and the OVHA layer.
- `moat_ovha/data/` provides a deterministic 1D/2D-ready operator zoo interface.
- `moat_ovha/baselines/` provides Transformer-only, Perceiver IO-style, ICON-style, DeepONet-only, FNO-only, simple-stack, and ablation predictors.
- `tests/` contains the Phase-1 contract tests.
- `scripts/` runs and summarizes the deterministic Phase-1 sweep.

## Run

```bash
python3 -m unittest discover -s tests
sh scripts/run_phase1_sweep.sh configs/phase1_minimal.json
```

Outputs are written to `outputs/phase1/`:

- `train_metrics.jsonl`
- `eval_metrics.jsonl`
- `phase1_report.md`

## Phase-1 Scope

The prototype is a minimal executable scaffold, not a large neural training stack. Once torch/numpy are available, the same interfaces can be upgraded to learned training loops without changing the Phase-1 acceptance surface.
