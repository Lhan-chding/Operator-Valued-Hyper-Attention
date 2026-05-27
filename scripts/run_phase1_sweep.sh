#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase1_minimal.json}"

python3 train_meta_operator.py --config "${CONFIG}"
python3 eval_meta_operator.py --config "${CONFIG}"
python3 scripts/summarize_phase1.py \
  --metrics outputs/phase1/eval_metrics.jsonl \
  --output outputs/phase1/phase1_report.md
