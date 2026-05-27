#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" train_torch_meta_operator.py --config configs/phase1_5_gpu_small.json --device cuda
"${PYTHON_BIN}" eval_torch_meta_operator.py --config configs/phase1_5_gpu_small.json --device cuda
"${PYTHON_BIN}" scripts/summarize_phase1_5.py outputs/phase1_5_gpu_small
