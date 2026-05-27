#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase1_5_cpu_smoke.json}"
OUTPUT_DIR="${2:-outputs/phase1_5_cpu_smoke}"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" train_torch_meta_operator.py --config "${CONFIG}" --device cpu --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" eval_torch_meta_operator.py --config "${CONFIG}" --device cpu --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" scripts/summarize_phase1_5.py "${OUTPUT_DIR}"
