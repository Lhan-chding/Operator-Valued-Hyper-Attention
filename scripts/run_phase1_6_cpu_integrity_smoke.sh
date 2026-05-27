#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase1_6_cpu_integrity_smoke.json}"
OUTPUT_DIR="${2:-outputs/phase1_6/cpu_integrity_smoke}"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" train_torch_meta_operator.py --config "${CONFIG}" --device cpu --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" eval_torch_meta_operator.py --config "${CONFIG}" --device cpu --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" scripts/summarize_phase1_6.py --root "${OUTPUT_DIR}"
