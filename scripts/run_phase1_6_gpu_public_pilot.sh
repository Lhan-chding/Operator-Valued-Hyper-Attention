#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase1_6_gpu_public_pilot.json}"
OUTPUT_DIR="${2:-outputs/phase1_6/gpu_public_pilot}"
PYTHON_BIN="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

"${PYTHON_BIN}" train_torch_meta_operator.py --config "${CONFIG}" --device cuda --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" eval_torch_meta_operator.py --config "${CONFIG}" --device cuda --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" scripts/summarize_phase1_6.py --root "${OUTPUT_DIR}"
