#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase1_7_controlled_v2_sanity.json}"
OUTPUT_ROOT="${2:-outputs/phase1_7/controlled_v2_sanity_gpu3_parallel}"
PYTHON_BIN="${PYTHON:-python3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OVHA_PROGRESS_INTERVAL="${OVHA_PROGRESS_INTERVAL:-100}"
export OVHA_EVAL_PROGRESS_INTERVAL="${OVHA_EVAL_PROGRESS_INTERVAL:-256}"

echo "Using only GPU 3 with ${JOBS_PER_GPU} concurrent job(s)."
echo "Output root: ${OUTPUT_ROOT}"

"${PYTHON_BIN}" -u scripts/run_phase1_7_sanity_parallel.py \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --gpus 3 \
  --jobs-per-gpu "${JOBS_PER_GPU}" \
  --python "${PYTHON_BIN}"
