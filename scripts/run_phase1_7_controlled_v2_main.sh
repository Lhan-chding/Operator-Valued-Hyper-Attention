#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase1_7_controlled_v2_main.json}"
OUTPUT_DIR="${2:-outputs/phase1_7/controlled_v2_main}"
PYTHON_BIN="${PYTHON:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-10}"
MONITOR_LOG="${OUTPUT_DIR}/process_monitor.log"
MONITOR_PID=""

mkdir -p "${OUTPUT_DIR}"

cleanup() {
  local status=$?
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "Stopped with status ${status}. If a Python training process is still alive, run:"
    echo "  pkill -TERM -f 'train_torch_meta_operator.py|eval_torch_meta_operator.py'"
    echo "  pkill -KILL -f 'train_torch_meta_operator.py|eval_torch_meta_operator.py'  # only if TERM fails"
  fi
}
trap cleanup EXIT INT TERM

start_monitor() {
  (
    while true; do
      echo
      echo "===== $(date '+%Y-%m-%d %H:%M:%S') | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ====="
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
        nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
      else
        echo "nvidia-smi not found"
      fi
      ps -o pid,ppid,stat,etime,%cpu,%mem,command -u "$(id -un)" \
        | grep -E 'train_torch_meta_operator.py|eval_torch_meta_operator.py|summarize_phase1_6.py' \
        | grep -v grep || true
      sleep "${MONITOR_INTERVAL_SECONDS}"
    done
  ) | tee -a "${MONITOR_LOG}" &
  MONITOR_PID=$!
}

run_python() {
  echo
  echo "+ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON_BIN} -u $*"
  "${PYTHON_BIN}" -u "$@"
}

echo "Using GPU ${CUDA_VISIBLE_DEVICES}; monitor log: ${MONITOR_LOG}"
start_monitor

run_python train_torch_meta_operator.py --config "${CONFIG}" --device cuda --output-dir "${OUTPUT_DIR}"
run_python eval_torch_meta_operator.py --config "${CONFIG}" --device cuda --output-dir "${OUTPUT_DIR}"
run_python scripts/summarize_phase1_6.py --root "${OUTPUT_DIR}"
