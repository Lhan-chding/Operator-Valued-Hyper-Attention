#!/usr/bin/env bash
set -euo pipefail
umask 077
unset PYTHONPATH PYTHONHOME BASH_ENV ENV CDPATH
export PATH="/usr/bin:/bin"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_GLOBAL_BATCH=32

SOURCE_WORK_DIR=""
RUNTIME_PROJECT_DIR=""
EXPECTED_SOURCE_PROJECT_COMMIT=""
TARGET_PROJECT_COMMIT=""
MMDET_ROOT=""
DATA_ROOT=""
CHECKPOINT=""
CHECKPOINT_SHA256=""
BERT_ROOT=""
PHASE1_WORK_ROOT=""
PHASE2_WORK_ROOT=""
PYTHON_BIN="python"
PHASE1_GPU="4"
PHASE2_GPU="5"
PHASE1_PORT="29627"
PHASE2_PORT="29628"
PER_DEVICE_BATCH="8"
SEED="2026"
DRY_RUN=false

usage() {
  printf '%s\n' \
    "Usage: $0 --source-work-dir PATH --runtime-project-dir PATH" \
    "          --expected-source-project-commit SHA" \
    "          --target-project-commit SHA --mmdet-root PATH" \
    "          --data-root PATH --checkpoint PATH" \
    "          --checkpoint-sha256 HEX --bert-root PATH" \
    "          --phase1-work-root PATH --phase2-work-root PATH [options]" \
    "" \
    "Fail-closed handoff: migrate exact epoch_1.pth, resume only epoch 2," \
    "then start the fresh three-epoch complete decoder-operator bank."
}

require_value() {
  local option="$1"
  local count="$2"
  [[ "${count}" -ge 2 ]] || {
    printf 'missing value for %s\n' "${option}" >&2
    exit 2
  }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-work-dir) require_value "$1" "$#"; SOURCE_WORK_DIR="$2"; shift 2 ;;
    --runtime-project-dir) require_value "$1" "$#"; RUNTIME_PROJECT_DIR="$2"; shift 2 ;;
    --expected-source-project-commit) require_value "$1" "$#"; EXPECTED_SOURCE_PROJECT_COMMIT="$2"; shift 2 ;;
    --target-project-commit) require_value "$1" "$#"; TARGET_PROJECT_COMMIT="$2"; shift 2 ;;
    --mmdet-root) require_value "$1" "$#"; MMDET_ROOT="$2"; shift 2 ;;
    --data-root) require_value "$1" "$#"; DATA_ROOT="$2"; shift 2 ;;
    --checkpoint) require_value "$1" "$#"; CHECKPOINT="$2"; shift 2 ;;
    --checkpoint-sha256) require_value "$1" "$#"; CHECKPOINT_SHA256="$2"; shift 2 ;;
    --bert-root) require_value "$1" "$#"; BERT_ROOT="$2"; shift 2 ;;
    --phase1-work-root) require_value "$1" "$#"; PHASE1_WORK_ROOT="$2"; shift 2 ;;
    --phase2-work-root) require_value "$1" "$#"; PHASE2_WORK_ROOT="$2"; shift 2 ;;
    --python) require_value "$1" "$#"; PYTHON_BIN="$2"; shift 2 ;;
    --phase1-gpu) require_value "$1" "$#"; PHASE1_GPU="$2"; shift 2 ;;
    --phase2-gpu) require_value "$1" "$#"; PHASE2_GPU="$2"; shift 2 ;;
    --phase1-port) require_value "$1" "$#"; PHASE1_PORT="$2"; shift 2 ;;
    --phase2-port) require_value "$1" "$#"; PHASE2_PORT="$2"; shift 2 ;;
    --per-device-batch) require_value "$1" "$#"; PER_DEVICE_BATCH="$2"; shift 2 ;;
    --seed) require_value "$1" "$#"; SEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in SOURCE_WORK_DIR RUNTIME_PROJECT_DIR \
    EXPECTED_SOURCE_PROJECT_COMMIT TARGET_PROJECT_COMMIT MMDET_ROOT DATA_ROOT \
    CHECKPOINT CHECKPOINT_SHA256 BERT_ROOT PHASE1_WORK_ROOT PHASE2_WORK_ROOT; do
  [[ -n "${!required}" ]] || {
    printf '%s is required\n' "${required}" >&2
    exit 2
  }
done
for commit in EXPECTED_SOURCE_PROJECT_COMMIT TARGET_PROJECT_COMMIT; do
  [[ "${!commit}" =~ ^[0-9a-f]{40}$ ]] || {
    printf '%s must be a full lowercase Git commit\n' "${commit}" >&2
    exit 2
  }
done
[[ "${EXPECTED_SOURCE_PROJECT_COMMIT}" != "${TARGET_PROJECT_COMMIT}" ]] || {
  printf '%s\n' 'source and target project commits must differ' >&2
  exit 2
}
[[ "${CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  printf '%s\n' '--checkpoint-sha256 must be 64 lowercase hex characters' >&2
  exit 2
}
for numeric in PHASE1_GPU PHASE2_GPU SEED; do
  [[ "${!numeric}" =~ ^[0-9]+$ ]] || {
    printf '%s must be a non-negative integer\n' "${numeric}" >&2
    exit 2
  }
done
[[ "${PER_DEVICE_BATCH}" =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--per-device-batch must be a positive integer' >&2
  exit 2
}
(( PER_DEVICE_BATCH <= TARGET_GLOBAL_BATCH \
    && TARGET_GLOBAL_BATCH % PER_DEVICE_BATCH == 0 )) || {
  printf 'per-device batch must divide global batch %d\n' \
    "${TARGET_GLOBAL_BATCH}" >&2
  exit 2
}
ACCUMULATIVE_COUNTS=$((TARGET_GLOBAL_BATCH / PER_DEVICE_BATCH))
WARMUP_ITERS=$((500 * ACCUMULATIVE_COUNTS))
for port in PHASE1_PORT PHASE2_PORT; do
  [[ "${!port}" =~ ^[0-9]+$ ]] \
    && (( ${!port} >= 1024 && ${!port} <= 65535 )) || {
      printf '%s must be in [1024, 65535]\n' "${port}" >&2
      exit 2
    }
done
[[ "${PHASE1_PORT}" != "${PHASE2_PORT}" ]] || {
  printf '%s\n' 'Phase 1 and Phase 2 ports must differ' >&2
  exit 2
}
if [[ "${DRY_RUN}" == false ]]; then
  PYTHON_RESOLVED="$(command -v "${PYTHON_BIN}")" || {
    printf 'locked Python executable is unavailable: %s\n' \
      "${PYTHON_BIN}" >&2
    exit 2
  }
  PYTHON_BIN="${PYTHON_RESOLVED}"
fi

MIGRATED_WORK_DIR="${PHASE1_WORK_ROOT}/refcoco/rqgo/seed_${SEED}"
EPOCH2_CHECKPOINT="${MIGRATED_WORK_DIR}/epoch_2.pth"
MIGRATE_COMMAND=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/migrate_epoch_resume.py"
  --source-work-dir "${SOURCE_WORK_DIR}"
  --target-work-dir "${MIGRATED_WORK_DIR}"
  --project-dir "${RUNTIME_PROJECT_DIR}"
  --expected-source-project-commit "${EXPECTED_SOURCE_PROJECT_COMMIT}"
  --target-project-commit "${TARGET_PROJECT_COMMIT}"
  --expected-source-checkpoint epoch_1.pth
)
PHASE1_COMMAND=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_migrated_resume.py"
  --runtime-project-dir "${RUNTIME_PROJECT_DIR}"
  --target-project-commit "${TARGET_PROJECT_COMMIT}"
  --python-bin "${PYTHON_BIN}" --
  --dataset refcoco --variant rqgo
  --mmdet-root "${MMDET_ROOT}" --data-root "${DATA_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --checkpoint-sha256 "${CHECKPOINT_SHA256}"
  --bert-root "${BERT_ROOT}" --work-root "${PHASE1_WORK_ROOT}"
  --gpus 1 --per-device-batch "${PER_DEVICE_BATCH}"
  --master-port "${PHASE1_PORT}" --seed "${SEED}"
  --resume --stop-after-epoch2
)
PHASE2_ATTEST_COMMAND=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/phase2_preflight.py"
  --runtime-project-dir "${RUNTIME_PROJECT_DIR}"
  --target-project-commit "${TARGET_PROJECT_COMMIT}"
  --mmdet-root "${MMDET_ROOT}" --data-root "${DATA_ROOT}"
  --bert-root "${BERT_ROOT}" --work-root "${PHASE2_WORK_ROOT}"
  --per-device-batch "${PER_DEVICE_BATCH}"
  --accumulative-counts "${ACCUMULATIVE_COUNTS}"
  --warmup-iters "${WARMUP_ITERS}" --seed "${SEED}"
)
PHASE2_COMMAND=(
  /bin/bash "${RUNTIME_PROJECT_DIR}/scripts/run_phase2_full_server.sh"
  --mmdet-root "${MMDET_ROOT}" --data-root "${DATA_ROOT}"
  --epoch2-checkpoint "${EPOCH2_CHECKPOINT}" --bert-root "${BERT_ROOT}"
  --expected-source-project-commit "${TARGET_PROJECT_COMMIT}"
  --target-project-commit "${TARGET_PROJECT_COMMIT}"
  --work-root "${PHASE2_WORK_ROOT}" --python "${PYTHON_BIN}"
  --gpus 1 --per-device-batch "${PER_DEVICE_BATCH}"
  --master-port "${PHASE2_PORT}" --seed "${SEED}"
)

printf 'Migration:'
printf ' %q' "${MIGRATE_COMMAND[@]}"
printf '\nPhase 1: CUDA_VISIBLE_DEVICES=%q' "${PHASE1_GPU}"
printf ' %q' "${PHASE1_COMMAND[@]}"
printf '\nPhase 2 attestation: CUDA_VISIBLE_DEVICES=%q' "${PHASE2_GPU}"
printf ' %q' "${PHASE2_ATTEST_COMMAND[@]}"
printf '\nPhase 2: CUDA_VISIBLE_DEVICES=%q' "${PHASE2_GPU}"
printf ' %q' "${PHASE2_COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" == true ]]; then
  exit 0
fi

for script in migrate_epoch_resume.py run_migrated_resume.py \
    phase2_preflight.py; do
  [[ -f "${SCRIPT_DIR}/${script}" ]] || {
    printf 'trusted handoff script is missing: %s\n' "${script}" >&2
    exit 2
  }
done
[[ -f "${RUNTIME_PROJECT_DIR}/scripts/run_phase2_full_server.sh" ]] || {
  printf '%s\n' 'runtime Phase-2 runner is missing' >&2
  exit 2
}
[[ -x /bin/bash ]] || {
  printf '%s\n' 'system Bash executable is unavailable' >&2
  exit 2
}

"${MIGRATE_COMMAND[@]}"
CUDA_VISIBLE_DEVICES="${PHASE1_GPU}" "${PHASE1_COMMAND[@]}"
[[ -f "${EPOCH2_CHECKPOINT}" ]] || {
  printf 'Phase 1 exited without exact epoch_2.pth\n' >&2
  exit 1
}
CUDA_VISIBLE_DEVICES="${PHASE2_GPU}" \
  PYTHONPATH="${SCRIPT_DIR}/..:${MMDET_ROOT}" \
  "${PHASE2_ATTEST_COMMAND[@]}"
CUDA_VISIBLE_DEVICES="${PHASE2_GPU}" "${PHASE2_COMMAND[@]}"
