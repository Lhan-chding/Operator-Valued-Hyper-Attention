#!/usr/bin/env bash
set -euo pipefail
umask 077
unset PYTHONPATH PYTHONHOME BASH_ENV ENV CDPATH
export PATH="/usr/bin:/bin"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/configs/phase2/ovha_rod_swin_t_3e_refcoco_full.py"
PINNED_MMDET_COMMIT="cfd5d3a985b0249de009b67d04f37263e11cdf3d"
TARGET_GLOBAL_BATCH=32

MMDET_ROOT=""
DATA_ROOT=""
EPOCH2_CHECKPOINT=""
BERT_ROOT=""
WORK_ROOT="${HOME}/.local/share/ovha-rod/runs-phase2-full"
PYTHON_BIN="python"
GPUS=1
PER_DEVICE_BATCH=8
MASTER_PORT=""
SEED=2026
DRY_RUN=false
EXPECTED_SOURCE_PROJECT_COMMIT=""
TARGET_PROJECT_COMMIT=""

usage() {
  printf '%s\n' \
    "Usage: $0 --mmdet-root PATH --data-root PATH" \
    "          --epoch2-checkpoint PATH --bert-root PATH" \
    "          --expected-source-project-commit SHA" \
    "          --target-project-commit SHA [options]" \
    "" \
    "This starts a fresh three-epoch RefCOCO Phase 2 from epoch_2.pth." \
    "All Phase-1 parameters are frozen; only the complete decoder bank trains." \
    "The command intentionally uses load_from and never restores optimizer state."
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
    --mmdet-root) require_value "$1" "$#"; MMDET_ROOT="$2"; shift 2 ;;
    --data-root) require_value "$1" "$#"; DATA_ROOT="$2"; shift 2 ;;
    --epoch2-checkpoint) require_value "$1" "$#"; EPOCH2_CHECKPOINT="$2"; shift 2 ;;
    --bert-root) require_value "$1" "$#"; BERT_ROOT="$2"; shift 2 ;;
    --expected-source-project-commit) require_value "$1" "$#"; EXPECTED_SOURCE_PROJECT_COMMIT="$2"; shift 2 ;;
    --target-project-commit) require_value "$1" "$#"; TARGET_PROJECT_COMMIT="$2"; shift 2 ;;
    --work-root) require_value "$1" "$#"; WORK_ROOT="$2"; shift 2 ;;
    --python) require_value "$1" "$#"; PYTHON_BIN="$2"; shift 2 ;;
    --gpus) require_value "$1" "$#"; GPUS="$2"; shift 2 ;;
    --per-device-batch) require_value "$1" "$#"; PER_DEVICE_BATCH="$2"; shift 2 ;;
    --master-port) require_value "$1" "$#"; MASTER_PORT="$2"; shift 2 ;;
    --seed) require_value "$1" "$#"; SEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in MMDET_ROOT DATA_ROOT EPOCH2_CHECKPOINT BERT_ROOT \
    EXPECTED_SOURCE_PROJECT_COMMIT TARGET_PROJECT_COMMIT; do
  [[ -n "${!required}" ]] || {
    printf '%s is required\n' "${required}" >&2
    exit 2
  }
done
[[ "${EXPECTED_SOURCE_PROJECT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  printf '%s\n' '--expected-source-project-commit must be a full lowercase commit' >&2
  exit 2
}
[[ "${TARGET_PROJECT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  printf '%s\n' '--target-project-commit must be a full lowercase commit' >&2
  exit 2
}
[[ "$(basename -- "${EPOCH2_CHECKPOINT}")" == "epoch_2.pth" ]] || {
  printf 'Phase 2 requires the exact epoch_2.pth checkpoint\n' >&2
  exit 2
}
[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--gpus must be a positive integer' >&2
  exit 2
}
[[ "${PER_DEVICE_BATCH}" =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--per-device-batch must be a positive integer' >&2
  exit 2
}
[[ "${SEED}" =~ ^[0-9]+$ ]] || {
  printf '%s\n' '--seed must be a non-negative integer' >&2
  exit 2
}
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] && \
  (( MASTER_PORT >= 1024 && MASTER_PORT <= 65535 )) || {
    printf '%s\n' '--master-port must be an integer in [1024, 65535]' >&2
    exit 2
  }

DENOMINATOR=$((GPUS * PER_DEVICE_BATCH))
if (( DENOMINATOR > TARGET_GLOBAL_BATCH || TARGET_GLOBAL_BATCH % DENOMINATOR != 0 )); then
  printf 'gpus * per-device-batch must divide global batch %d; got %d\n' \
    "${TARGET_GLOBAL_BATCH}" "${DENOMINATOR}" >&2
  exit 2
fi
ACCUMULATIVE_COUNTS=$((TARGET_GLOBAL_BATCH / DENOMINATOR))
WARMUP_ITERS=$((500 * ACCUMULATIVE_COUNTS))
PYTHON_RESOLVED="$(command -v "${PYTHON_BIN}")" || {
  printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
}
export PATH="$(dirname "${PYTHON_RESOLVED}"):${PATH}"

WORK_DIR="${WORK_ROOT}/refcoco/full_bank_from_epoch2/seed_${SEED}"
TRAIN_CHECKPOINT="${EPOCH2_CHECKPOINT}"
SOURCE_SHA256="dry-run"
SOURCE_IDENTITY_SHA256="dry-run"
SOURCE_PROVENANCE_SHA256="dry-run"
VISIBLE_DEVICE_IDENTITY="dry-run"
DATA_ROOT_IDENTITY="${DATA_ROOT}"
BERT_ROOT_IDENTITY="${BERT_ROOT}"

if [[ "${DRY_RUN}" == false ]]; then
  [[ -f "${CONFIG_PATH}" && -d "${MMDET_ROOT}" ]] || {
    printf 'reviewed config or MMDetection root is missing\n' >&2
    exit 2
  }
  [[ -d "${DATA_ROOT}" && -d "${BERT_ROOT}" ]] || {
    printf 'data root and BERT root must be directories\n' >&2
    exit 2
  }
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
    printf 'CUDA_VISIBLE_DEVICES must explicitly select idle GPUs\n' >&2
    exit 2
  }
  PYTHONPATH="${PROJECT_DIR}:${MMDET_ROOT}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/phase2_preflight.py" \
    --runtime-project-dir "${PROJECT_DIR}" \
    --target-project-commit "${TARGET_PROJECT_COMMIT}" \
    --mmdet-root "${MMDET_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --bert-root "${BERT_ROOT}" \
    --work-root "${WORK_ROOT}" \
    --per-device-batch "${PER_DEVICE_BATCH}" \
    --accumulative-counts "${ACCUMULATIVE_COUNTS}" \
    --warmup-iters "${WARMUP_ITERS}" \
    --seed "${SEED}"
  DATA_ROOT_IDENTITY="$(realpath -e "${DATA_ROOT}")"
  BERT_ROOT_IDENTITY="$(realpath -e "${BERT_ROOT}")"
  mapfile -t FROZEN_INFO < <(
    PYTHONPATH="${PROJECT_DIR}:${MMDET_ROOT}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/freeze_phase2_checkpoint.py" \
      --checkpoint "${EPOCH2_CHECKPOINT}" \
      --freeze-dir "${HOME}/.local/share/ovha-rod/trusted_phase2_inputs" \
      --expected-source-project-commit "${EXPECTED_SOURCE_PROJECT_COMMIT}" \
      --expected-seed "${SEED}" \
      --expected-data-root "${DATA_ROOT_IDENTITY}" \
      --expected-bert-root "${BERT_ROOT_IDENTITY}"
  )
  [[ "${#FROZEN_INFO[@]}" -eq 4 ]] || {
    printf 'checkpoint freezer returned an invalid response\n' >&2
    exit 2
  }
  TRAIN_CHECKPOINT="${FROZEN_INFO[0]}"
  SOURCE_SHA256="${FROZEN_INFO[1]}"
  SOURCE_IDENTITY_SHA256="${FROZEN_INFO[2]}"
  SOURCE_PROVENANCE_SHA256="${FROZEN_INFO[3]}"
  [[ "${SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ \
      && "${SOURCE_IDENTITY_SHA256}" =~ ^[0-9a-f]{64}$ \
      && "${SOURCE_PROVENANCE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'checkpoint freezer returned an invalid digest\n' >&2
    exit 2
  }
  PYTHONPATH="${PROJECT_DIR}:${MMDET_ROOT}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_work_dir.py" "${WORK_DIR}"
  VISIBLE_DEVICE_IDENTITY="$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr -d '[:space:]')"
fi

PROJECT_COMMIT="${TARGET_PROJECT_COMMIT}"
CFG_OPTIONS=(
  "load_from=${TRAIN_CHECKPOINT}"
  "model.train_decoder_operator_only=True"
  "model.language_model.name=${BERT_ROOT}"
  "train_dataloader.batch_size=${PER_DEVICE_BATCH}"
  "train_dataloader.dataset.data_root=${DATA_ROOT}"
  "train_dataloader.dataset.pipeline.5.tokenizer_name=${BERT_ROOT}"
  "val_dataloader.dataset.data_root=${DATA_ROOT}"
  "val_evaluator.ann_file=${DATA_ROOT}/mdetr_annotations/finetune_refcoco_val.json"
  "optim_wrapper.accumulative_counts=${ACCUMULATIVE_COUNTS}"
  "optim_wrapper.clip_grad.error_if_nonfinite=True"
  "param_scheduler.0.end=${WARMUP_ITERS}"
  "randomness.seed=${SEED}"
  "randomness.deterministic=False"
  "default_hooks.checkpoint.by_epoch=True"
  "default_hooks.checkpoint.interval=1"
  "default_hooks.checkpoint.max_keep_ckpts=3"
  "default_hooks.checkpoint.save_last=True"
  "custom_hooks.1.identity_path=${WORK_DIR}/run_identity.json"
)
COMMAND=(
  /bin/bash "${MMDET_ROOT}/tools/dist_train.sh" "${CONFIG_PATH}" "${GPUS}"
  --work-dir "${WORK_DIR}"
  --cfg-options "${CFG_OPTIONS[@]}"
)

printf 'Phase 2 full bank: frozen epoch 2, %d new epochs, global batch %d, accumulation %d\n' \
  3 "${TARGET_GLOBAL_BATCH}" "${ACCUMULATIVE_COUNTS}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" == false ]]; then
  IDENTITY_OPTIONS=(
    --expected-identity "dataset=refcoco"
    --expected-identity "variant=phase2_full_bank"
    --expected-identity "seed=${SEED}"
    --expected-identity "global_batch=${TARGET_GLOBAL_BATCH}"
    --expected-identity "gpus=${GPUS}"
    --expected-identity "per_device_batch=${PER_DEVICE_BATCH}"
    --expected-identity "accumulative_counts=${ACCUMULATIVE_COUNTS}"
    --expected-identity "physical_cuda_devices=${VISIBLE_DEVICE_IDENTITY}"
    --expected-identity "data_root=${DATA_ROOT_IDENTITY}"
    --expected-identity "bert_root=${BERT_ROOT_IDENTITY}"
    --expected-identity "source_epoch2_sha256=${SOURCE_SHA256}"
    --expected-identity "source_identity_sha256=${SOURCE_IDENTITY_SHA256}"
    --expected-identity "source_provenance_sha256=${SOURCE_PROVENANCE_SHA256}"
    --expected-identity "source_project_commit=${EXPECTED_SOURCE_PROJECT_COMMIT}"
    --expected-identity "project_commit=${PROJECT_COMMIT}"
    --expected-identity "mmdetection_commit=${PINNED_MMDET_COMMIT}"
    --expected-identity "enabled_operators=qsro,tq_cato,ms_tleo"
    --expected-identity "qsro_query_chunk_size=128"
    --expected-identity "train_decoder_operator_only=true"
    --expected-identity "phase2_epochs=3"
    --expected-identity "deterministic=false"
    --expected-identity "amp=false"
  )
  PORT="${MASTER_PORT}" MASTER_ADDR="127.0.0.1" \
    PYTHONPATH="${PROJECT_DIR}:${MMDET_ROOT}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_lock.py" \
    --work-dir "${WORK_DIR}" --mode fresh \
    --freeze-dir "${HOME}/.local/share/ovha-rod/trusted_phase2_inputs" \
    --cwd "${MMDET_ROOT}" \
    --guard-gpus "${GPUS}" --guard-port "${MASTER_PORT}" \
    "${IDENTITY_OPTIONS[@]}" -- "${COMMAND[@]}"
fi
