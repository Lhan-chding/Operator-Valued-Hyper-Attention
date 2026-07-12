#!/usr/bin/env bash
set -euo pipefail
umask 077
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_GLOBAL_BATCH=32
DEFAULT_PER_DEVICE_BATCH=4

DATASET=""
VARIANT="rqgo"
MMDET_ROOT=""
DATA_ROOT=""
CHECKPOINT=""
BERT_ROOT=""
WORK_ROOT="${PROJECT_DIR}/work_dirs"
GPUS=8
PER_DEVICE_BATCH="${DEFAULT_PER_DEVICE_BATCH}"
SEED=2026
PYTHON_BIN="python"
CHECKPOINT_SHA256=""
LOCKED_CHECKPOINT_SHA256="b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a"
LOCKED_CHECKPOINT_SIZE=1093815743
USE_AMP=true
DRY_RUN=false

usage() {
  printf '%s\n' \
    "Usage: $0 --dataset NAME --mmdet-root PATH --data-root PATH" \
    "          --checkpoint PATH --bert-root PATH [options]" \
    "" \
    "Required:" \
    "  --dataset NAME          refcoco, refcoco_plus, or refcocog" \
    "  --mmdet-root PATH       Official pinned MMDetection checkout" \
    "  --data-root PATH        COCO root containing train2014 and annotations" \
    "  --checkpoint PATH       Official Swin-T pretrained checkpoint" \
    "  --bert-root PATH        Local bert-base-uncased directory" \
    "" \
    "Options:" \
    "  --variant NAME          phase0_parent, parent_ref, generic, rqgo, or all" \
    "  --work-root PATH        Output root" \
    "  --gpus N                Number of visible training GPUs (default: 8)" \
    "  --per-device-batch N    Batch per GPU (default: 4)" \
    "  --seed N                Run seed (default: 2026)" \
    "  --python PATH           Python in the locked environment" \
    "  --checkpoint-sha256 HEX Required trusted checkpoint digest" \
    "  --no-amp                Disable AMP" \
    "  --dry-run               Validate arguments and print commands only" \
    "  -h, --help              Show this help" \
    "" \
    "The script preserves the official global batch of 32 with gradient" \
    "accumulation. The main Phase 1 protocol has no separate freeze stage."
}

require_value() {
  local option="$1"
  local count="$2"
  [[ "${count}" -ge 2 ]] || { printf 'missing value for %s\n' "${option}" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) require_value "$1" "$#"; DATASET="$2"; shift 2 ;;
    --variant) require_value "$1" "$#"; VARIANT="$2"; shift 2 ;;
    --mmdet-root) require_value "$1" "$#"; MMDET_ROOT="$2"; shift 2 ;;
    --data-root) require_value "$1" "$#"; DATA_ROOT="$2"; shift 2 ;;
    --checkpoint) require_value "$1" "$#"; CHECKPOINT="$2"; shift 2 ;;
    --bert-root) require_value "$1" "$#"; BERT_ROOT="$2"; shift 2 ;;
    --work-root) require_value "$1" "$#"; WORK_ROOT="$2"; shift 2 ;;
    --gpus) require_value "$1" "$#"; GPUS="$2"; shift 2 ;;
    --per-device-batch) require_value "$1" "$#"; PER_DEVICE_BATCH="$2"; shift 2 ;;
    --seed) require_value "$1" "$#"; SEED="$2"; shift 2 ;;
    --python) require_value "$1" "$#"; PYTHON_BIN="$2"; shift 2 ;;
    --checkpoint-sha256) require_value "$1" "$#"; CHECKPOINT_SHA256="$2"; shift 2 ;;
    --no-amp) USE_AMP=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${DATASET}" in
  refcoco) CONFIG_NAME="ovha_rod_swin_t_5e_refcoco.py"; PHASE0_CONFIG_NAME="phase0_parent_swin_t_5e_refcoco.py"; VAL_ANN="finetune_refcoco_val.json" ;;
  refcoco_plus) CONFIG_NAME="ovha_rod_swin_t_5e_refcoco_plus.py"; PHASE0_CONFIG_NAME="phase0_parent_swin_t_5e_refcoco_plus.py"; VAL_ANN="finetune_refcoco+_val.json" ;;
  refcocog) CONFIG_NAME="ovha_rod_swin_t_5e_refcocog.py"; PHASE0_CONFIG_NAME="phase0_parent_swin_t_5e_refcocog.py"; VAL_ANN="finetune_refcocog_val.json" ;;
  *) printf 'invalid --dataset: %s\n' "${DATASET}" >&2; usage >&2; exit 2 ;;
esac

case "${VARIANT}" in
  phase0_parent|parent_ref|generic|rqgo) VARIANTS=("${VARIANT}") ;;
  all) VARIANTS=(phase0_parent parent_ref generic rqgo) ;;
  *) printf 'invalid --variant: %s\n' "${VARIANT}" >&2; exit 2 ;;
esac

for required in MMDET_ROOT DATA_ROOT CHECKPOINT BERT_ROOT; do
  [[ -n "${!required}" ]] || { printf '%s is required\n' "${required}" >&2; exit 2; }
done
[[ "${CHECKPOINT_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]] || {
  printf '--checkpoint-sha256 must be exactly 64 hexadecimal characters\n' >&2
  exit 2
}
NORMALIZED_CHECKPOINT_SHA256="$(printf '%s' "${CHECKPOINT_SHA256}" | tr '[:upper:]' '[:lower:]')"
if [[ "${NORMALIZED_CHECKPOINT_SHA256}" != "${LOCKED_CHECKPOINT_SHA256}" ]]; then
  printf 'checkpoint SHA-256 does not match the locked official artifact\n' >&2
  exit 2
fi
TRAIN_CHECKPOINT="${CHECKPOINT}"
if [[ "${DRY_RUN}" == false ]]; then
  [[ -f "${CHECKPOINT}" && ! -L "${CHECKPOINT}" ]] || {
    printf 'checkpoint must be a regular non-symlink file: %s\n' "${CHECKPOINT}" >&2
    exit 2
  }
  [[ -n "${HOME:-}" && -d "${HOME}" && ! -L "${HOME}" ]] || {
    printf 'HOME must be a real user-owned directory\n' >&2
    exit 2
  }
  PRIVATE_ROOT="${HOME}/.local/share/ovha-rod"
  PRIVATE_INPUT_DIR="${PRIVATE_ROOT}/trusted_inputs"
  install -d -m 700 "${PRIVATE_ROOT}" "${PRIVATE_INPUT_DIR}"
  for private_dir in \
      "${HOME}" "${HOME}/.local" "${HOME}/.local/share" \
      "${PRIVATE_ROOT}" "${PRIVATE_INPUT_DIR}"; do
    [[ -d "${private_dir}" && ! -L "${private_dir}" ]] || {
      printf 'private input parent is missing or symlinked: %s\n' "${private_dir}" >&2
      exit 2
    }
    OWNER_ID="$(stat -c %u "${private_dir}")"
    DIR_MODE="$(stat -c %a "${private_dir}")"
    if [[ "${OWNER_ID}" != "$(id -u)" ]] || (( (8#${DIR_MODE} & 022) != 0 )); then
      printf 'private input parent is not user-owned/private: %s mode=%s\n' \
        "${private_dir}" "${DIR_MODE}" >&2
      exit 2
    fi
  done
  TRAIN_CHECKPOINT="${PRIVATE_INPUT_DIR}/mm_grounding_dino_swin_t-b448804b.pth"
  if [[ ! -e "${TRAIN_CHECKPOINT}" ]]; then
    TEMP_CHECKPOINT="$(mktemp "${PRIVATE_INPUT_DIR}/.checkpoint.XXXXXX")"
    cp --reflink=auto -- "${CHECKPOINT}" "${TEMP_CHECKPOINT}"
    OBSERVED_SIZE="$(stat -c %s "${TEMP_CHECKPOINT}")"
    OBSERVED_SHA256="$(sha256sum "${TEMP_CHECKPOINT}" | awk '{print $1}')"
    [[ "${OBSERVED_SIZE}" == "${LOCKED_CHECKPOINT_SIZE}" && \
       "${OBSERVED_SHA256}" == "${LOCKED_CHECKPOINT_SHA256}" ]] || {
      printf 'copied checkpoint does not match locked size/SHA-256\n' >&2
      rm -f -- "${TEMP_CHECKPOINT}"
      exit 2
    }
    chmod 400 "${TEMP_CHECKPOINT}"
    mv -- "${TEMP_CHECKPOINT}" "${TRAIN_CHECKPOINT}"
  fi
  [[ -f "${TRAIN_CHECKPOINT}" && ! -L "${TRAIN_CHECKPOINT}" ]] || {
    printf 'trusted checkpoint copy is invalid\n' >&2
    exit 2
  }
  OBSERVED_SIZE="$(stat -c %s "${TRAIN_CHECKPOINT}")"
  OBSERVED_SHA256="$(sha256sum "${TRAIN_CHECKPOINT}" | awk '{print $1}')"
  FILE_MODE="$(stat -c %a "${TRAIN_CHECKPOINT}")"
  [[ "${OBSERVED_SIZE}" == "${LOCKED_CHECKPOINT_SIZE}" && \
     "${OBSERVED_SHA256}" == "${LOCKED_CHECKPOINT_SHA256}" && \
     "${FILE_MODE}" == "400" ]] || {
    printf 'trusted checkpoint copy failed final integrity/permission check\n' >&2
    exit 2
  }
fi
PYTHON_RESOLVED="$(command -v "${PYTHON_BIN}")" || {
  printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
}
export PATH="$(dirname "${PYTHON_RESOLVED}"):${PATH}"
[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || { printf '--gpus must be a positive integer\n' >&2; exit 2; }
[[ "${PER_DEVICE_BATCH}" =~ ^[1-9][0-9]*$ ]] || { printf '--per-device-batch must be a positive integer\n' >&2; exit 2; }
[[ "${SEED}" =~ ^[0-9]+$ ]] || { printf '--seed must be a non-negative integer\n' >&2; exit 2; }
if [[ "${DRY_RUN}" == false ]]; then
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
    printf 'CUDA_VISIBLE_DEVICES must explicitly select the requested idle GPUs\n' >&2
    exit 2
  }
  PYTHONPATH="${PROJECT_DIR}:${MMDET_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/gpu_guard.py" --expected-count "${GPUS}"
fi

DENOMINATOR=$((GPUS * PER_DEVICE_BATCH))
if (( DENOMINATOR > TARGET_GLOBAL_BATCH || TARGET_GLOBAL_BATCH % DENOMINATOR != 0 )); then
  printf 'gpus * per-device-batch must divide the locked global batch %d; got %d\n' \
    "${TARGET_GLOBAL_BATCH}" "${DENOMINATOR}" >&2
  exit 2
fi
ACCUMULATIVE_COUNTS=$((TARGET_GLOBAL_BATCH / DENOMINATOR))
WARMUP_ITERS=$((500 * ACCUMULATIVE_COUNTS))
CONFIG_PATH="${PROJECT_DIR}/configs/${CONFIG_NAME}"

set_variant_options() {
  local name="$1"
  VARIANT_OPTIONS=()
  case "${name}" in
    phase0_parent)
      VARIANT_OPTIONS=()
      ;;
    parent_ref)
      VARIANT_OPTIONS=(
        'model.seed_operator=none' \
        'model.bbox_head.loss_seed_weight=0.0' \
        'model.bbox_head.loss_ref_weight=0.5' \
        'model.bbox_head.loss_role_div_weight=0.0')
      ;;
    generic)
      VARIANT_OPTIONS=(
        'model.seed_operator=generic' \
        'model.bbox_head.loss_seed_weight=0.5' \
        'model.bbox_head.loss_ref_weight=0.5' \
        'model.bbox_head.loss_role_div_weight=0.0')
      ;;
    rqgo)
      VARIANT_OPTIONS=(
        'model.seed_operator=rqgo' \
        'model.bbox_head.loss_seed_weight=0.5' \
        'model.bbox_head.loss_ref_weight=0.5' \
        'model.bbox_head.loss_role_div_weight=0.005')
      ;;
  esac
}

if [[ "${DRY_RUN}" == false ]]; then
  for run_variant in "${VARIANTS[@]}"; do
    WORK_DIR="${WORK_ROOT}/${DATASET}/${run_variant}/seed_${SEED}"
    PYTHONPATH="${PROJECT_DIR}:${MMDET_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_work_dir.py" "${WORK_DIR}"
  done
  PREFLIGHT_DIR="${WORK_ROOT}/${DATASET}/preflight"
  mkdir -p "${PREFLIGHT_DIR}"
  PREFLIGHT=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/server_preflight.py"
    --dataset "${DATASET}"
    --mmdet-root "${MMDET_ROOT}"
    --data-root "${DATA_ROOT}"
    --checkpoint "${TRAIN_CHECKPOINT}"
    --bert-root "${BERT_ROOT}"
    --work-root "${WORK_ROOT}"
    --output "${PREFLIGHT_DIR}/preflight.json"
  )
  PREFLIGHT+=(--checkpoint-sha256 "${CHECKPOINT_SHA256}")
  PYTHONPATH="${PROJECT_DIR}:${MMDET_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PREFLIGHT[@]}"
fi

for run_variant in "${VARIANTS[@]}"; do
  WORK_DIR="${WORK_ROOT}/${DATASET}/${run_variant}/seed_${SEED}"
  set_variant_options "${run_variant}"
  ACTIVE_CONFIG_PATH="${CONFIG_PATH}"
  if [[ "${run_variant}" == "phase0_parent" ]]; then
    ACTIVE_CONFIG_PATH="${PROJECT_DIR}/configs/${PHASE0_CONFIG_NAME}"
  fi
  CFG_OPTIONS=()
  if [[ "${run_variant}" != "phase0_parent" ]]; then
    CFG_OPTIONS=("${VARIANT_OPTIONS[@]}")
  fi
  CFG_OPTIONS+=(
    "load_from=${TRAIN_CHECKPOINT}"
    "model.language_model.name=${BERT_ROOT}"
    "train_dataloader.batch_size=${PER_DEVICE_BATCH}"
    "train_dataloader.dataset.data_root=${DATA_ROOT}"
    "train_dataloader.dataset.pipeline.5.tokenizer_name=${BERT_ROOT}"
    "val_dataloader.dataset.data_root=${DATA_ROOT}"
    "val_evaluator.ann_file=${DATA_ROOT}/mdetr_annotations/${VAL_ANN}"
    "optim_wrapper.accumulative_counts=${ACCUMULATIVE_COUNTS}"
    "randomness.seed=${SEED}"
    "randomness.deterministic=True"
  )
  if [[ "${run_variant}" != "phase0_parent" ]]; then
    CFG_OPTIONS+=(
      "param_scheduler.0.end=${WARMUP_ITERS}"
      "custom_hooks.0.warmup_iters=${WARMUP_ITERS}")
  fi
  if [[ "${USE_AMP}" == true && "${run_variant}" != "phase0_parent" ]]; then
    CFG_OPTIONS+=("optim_wrapper.type=AmpOptimWrapper" "optim_wrapper.loss_scale=dynamic")
  fi

  COMMAND=(
    bash "${MMDET_ROOT}/tools/dist_train.sh" "${ACTIVE_CONFIG_PATH}" "${GPUS}"
    --work-dir "${WORK_DIR}"
    --cfg-options "${CFG_OPTIONS[@]}"
  )
  printf 'Variant %s, effective global batch %d, accumulation %d, warmup iterations %d\n' \
    "${run_variant}" "${TARGET_GLOBAL_BATCH}" "${ACCUMULATIVE_COUNTS}" "${WARMUP_ITERS}"
  printf 'Command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == false ]]; then
    CURRENT_CHECKPOINT_SHA256="$(sha256sum "${TRAIN_CHECKPOINT}" | awk '{print $1}')"
    [[ "${CURRENT_CHECKPOINT_SHA256}" == "${LOCKED_CHECKPOINT_SHA256}" ]] || {
      printf 'trusted checkpoint changed before variant launch\n' >&2
      exit 2
    }
    if [[ -L "${WORK_DIR}/command.txt" ]]; then
      printf 'refusing symlink command log: %s\n' "${WORK_DIR}/command.txt" >&2
      exit 2
    fi
    printf '%q ' "${COMMAND[@]}" > "${WORK_DIR}/command.txt"
    printf '\n' >> "${WORK_DIR}/command.txt"
    (
      cd "${MMDET_ROOT}"
      PYTHONPATH="${PROJECT_DIR}:${MMDET_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${COMMAND[@]}"
    )
  fi
done
