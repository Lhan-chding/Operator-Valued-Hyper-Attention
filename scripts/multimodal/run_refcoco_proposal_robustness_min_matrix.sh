#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
if [[ -d ".venv" && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RUN_ROOT="${RUN_ROOT:-outputs/multimodal/refcoco_proposal_robustness_min_matrix_$(date +%Y%m%d_%H%M%S)}"
CONTROLLED_REPORT="${CONTROLLED_REPORT:-outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json}"
CACHE_ROOT="${CACHE_ROOT:-data/multimodal_cache}"
CACHE_VERSION="${CACHE_VERSION:-v0.1}"
DEVICE="${DEVICE:-cuda}"
TRAIN_STEPS="${TRAIN_STEPS:-7000}"
BASELINE_TRAIN_STEPS="${BASELINE_TRAIN_STEPS:-7000}"
D_MODEL="${D_MODEL:-64}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-500}"
ALLOW_MISSING_CACHES="${ALLOW_MISSING_CACHES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "[missing] python interpreter: set PYTHON_BIN or activate the project environment"
    exit 2
  fi
fi

declare -a MATRIX=(
  "refcoco_pool_a_gdino_swint|refcoco_gdino_proposals|configs/multimodal_refcoco_gdino_proposal_rerankers.json|val testA testB"
  "refcoco_pool_b_gdino_swinb|${ALT_POOL_DATASET_NAME:-refcoco_gdino_swinb_proposals}|configs/multimodal_refcoco_gdino_swinb_proposal_rerankers.json|val testA testB"
  "refcoco_plus_gdino_swint|refcoco_plus_gdino_proposals|configs/multimodal_refcoco_plus_gdino_proposal_rerankers.json|val testA testB"
  "refcocog_gdino_swint|refcocog_gdino_proposals|configs/multimodal_refcocog_gdino_proposal_rerankers.json|val test"
)

echo "[start] $(date)"
echo "[run_root] $RUN_ROOT"
echo "[device] $DEVICE"
echo "[cache_root] $CACHE_ROOT"
echo "[cache_version] $CACHE_VERSION"
echo "[controlled_report] $CONTROLLED_REPORT"
echo "[python] $PYTHON_BIN"

if [[ ! -f "$CONTROLLED_REPORT" ]]; then
  echo "[missing] controlled report: $CONTROLLED_REPORT"
  exit 2
fi

mkdir -p "$RUN_ROOT"/logs

missing=0
for entry in "${MATRIX[@]}"; do
  IFS='|' read -r cell dataset_name config_path split_list <<< "$entry"
  data_card="$CACHE_ROOT/$dataset_name/$CACHE_VERSION/data_card.json"
  if [[ ! -f "$config_path" ]]; then
    echo "[missing] config for $cell: $config_path" | tee -a "$RUN_ROOT/missing_caches.txt"
    missing=1
  fi
  if [[ ! -f "$data_card" ]]; then
    echo "[missing] cache for $cell: $data_card" | tee -a "$RUN_ROOT/missing_caches.txt"
    missing=1
  fi
done

if [[ "$missing" == "1" && "$ALLOW_MISSING_CACHES" != "1" ]]; then
  echo "[stop] missing caches or configs. Build proposal caches first, or set ALLOW_MISSING_CACHES=1 to skip missing cells."
  exit 2
fi

for entry in "${MATRIX[@]}"; do
  IFS='|' read -r cell dataset_name config_path split_list <<< "$entry"
  data_card="$CACHE_ROOT/$dataset_name/$CACHE_VERSION/data_card.json"
  if [[ ! -f "$data_card" ]]; then
    echo "[skip] $cell missing cache: $data_card"
    continue
  fi
  for split in $split_list; do
    artifact_root="$RUN_ROOT/$cell/$split"
    mkdir -p "$artifact_root"
    echo "{\"cell\":\"$cell\",\"dataset_name\":\"$dataset_name\",\"config\":\"$config_path\",\"split\":\"$split\",\"artifact_root\":\"$artifact_root\"}" >> "$RUN_ROOT/matrix_manifest.jsonl"
    echo "[run] cell=$cell split=$split $(date)"
    "$PYTHON_BIN" scripts/multimodal/run_public_main.py \
      "$config_path" \
      --controlled-report "$CONTROLLED_REPORT" \
      --artifact-root "$artifact_root" \
      --train-steps "$TRAIN_STEPS" \
      --baseline-train-steps "$BASELINE_TRAIN_STEPS" \
      --train-split train \
      --selection-split val \
      --eval-split "$split" \
      --d-model "$D_MODEL" \
      --batch-size "$BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --eval-interval "$EVAL_INTERVAL" \
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
      --weight-decay "$WEIGHT_DECAY" \
      --device "$DEVICE" \
      --only-baseline gdino_score_box_aware_cross_attention_reranker \
      --progress-interval "$PROGRESS_INTERVAL" \
      2>&1 | tee "$RUN_ROOT/logs/${cell}_${split}.log"
  done
done

echo "[done] $(date)"
echo "[outputs] $RUN_ROOT"
