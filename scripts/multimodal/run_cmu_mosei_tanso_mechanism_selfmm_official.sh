#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/multimodal_cmu_mosei_tanso_mechanism_selfmm_official.json}"
CACHE_ROOT="${CACHE_ROOT:-}"
CONTROLLED_REPORT="${CONTROLLED_REPORT:-outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-outputs/multimodal/cmu_mosei_tanso_mechanism_missing_only}"
DEVICE="${DEVICE:-cuda}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BASELINE_TRAIN_STEPS="${BASELINE_TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
D_MODEL="${D_MODEL:-128}"
MEMORY_TOKENS="${MEMORY_TOKENS:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
PROGRESS_INTERVAL="${OVHA_PUBLIC_MAIN_PROGRESS_INTERVAL:-500}"
RUN_SCOPE="${RUN_SCOPE:-missing_only}"
ONLY_BASELINES_DEFAULT="raw_tanso_mlp ovha_tanso_no_source_gate ovha_tanso_no_hyper_adapter ovha_tanso_no_operator_memory ovha_tanso_no_gate_aux"
ONLY_BASELINES="${ONLY_BASELINES:-${ONLY_BASELINES_DEFAULT}}"
REFERENCE_RAW_METRICS="${REFERENCE_RAW_METRICS:-}"
FULL_MODEL_NAME="${FULL_MODEL_NAME:-ovha_tanso_full}"
SKIP_CACHE_VALIDATION="${SKIP_CACHE_VALIDATION:-0}"

mkdir -p "${ARTIFACT_ROOT}"

RUN_ARGS=()
CACHE_ARGS=()
BASELINE_ARRAY=()
if [[ -n "${CACHE_ROOT}" ]]; then
  CACHE_ARGS+=(--cache-root "${CACHE_ROOT}")
fi
if [[ "${SKIP_CACHE_VALIDATION}" == "1" ]]; then
  CACHE_ARGS+=(--skip-cache-validation)
fi
if [[ "${RUN_SCOPE}" == "full_matrix" ]]; then
  echo "[cmu-mosei-tanso-mechanism] training full proof-plan matrix"
else
  echo "[cmu-mosei-tanso-mechanism] training missing proof-plan baselines only"
  RUN_ARGS+=(--skip-main-model)
  read -r -a BASELINE_ARRAY <<< "${ONLY_BASELINES}"
  for BASELINE in "${BASELINE_ARRAY[@]}"; do
    RUN_ARGS+=(--only-baseline "${BASELINE}")
  done
fi

"${PYTHON_BIN}" scripts/multimodal/run_public_main.py "${CONFIG}" \
  "${CACHE_ARGS[@]}" \
  --controlled-report "${CONTROLLED_REPORT}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --train-steps "${TRAIN_STEPS}" \
  --baseline-train-steps "${BASELINE_TRAIN_STEPS}" \
  --train-split train \
  --selection-split val \
  --eval-split test \
  --d-model "${D_MODEL}" \
  --memory-tokens "${MEMORY_TOKENS}" \
  --learning-rate "${LEARNING_RATE}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --device "${DEVICE}" \
  --progress-interval "${PROGRESS_INTERVAL}" \
  "${RUN_ARGS[@]}" \
  | tee "${ARTIFACT_ROOT}/run_public_main_payload.json"

if [[ "${RUN_SCOPE}" == "full_matrix" ]]; then
  echo "[cmu-mosei-tanso-mechanism] validating non-smoke public-main artifacts"
  "${PYTHON_BIN}" scripts/multimodal/validate_public_main_artifacts.py \
    --config "${CONFIG}" \
    --raw-metrics "${ARTIFACT_ROOT}/raw_metrics.jsonl" \
    --diagnostics "${ARTIFACT_ROOT}/diagnostics.jsonl" \
    --robustness-rows "${ARTIFACT_ROOT}/robustness_rows.jsonl" \
    --split test \
    | tee "${ARTIFACT_ROOT}/artifact_validation.json"
else
  echo "[cmu-mosei-tanso-mechanism] skipped full-matrix artifact validation for targeted continuation output"
fi

if [[ -n "${REFERENCE_RAW_METRICS}" ]]; then
  COMBINED_RAW_METRICS="${ARTIFACT_ROOT}/combined_raw_metrics.jsonl"
  cat "${REFERENCE_RAW_METRICS}" "${ARTIFACT_ROOT}/raw_metrics.jsonl" > "${COMBINED_RAW_METRICS}"
  if [[ "${#BASELINE_ARRAY[@]}" -eq 0 ]]; then
    read -r -a BASELINE_ARRAY <<< "${ONLY_BASELINES}"
  fi
  for BASELINE in "${BASELINE_ARRAY[@]}"; do
    echo "[cmu-mosei-tanso-mechanism] summarizing ${FULL_MODEL_NAME} vs ${BASELINE}"
    "${PYTHON_BIN}" scripts/multimodal/summarize_public_results.py \
      "${COMBINED_RAW_METRICS}" \
      --full-model "${FULL_MODEL_NAME}" \
      --baseline-model "${BASELINE}" \
      > "${ARTIFACT_ROOT}/summary_vs_${BASELINE}.json"
  done
else
  echo "[cmu-mosei-tanso-mechanism] set REFERENCE_RAW_METRICS to summarize against the existing ovha_tanso_full run"
fi

echo "[cmu-mosei-tanso-mechanism] done"
echo "raw metrics: ${ARTIFACT_ROOT}/raw_metrics.jsonl"
echo "per-sample predictions: ${ARTIFACT_ROOT}/per_sample_predictions.jsonl"
if [[ -n "${REFERENCE_RAW_METRICS}" ]]; then
  echo "combined raw metrics: ${ARTIFACT_ROOT}/combined_raw_metrics.jsonl"
fi
