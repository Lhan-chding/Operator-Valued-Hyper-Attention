#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/multimodal_cmu_mosei_tanso_mechanism_selfmm_official.json}"
CONTROLLED_REPORT="${CONTROLLED_REPORT:-outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-outputs/multimodal/cmu_mosei_tanso_mechanism_selfmm_official}"
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

mkdir -p "${ARTIFACT_ROOT}"

echo "[cmu-mosei-tanso-mechanism] training full proof-plan matrix"
"${PYTHON_BIN}" scripts/multimodal/run_public_main.py "${CONFIG}" \
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
  | tee "${ARTIFACT_ROOT}/run_public_main_payload.json"

echo "[cmu-mosei-tanso-mechanism] validating non-smoke public-main artifacts"
"${PYTHON_BIN}" scripts/multimodal/validate_public_main_artifacts.py \
  --config "${CONFIG}" \
  --raw-metrics "${ARTIFACT_ROOT}/raw_metrics.jsonl" \
  --diagnostics "${ARTIFACT_ROOT}/diagnostics.jsonl" \
  --robustness-rows "${ARTIFACT_ROOT}/robustness_rows.jsonl" \
  --split test \
  | tee "${ARTIFACT_ROOT}/artifact_validation.json"

for BASELINE in \
  raw_tanso_mlp \
  ovha_tanso_no_source_gate \
  ovha_tanso_no_hyper_adapter \
  ovha_tanso_no_operator_memory \
  ovha_tanso_no_gate_aux \
  ovha_lrio_tanso \
  ovha_all_candidates_exploratory \
  concat_fusion
do
  echo "[cmu-mosei-tanso-mechanism] summarizing ovha_tanso_full vs ${BASELINE}"
  "${PYTHON_BIN}" scripts/multimodal/summarize_public_results.py \
    "${ARTIFACT_ROOT}/raw_metrics.jsonl" \
    --full-model ovha_tanso_full \
    --baseline-model "${BASELINE}" \
    > "${ARTIFACT_ROOT}/summary_vs_${BASELINE}.json"
done

echo "[cmu-mosei-tanso-mechanism] done"
echo "raw metrics: ${ARTIFACT_ROOT}/raw_metrics.jsonl"
echo "per-sample predictions: ${ARTIFACT_ROOT}/per_sample_predictions.jsonl"
echo "artifact validation: ${ARTIFACT_ROOT}/artifact_validation.json"
