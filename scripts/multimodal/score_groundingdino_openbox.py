#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
from scripts.multimodal.score_groundingdino_same_candidates import (
    _load_predictions,
    _load_source_ids,
    _pairwise_iou,
    _safe_rate,
    _write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score GroundingDINO in original open-box REC mode against RefCOCO target boxes."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--dataset-name", default="refcoco")
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-box-format",
        choices=("xyxy_normalized", "cxcywh_normalized"),
        default="xyxy_normalized",
    )
    parser.add_argument("--max-predictions-per-sample", type=int, default=None)
    payload = score_groundingdino_openbox(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def score_groundingdino_openbox(args: argparse.Namespace) -> dict[str, Any]:
    layout = MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version)
    root = layout.root
    source_ids = _load_source_ids(root, args.split)
    bbox_targets = np.load(root / "supervision" / f"bbox_targets_{args.split}.npy")
    if bbox_targets.shape != (len(source_ids), 4):
        raise ValueError(f"bbox_targets must have shape [{len(source_ids)}, 4]")
    predictions = _load_predictions(
        args.predictions,
        box_format=args.prediction_box_format,
        max_predictions_per_sample=args.max_predictions_per_sample,
    )
    rows = []
    missing_prediction_count = 0
    empty_prediction_count = 0
    acc_at_0_5_count = 0
    acc_at_0_7_count = 0
    selected_iou_sum = 0.0
    for row_index, source_id in enumerate(source_ids):
        prediction = predictions.get(source_id)
        if prediction is None:
            missing_prediction_count += 1
        boxes = prediction.boxes if prediction is not None else np.zeros((0, 4), dtype=np.float32)
        scores = prediction.scores if prediction is not None else np.zeros((0,), dtype=np.float32)
        if boxes.shape[0] == 0:
            empty_prediction_count += 1
            selected_index = -1
            selected_score = 0.0
            selected_iou = 0.0
        else:
            selected_index = int(np.argmax(scores))
            selected_score = float(scores[selected_index])
            selected_iou = float(_pairwise_iou(boxes[selected_index].reshape(1, 4), bbox_targets[row_index].reshape(1, 4))[0, 0])
        acc_at_0_5 = selected_iou >= 0.5
        acc_at_0_7 = selected_iou >= 0.7
        acc_at_0_5_count += int(acc_at_0_5)
        acc_at_0_7_count += int(acc_at_0_7)
        selected_iou_sum += selected_iou
        rows.append(
            {
                "source_id": source_id,
                "split": args.split,
                "row_index": row_index,
                "selected_prediction_index": selected_index,
                "selected_prediction_score": selected_score,
                "selected_iou": selected_iou,
                "acc_at_0_5": acc_at_0_5,
                "acc_at_0_7": acc_at_0_7,
                "prediction_count": int(boxes.shape[0]),
            }
        )
    sample_count = len(source_ids)
    summary = {
        "artifact_type": "groundingdino_openbox_summary",
        "comparison_scope": "groundingdino_original_openbox_rec",
        "dataset": args.dataset_name,
        "split": args.split,
        "version": args.version,
        "sample_count": sample_count,
        "prediction_count": len(predictions),
        "missing_prediction_count": missing_prediction_count,
        "empty_prediction_count": empty_prediction_count,
        "acc_at_0_5": _safe_rate(acc_at_0_5_count, sample_count),
        "acc_at_0_7": _safe_rate(acc_at_0_7_count, sample_count),
        "mean_iou": _safe_rate(selected_iou_sum, sample_count),
        "scoring_rule": "select highest-score GroundingDINO prediction and compare open box to GT bbox",
        "prediction_box_format": args.prediction_box_format,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "per_sample_openbox_scores.jsonl", rows)
    metric_rows = [
        {
            "artifact_type": "external_alignment_raw_metric",
            "evidence_scope": "groundingdino_openbox_alignment",
            "dataset": args.dataset_name,
            "task": "phrase_region_grounding",
            "model": "groundingdino_openbox",
            "split": args.split,
            "seed": None,
            "metric_name": metric_name,
            "score": float(summary[metric_name]),
            "higher_is_better": True,
            "same_candidate_source": False,
            "same_feature_source": False,
            "public_metrics": summary,
        }
        for metric_name in ("acc_at_0_5", "acc_at_0_7", "mean_iou")
    ]
    _write_jsonl(args.output_dir / "raw_metrics.jsonl", metric_rows)
    summary["output_dir"] = str(args.output_dir)
    summary["artifacts"] = {
        "per_sample_openbox_scores": str(args.output_dir / "per_sample_openbox_scores.jsonl"),
        "raw_metrics": str(args.output_dir / "raw_metrics.jsonl"),
        "summary": str(args.output_dir / "summary.json"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
