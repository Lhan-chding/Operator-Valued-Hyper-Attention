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
        description=(
            "Compute GroundingDINO proposal upper-bound metrics for RefCOCO. "
            "This does not score a reranker; it reports whether the proposal pool "
            "contains a usable target box before CLIP/MLP/cross-attention/OVHA reranking."
        )
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
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20, 32])
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    payload = score_groundingdino_proposal_oracle(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def score_groundingdino_proposal_oracle(args: argparse.Namespace) -> dict[str, Any]:
    top_k_values = _normalize_top_k(args.top_k)
    if args.iou_threshold <= 0.0 or args.iou_threshold > 1.0:
        raise ValueError("--iou-threshold must be in (0, 1]")

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
    per_sample_rows = []
    oracle_hits = {k: 0 for k in top_k_values}
    bucket_counts = {"easy": 0, "medium": 0, "hard": 0}
    bucket_iou_sums = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
    missing_prediction_count = 0
    empty_proposal_count = 0
    proposal_count_sum = 0
    best_iou_sum = 0.0

    max_k = max(top_k_values)
    for row_index, source_id in enumerate(source_ids):
        prediction = predictions.get(source_id)
        if prediction is None:
            missing_prediction_count += 1
        boxes = prediction.boxes if prediction is not None else np.zeros((0, 4), dtype=np.float32)
        scores = prediction.scores if prediction is not None else np.zeros((0,), dtype=np.float32)
        boxes, scores = _sort_predictions_by_score(boxes, scores)
        boxes = boxes[:max_k]
        scores = scores[:max_k]
        proposal_count = int(boxes.shape[0])
        proposal_count_sum += proposal_count

        if proposal_count == 0:
            empty_proposal_count += 1
            ious = np.zeros((0,), dtype=np.float32)
            best_iou = 0.0
            best_rank = None
            best_index = -1
            best_score = 0.0
        else:
            ious = _pairwise_iou(boxes, bbox_targets[row_index].reshape(1, 4)).reshape(-1)
            best_index = int(np.argmax(ious))
            best_iou = float(ious[best_index])
            best_rank = best_index + 1
            best_score = float(scores[best_index])

        best_iou_sum += best_iou
        for k in top_k_values:
            oracle_hits[k] += int(bool(ious[:k].size and float(np.max(ious[:k])) >= args.iou_threshold))
        bucket = _difficulty_bucket(best_iou)
        bucket_counts[bucket] += 1
        bucket_iou_sums[bucket] += best_iou
        per_sample_rows.append(
            {
                "source_id": source_id,
                "split": args.split,
                "row_index": row_index,
                "proposal_count": proposal_count,
                "best_proposal_index": best_index,
                "best_proposal_rank": best_rank,
                "best_proposal_score": best_score,
                "oracle_best_iou": best_iou,
                "difficulty_bucket": bucket,
                "oracle_hit_at_k": {
                    str(k): bool(ious[:k].size and float(np.max(ious[:k])) >= args.iou_threshold)
                    for k in top_k_values
                },
            }
        )

    sample_count = len(source_ids)
    proposal_oracle_recall_at_k = {str(k): _safe_rate(oracle_hits[k], sample_count) for k in top_k_values}
    positive_k = str(max_k)
    summary = {
        "artifact_type": "groundingdino_proposal_oracle_summary",
        "comparison_scope": "groundingdino_proposal_upper_bound",
        "dataset": args.dataset_name,
        "split": args.split,
        "version": args.version,
        "sample_count": sample_count,
        "prediction_count": len(predictions),
        "proposal_count": proposal_count_sum,
        "missing_prediction_count": missing_prediction_count,
        "empty_proposal_count": empty_proposal_count,
        "empty_proposal_rate": _safe_rate(empty_proposal_count, sample_count),
        "proposal_oracle_recall_at_k": proposal_oracle_recall_at_k,
        "oracle_best_iou": _safe_rate(best_iou_sum, sample_count),
        "positive_candidate_rate": proposal_oracle_recall_at_k[positive_k],
        "positive_candidate_definition": f"at least one proposal among top-{max_k} has IoU >= {args.iou_threshold}",
        "difficulty_definition": {
            "easy": "oracle_best_iou >= 0.8",
            "medium": "0.5 <= oracle_best_iou < 0.8",
            "hard": "oracle_best_iou < 0.5",
        },
        "bucket_counts": bucket_counts,
        "bucket_rates": {bucket: _safe_rate(count, sample_count) for bucket, count in bucket_counts.items()},
        "bucket_mean_best_iou": {
            bucket: _safe_rate(bucket_iou_sums[bucket], bucket_counts[bucket])
            for bucket in ("easy", "medium", "hard")
        },
        "top_k": top_k_values,
        "iou_threshold": args.iou_threshold,
        "prediction_box_format": args.prediction_box_format,
        "scoring_rule": "proposal oracle: max IoU between top-K GroundingDINO proposals and GT bbox",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "per_sample_proposal_oracle.jsonl", per_sample_rows)
    metric_rows = _metric_rows(args, summary, top_k_values)
    _write_jsonl(args.output_dir / "raw_metrics.jsonl", metric_rows)
    summary["output_dir"] = str(args.output_dir)
    summary["artifacts"] = {
        "per_sample_proposal_oracle": str(args.output_dir / "per_sample_proposal_oracle.jsonl"),
        "raw_metrics": str(args.output_dir / "raw_metrics.jsonl"),
        "summary": str(args.output_dir / "summary.json"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _normalize_top_k(values: list[int]) -> list[int]:
    top_k_values = sorted(set(int(value) for value in values))
    if not top_k_values or top_k_values[0] <= 0:
        raise ValueError("--top-k values must be positive")
    return top_k_values


def _sort_predictions_by_score(boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if boxes.shape[0] == 0:
        return boxes, scores
    order = np.argsort(-scores, kind="stable")
    return boxes[order], scores[order]


def _difficulty_bucket(best_iou: float) -> str:
    if best_iou >= 0.8:
        return "easy"
    if best_iou >= 0.5:
        return "medium"
    return "hard"


def _metric_rows(args: argparse.Namespace, summary: dict[str, Any], top_k_values: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in top_k_values:
        rows.append(
            _raw_metric_row(
                args,
                metric_name=f"proposal_oracle_recall_at_{k}",
                score=summary["proposal_oracle_recall_at_k"][str(k)],
                summary=summary,
            )
        )
    for metric_name in ("oracle_best_iou", "empty_proposal_rate", "positive_candidate_rate"):
        rows.append(_raw_metric_row(args, metric_name=metric_name, score=summary[metric_name], summary=summary))
    for bucket in ("easy", "medium", "hard"):
        rows.append(
            _raw_metric_row(
                args,
                metric_name=f"{bucket}_bucket_rate",
                score=summary["bucket_rates"][bucket],
                summary=summary,
            )
        )
    return rows


def _raw_metric_row(
    args: argparse.Namespace,
    *,
    metric_name: str,
    score: float,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": "external_alignment_raw_metric",
        "evidence_scope": "groundingdino_proposal_upper_bound",
        "dataset": args.dataset_name,
        "task": "phrase_region_grounding",
        "model": "groundingdino_proposal_oracle",
        "split": args.split,
        "seed": None,
        "metric_name": metric_name,
        "score": float(score),
        "higher_is_better": metric_name != "empty_proposal_rate",
        "same_candidate_source": False,
        "same_feature_source": False,
        "public_metrics": summary,
    }


if __name__ == "__main__":
    raise SystemExit(main())
