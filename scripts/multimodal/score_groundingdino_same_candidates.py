#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score GroundingDINO predictions by projecting them onto an existing RefCOCO candidate set."
    )
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL predictions keyed by source_id.")
    parser.add_argument("--cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--dataset-name", default="refcoco")
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-box-format",
        choices=("xyxy_normalized", "cxcywh_normalized"),
        default="xyxy_normalized",
        help="Format of prediction boxes in the JSONL file.",
    )
    parser.add_argument("--max-predictions-per-sample", type=int, default=None)
    payload = score_groundingdino_same_candidates(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def score_groundingdino_same_candidates(args: argparse.Namespace) -> dict[str, Any]:
    layout = MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version)
    root = layout.root
    source_ids = _load_source_ids(root, args.split)
    sample_records = _load_sample_records(root, args.split)
    candidate_boxes = np.load(root / "supervision" / f"candidate_region_boxes_{args.split}.npy")
    bbox_targets = np.load(root / "supervision" / f"bbox_targets_{args.split}.npy")
    region_targets = np.load(root / "supervision" / f"region_targets_{args.split}.npy").reshape(-1)
    region_mask = _load_region_mask(root, args.split, candidate_boxes)
    _validate_cache_shapes(source_ids, candidate_boxes, bbox_targets, region_targets, region_mask)

    predictions = _load_predictions(
        args.predictions,
        box_format=args.prediction_box_format,
        max_predictions_per_sample=args.max_predictions_per_sample,
    )
    per_sample_rows = []
    metric_rows = []
    missing_prediction_count = 0
    empty_prediction_count = 0
    hit_at_1_count = 0
    hit_at_5_count = 0
    acc_at_0_5_count = 0
    acc_at_0_7_count = 0
    reciprocal_rank_sum = 0.0
    selected_iou_sum = 0.0

    for row_index, source_id in enumerate(source_ids):
        prediction = predictions.get(source_id)
        if prediction is None:
            missing_prediction_count += 1
        pred_boxes = prediction.boxes if prediction is not None else np.zeros((0, 4), dtype=np.float32)
        pred_scores = prediction.scores if prediction is not None else np.zeros((0,), dtype=np.float32)
        if pred_boxes.shape[0] == 0:
            empty_prediction_count += 1
        valid_mask = region_mask[row_index].astype(bool, copy=False)
        scores = _candidate_scores(candidate_boxes[row_index], valid_mask, pred_boxes, pred_scores)
        target_index = int(region_targets[row_index])
        selected_index, ranked_indices = _rank_candidates(scores, valid_mask, has_predictions=pred_boxes.shape[0] > 0)
        selected_iou = (
            float(_single_iou(candidate_boxes[row_index, selected_index], bbox_targets[row_index]))
            if selected_index >= 0
            else 0.0
        )
        target_rank = _target_rank(ranked_indices, target_index)
        hit_at_1 = selected_index == target_index
        hit_at_5 = target_rank is not None and target_rank <= 5
        acc_at_0_5 = selected_iou >= 0.5
        acc_at_0_7 = selected_iou >= 0.7
        hit_at_1_count += int(hit_at_1)
        hit_at_5_count += int(hit_at_5)
        acc_at_0_5_count += int(acc_at_0_5)
        acc_at_0_7_count += int(acc_at_0_7)
        reciprocal_rank_sum += 0.0 if target_rank is None else 1.0 / float(target_rank)
        selected_iou_sum += selected_iou
        per_sample_rows.append(
            {
                "source_id": source_id,
                "split": args.split,
                "row_index": row_index,
                "target_index": target_index,
                "selected_index": selected_index,
                "selected_iou": selected_iou,
                "hit_at_1": hit_at_1,
                "hit_at_5": hit_at_5,
                "target_rank": target_rank,
                "prediction_count": int(pred_boxes.shape[0]),
                "candidate_scores": [float(value) for value in scores],
                "sample_record": sample_records.get(source_id, {}),
            }
        )

    sample_count = len(source_ids)
    summary = {
        "artifact_type": "groundingdino_same_candidate_summary",
        "comparison_scope": "groundingdino_same_candidate_scorer",
        "dataset": args.dataset_name,
        "split": args.split,
        "version": args.version,
        "sample_count": sample_count,
        "prediction_count": len(predictions),
        "missing_prediction_count": missing_prediction_count,
        "empty_prediction_count": empty_prediction_count,
        "recall_at_1": _safe_rate(hit_at_1_count, sample_count),
        "recall_at_5": _safe_rate(hit_at_5_count, sample_count),
        "acc_at_0_5": _safe_rate(acc_at_0_5_count, sample_count),
        "acc_at_0_7": _safe_rate(acc_at_0_7_count, sample_count),
        "mean_iou": _safe_rate(selected_iou_sum, sample_count),
        "mrr": _safe_rate(reciprocal_rank_sum, sample_count),
        "scoring_rule": "candidate_score=max_prediction(prediction_score * IoU(candidate_box, prediction_box))",
        "prediction_box_format": args.prediction_box_format,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "per_sample_scores.jsonl", per_sample_rows)
    for metric_name, score, higher_is_better in (
        ("acc_at_0_5", summary["acc_at_0_5"], True),
        ("recall_at_1", summary["recall_at_1"], True),
        ("recall_at_5", summary["recall_at_5"], True),
        ("mean_iou", summary["mean_iou"], True),
        ("mrr", summary["mrr"], True),
    ):
        metric_rows.append(
            {
                "artifact_type": "external_alignment_raw_metric",
                "evidence_scope": "groundingdino_same_candidate_alignment",
                "dataset": args.dataset_name,
                "task": "phrase_region_grounding",
                "model": "groundingdino_same_candidate",
                "split": args.split,
                "seed": None,
                "metric_name": metric_name,
                "score": float(score),
                "higher_is_better": higher_is_better,
                "same_candidate_source": True,
                "same_feature_source": False,
                "public_metrics": summary,
            }
        )
    _write_jsonl(args.output_dir / "raw_metrics.jsonl", metric_rows)
    summary["output_dir"] = str(args.output_dir)
    summary["artifacts"] = {
        "per_sample_scores": str(args.output_dir / "per_sample_scores.jsonl"),
        "raw_metrics": str(args.output_dir / "raw_metrics.jsonl"),
        "summary": str(args.output_dir / "summary.json"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


class _Prediction:
    def __init__(self, boxes: np.ndarray, scores: np.ndarray):
        self.boxes = boxes
        self.scores = scores


def _load_source_ids(root: Path, split: str) -> list[str]:
    source_ids_path = root / "provenance" / f"source_ids_{split}.txt"
    if source_ids_path.exists():
        return [line.strip() for line in source_ids_path.read_text().splitlines() if line.strip()]
    return [record["source_id"] for record in _iter_sample_records(root, split)]


def _load_sample_records(root: Path, split: str) -> dict[str, dict[str, Any]]:
    return {record["source_id"]: record for record in _iter_sample_records(root, split)}


def _iter_sample_records(root: Path, split: str) -> Iterable[dict[str, Any]]:
    path = root / "provenance" / f"sample_records_{split}.jsonl"
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                record = json.loads(line)
                if "source_id" not in record:
                    raise ValueError(f"{path} line {line_number} missing source_id")
                yield record


def _load_region_mask(root: Path, split: str, candidate_boxes: np.ndarray) -> np.ndarray:
    path = root / "masks" / f"region_mask_{split}.npy"
    if path.exists():
        return np.load(path).astype(bool, copy=False)
    return _box_areas(candidate_boxes) > 0.0


def _validate_cache_shapes(
    source_ids: list[str],
    candidate_boxes: np.ndarray,
    bbox_targets: np.ndarray,
    region_targets: np.ndarray,
    region_mask: np.ndarray,
) -> None:
    sample_count = len(source_ids)
    if candidate_boxes.ndim != 3 or candidate_boxes.shape[-1] != 4:
        raise ValueError("candidate_region_boxes must have shape [sample, candidate, 4]")
    if bbox_targets.shape != (sample_count, 4):
        raise ValueError(f"bbox_targets must have shape [{sample_count}, 4]")
    if region_targets.shape != (sample_count,):
        raise ValueError(f"region_targets must have shape [{sample_count}]")
    if region_mask.shape != candidate_boxes.shape[:2]:
        raise ValueError("region_mask must match candidate_region_boxes sample/candidate axes")
    if candidate_boxes.shape[0] != sample_count:
        raise ValueError("candidate_region_boxes sample count must match source_ids")


def _load_predictions(
    path: Path,
    *,
    box_format: str,
    max_predictions_per_sample: int | None,
) -> dict[str, _Prediction]:
    predictions: dict[str, _Prediction] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = row.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"{path} line {line_number} missing source_id")
            if source_id in predictions:
                raise ValueError(f"duplicate prediction source_id: {source_id}")
            boxes, scores = _prediction_arrays(row, box_format=box_format)
            if max_predictions_per_sample is not None:
                boxes = boxes[:max_predictions_per_sample]
                scores = scores[:max_predictions_per_sample]
            predictions[source_id] = _Prediction(boxes, scores)
    return predictions


def _prediction_arrays(row: dict[str, Any], *, box_format: str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(row.get("predictions"), list):
        items = row["predictions"]
        boxes = [item.get("box") or item.get("bbox") for item in items if isinstance(item, dict)]
        scores = [float(item.get("score", item.get("logit", 1.0))) for item in items if isinstance(item, dict)]
    else:
        boxes = row.get("boxes") or row.get("boxes_xyxy_normalized") or row.get("boxes_cxcywh_normalized") or []
        scores = row.get("scores") or row.get("logits") or [1.0 for _ in boxes]
    box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    if score_array.shape[0] != box_array.shape[0]:
        raise ValueError(f"prediction scores length must match boxes for source_id: {row.get('source_id')}")
    if box_format == "cxcywh_normalized":
        box_array = _cxcywh_to_xyxy(box_array)
    else:
        box_array = _normalize_xyxy(box_array)
    return box_array, score_array


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    width = np.maximum(boxes[:, 2], 0.0)
    height = np.maximum(boxes[:, 3], 0.0)
    converted = np.stack((cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height), axis=1)
    return _normalize_xyxy(converted)


def _normalize_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    return np.clip(np.stack((x1, y1, x2, y2), axis=1), 0.0, 1.0)


def _candidate_scores(
    candidate_boxes: np.ndarray,
    valid_mask: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
) -> list[float]:
    scores = np.zeros((candidate_boxes.shape[0],), dtype=np.float32)
    if pred_boxes.shape[0] == 0:
        return [float(value) for value in scores]
    ious = _pairwise_iou(candidate_boxes, pred_boxes)
    weighted = ious * pred_scores.reshape(1, -1)
    scores = np.max(weighted, axis=1)
    scores[~valid_mask] = -1.0
    return [float(value) for value in scores]


def _rank_candidates(scores: list[float], valid_mask: np.ndarray, *, has_predictions: bool) -> tuple[int, list[int]]:
    if not has_predictions:
        return -1, []
    valid_indices = [index for index, is_valid in enumerate(valid_mask) if bool(is_valid)]
    ranked = sorted(valid_indices, key=lambda index: (-float(scores[index]), index))
    return (ranked[0] if ranked else -1), ranked


def _target_rank(ranked_indices: list[int], target_index: int) -> int | None:
    for rank, index in enumerate(ranked_indices, start=1):
        if index == target_index:
            return rank
    return None


def _pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    x1 = np.maximum(left[:, None, 0], right[None, :, 0])
    y1 = np.maximum(left[:, None, 1], right[None, :, 1])
    x2 = np.minimum(left[:, None, 2], right[None, :, 2])
    y2 = np.minimum(left[:, None, 3], right[None, :, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = _box_areas(left)[:, None] + _box_areas(right)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=union > 0.0)


def _single_iou(left: np.ndarray, right: np.ndarray) -> float:
    return float(_pairwise_iou(left.reshape(1, 4), right.reshape(1, 4))[0, 0])


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, boxes[..., 2] - boxes[..., 0]) * np.maximum(0.0, boxes[..., 3] - boxes[..., 1])


def _safe_rate(numerator: float, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))


if __name__ == "__main__":
    raise SystemExit(main())
