#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
from scripts.multimodal.run_groundingdino_refcoco_predictions import _image_path_from_record
from scripts.multimodal.score_clip_crop_same_candidates import (
    _crop_normalized_box,
    _encode_text_features,
    _load_clip_dependencies,
    _load_expressions,
    _resolve_device,
)
from scripts.multimodal.score_groundingdino_same_candidates import (
    _load_predictions,
    _load_sample_records,
    _load_source_ids,
    _pairwise_iou,
    _safe_rate,
    _write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rank GroundingDINO proposal boxes by CLIP crop-text similarity. "
            "All metrics are proposal-conditioned, so recall@K means a positive "
            "IoU>=threshold proposal appears in the CLIP-ranked top K."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--dataset-name", default="refcoco")
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--split", required=True)
    parser.add_argument("--expressions-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-template", default="COCO_train2014_{image_number:012d}.jpg")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--crop-batch-size", type=int, default=128)
    parser.add_argument(
        "--prediction-box-format",
        choices=("xyxy_normalized", "cxcywh_normalized"),
        default="xyxy_normalized",
    )
    parser.add_argument("--max-predictions-per-sample", type=int, default=32)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    payload = score_groundingdino_proposal_clip_similarity(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def score_groundingdino_proposal_clip_similarity(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    layout = MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version)
    root = layout.root
    source_ids = _load_source_ids(root, args.split)
    sample_records = _load_sample_records(root, args.split)
    bbox_targets = np.load(root / "supervision" / f"bbox_targets_{args.split}.npy")
    if bbox_targets.shape != (len(source_ids), 4):
        raise ValueError(f"bbox_targets must have shape [{len(source_ids)}, 4]")
    predictions = _load_predictions(
        args.predictions,
        box_format=args.prediction_box_format,
        max_predictions_per_sample=args.max_predictions_per_sample,
    )
    expressions = _load_expressions(args.expressions_jsonl)
    missing_expressions = [source_id for source_id in source_ids if source_id not in expressions]
    if missing_expressions:
        raise ValueError("missing expressions for source_id values: " + ", ".join(missing_expressions[:20]))

    torch, transformers, image_module = _load_clip_dependencies()
    device = _resolve_device(torch, args.device)
    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32
    model = transformers.CLIPModel.from_pretrained(args.clip_model, revision=args.revision).to(device=device, dtype=dtype)
    model.eval()
    tokenizer = transformers.CLIPTokenizer.from_pretrained(args.clip_model, revision=args.revision)
    image_processor = transformers.CLIPImageProcessor.from_pretrained(args.clip_model, revision=args.revision)

    ordered_texts = [expressions[source_id] for source_id in source_ids]
    text_features = _encode_text_features(args, ordered_texts, tokenizer, model, torch, device)
    proposal_scores, crop_failure_count = _score_proposal_crops(
        args,
        source_ids,
        sample_records,
        predictions,
        text_features,
        image_processor,
        model,
        torch,
        image_module,
        device,
        dtype,
    )
    return score_precomputed_groundingdino_proposal_scores(
        args,
        source_ids=source_ids,
        bbox_targets=bbox_targets,
        predictions=predictions,
        proposal_similarity_scores=proposal_scores,
        crop_failure_count=crop_failure_count,
    )


def score_precomputed_groundingdino_proposal_scores(
    args: argparse.Namespace,
    *,
    source_ids: list[str] | None = None,
    bbox_targets: np.ndarray | None = None,
    predictions: dict[str, Any] | None = None,
    proposal_similarity_scores: dict[str, list[float]] | None = None,
    crop_failure_count: int = 0,
) -> dict[str, Any]:
    if source_ids is None or bbox_targets is None or predictions is None or proposal_similarity_scores is None:
        layout = MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version)
        source_ids = _load_source_ids(layout.root, args.split)
        bbox_targets = np.load(layout.root / "supervision" / f"bbox_targets_{args.split}.npy")
        predictions = _load_predictions(
            args.predictions,
            box_format=args.prediction_box_format,
            max_predictions_per_sample=args.max_predictions_per_sample,
        )
        proposal_similarity_scores = {}
    threshold = float(getattr(args, "iou_threshold", 0.5))
    rows = []
    metric_rows = []
    missing_prediction_count = 0
    empty_proposal_count = 0
    no_valid_score_count = 0
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
        boxes = prediction.boxes if prediction is not None else np.zeros((0, 4), dtype=np.float32)
        scores = prediction.scores if prediction is not None else np.zeros((0,), dtype=np.float32)
        boxes, scores = _sort_predictions_by_score(boxes, scores)
        similarity = np.asarray(proposal_similarity_scores.get(source_id, []), dtype=np.float32).reshape(-1)
        if similarity.shape[0] < boxes.shape[0]:
            similarity = np.pad(similarity, (0, boxes.shape[0] - similarity.shape[0]), constant_values=-1.0)
        similarity = similarity[: boxes.shape[0]]
        if boxes.shape[0] == 0:
            empty_proposal_count += 1
            selected_index = -1
            selected_iou = 0.0
            first_positive_rank = None
            ranked_indices: list[int] = []
            ious = np.zeros((0,), dtype=np.float32)
        else:
            ious = _pairwise_iou(boxes, bbox_targets[row_index].reshape(1, 4)).reshape(-1)
            valid_indices = [index for index, value in enumerate(similarity) if np.isfinite(value) and float(value) > -1.0]
            if not valid_indices:
                no_valid_score_count += 1
            ranked_indices = sorted(valid_indices, key=lambda index: (-float(similarity[index]), index))
            selected_index = ranked_indices[0] if ranked_indices else -1
            selected_iou = float(ious[selected_index]) if selected_index >= 0 else 0.0
            first_positive_rank = _first_positive_rank(ranked_indices, ious, threshold)
        hit_at_1 = selected_iou >= threshold
        hit_at_5 = first_positive_rank is not None and first_positive_rank <= 5
        acc_at_0_5 = selected_iou >= 0.5
        acc_at_0_7 = selected_iou >= 0.7
        hit_at_1_count += int(hit_at_1)
        hit_at_5_count += int(hit_at_5)
        acc_at_0_5_count += int(acc_at_0_5)
        acc_at_0_7_count += int(acc_at_0_7)
        reciprocal_rank_sum += 0.0 if first_positive_rank is None else 1.0 / float(first_positive_rank)
        selected_iou_sum += selected_iou
        rows.append(
            {
                "source_id": source_id,
                "split": args.split,
                "row_index": row_index,
                "proposal_count": int(boxes.shape[0]),
                "selected_proposal_index": selected_index,
                "selected_iou": selected_iou,
                "hit_at_1": hit_at_1,
                "hit_at_5": hit_at_5,
                "first_positive_rank": first_positive_rank,
                "oracle_best_iou": float(np.max(ious)) if ious.size else 0.0,
                "difficulty_bucket": _difficulty_bucket(float(np.max(ious)) if ious.size else 0.0),
                "proposal_scores": [float(value) for value in similarity],
                "proposal_detector_scores": [float(value) for value in scores],
            }
        )

    sample_count = len(source_ids)
    model_name = getattr(args, "model_name", "groundingdino_proposals_clip_similarity")
    summary = {
        "artifact_type": "groundingdino_proposal_reranker_summary",
        "comparison_scope": "groundingdino_proposal_conditioned_reranker",
        "dataset": args.dataset_name,
        "split": args.split,
        "version": args.version,
        "model": model_name,
        "clip_model": getattr(args, "clip_model", None),
        "sample_count": sample_count,
        "prediction_count": len(predictions),
        "missing_prediction_count": missing_prediction_count,
        "empty_proposal_count": empty_proposal_count,
        "empty_proposal_rate": _safe_rate(empty_proposal_count, sample_count),
        "crop_failure_count": int(crop_failure_count),
        "no_valid_score_count": no_valid_score_count,
        "recall_at_1": _safe_rate(hit_at_1_count, sample_count),
        "recall_at_5": _safe_rate(hit_at_5_count, sample_count),
        "acc_at_0_5": _safe_rate(acc_at_0_5_count, sample_count),
        "acc_at_0_7": _safe_rate(acc_at_0_7_count, sample_count),
        "mean_iou": _safe_rate(selected_iou_sum, sample_count),
        "mrr": _safe_rate(reciprocal_rank_sum, sample_count),
        "positive_proposal_iou_threshold": threshold,
        "scoring_rule": "rank GroundingDINO proposals by cosine(CLIP(crop(proposal_box)), CLIP(expression))",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "per_sample_scores.jsonl", rows)
    for metric_name in ("acc_at_0_5", "acc_at_0_7", "recall_at_1", "recall_at_5", "mean_iou", "mrr"):
        metric_rows.append(
            {
                "artifact_type": "external_alignment_raw_metric",
                "evidence_scope": "groundingdino_proposal_conditioned_reranking",
                "dataset": args.dataset_name,
                "task": "phrase_region_grounding",
                "model": model_name,
                "split": args.split,
                "seed": None,
                "metric_name": metric_name,
                "score": float(summary[metric_name]),
                "higher_is_better": True,
                "same_candidate_source": False,
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


def _score_proposal_crops(
    args: argparse.Namespace,
    source_ids: list[str],
    sample_records: dict[str, dict[str, Any]],
    predictions: dict[str, Any],
    text_features: Any,
    image_processor: Any,
    model: Any,
    torch: Any,
    image_module: Any,
    device: Any,
    dtype: Any,
) -> tuple[dict[str, list[float]], int]:
    scores_by_source = {source_id: [] for source_id in source_ids}
    crop_batch = []
    crop_keys: list[tuple[str, int]] = []
    crop_failure_count = 0

    def flush_batch() -> None:
        nonlocal crop_batch, crop_keys
        if not crop_batch:
            return
        inputs = image_processor(images=crop_batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
            row_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
            row_indices = torch.tensor([row_lookup[source_id] for source_id, _ in crop_keys], dtype=torch.long, device=device)
            batch_text_features = text_features.index_select(0, row_indices)
            similarities = (image_features * batch_text_features).sum(dim=-1).detach().cpu().numpy()
        for (source_id, proposal_index), score in zip(crop_keys, similarities):
            scores_by_source[source_id][proposal_index] = float(score)
        crop_batch = []
        crop_keys = []

    for source_id in source_ids:
        prediction = predictions.get(source_id)
        if prediction is None:
            continue
        boxes, _ = _sort_predictions_by_score(prediction.boxes, prediction.scores)
        record = sample_records[source_id]
        scores_by_source[source_id] = [-1.0 for _ in range(int(boxes.shape[0]))]
        image_path = _image_path_from_record(record, image_root=args.image_root, image_template=args.image_template)
        with image_module.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
            for proposal_index, box in enumerate(boxes):
                crop = _crop_normalized_box(image, box)
                if crop is None:
                    crop_failure_count += 1
                    continue
                crop_batch.append(crop)
                crop_keys.append((source_id, proposal_index))
                if len(crop_batch) >= args.crop_batch_size:
                    flush_batch()
    flush_batch()
    return scores_by_source, crop_failure_count


def _validate_args(args: argparse.Namespace) -> None:
    if args.text_batch_size <= 0:
        raise ValueError("--text-batch-size must be positive")
    if args.crop_batch_size <= 0:
        raise ValueError("--crop-batch-size must be positive")
    if args.iou_threshold <= 0.0 or args.iou_threshold > 1.0:
        raise ValueError("--iou-threshold must be in (0, 1]")


def _sort_predictions_by_score(boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if boxes.shape[0] == 0:
        return boxes, scores
    order = np.argsort(-scores, kind="stable")
    return boxes[order], scores[order]


def _first_positive_rank(ranked_indices: list[int], ious: np.ndarray, threshold: float) -> int | None:
    for rank, index in enumerate(ranked_indices, start=1):
        if float(ious[index]) >= threshold:
            return rank
    return None


def _difficulty_bucket(best_iou: float) -> str:
    if best_iou >= 0.8:
        return "easy"
    if best_iou >= 0.5:
        return "medium"
    return "hard"


if __name__ == "__main__":
    raise SystemExit(main())
