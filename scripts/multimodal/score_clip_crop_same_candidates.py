#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
from scripts.multimodal.run_groundingdino_refcoco_predictions import _image_path_from_record
from scripts.multimodal.score_groundingdino_same_candidates import (
    _load_region_mask,
    _load_sample_records,
    _load_source_ids,
    _rank_candidates,
    _safe_rate,
    _single_iou,
    _target_rank,
    _validate_cache_shapes,
    _write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score RefCOCO fixed candidates by direct CLIP crop-text similarity. "
            "This is a ReCLIP-style same-candidate external baseline: every valid "
            "candidate box is cropped from the original image, encoded with CLIP, "
            "and ranked by cosine similarity to the referring expression."
        )
    )
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
    parser.add_argument("--limit", type=int)
    payload = score_clip_crop_same_candidates(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def score_clip_crop_same_candidates(args: argparse.Namespace) -> dict[str, Any]:
    _validate_runtime_args(args)
    layout = MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version)
    root = layout.root
    source_ids, sample_records, candidate_boxes, bbox_targets, region_targets, region_mask = _load_cache_inputs(
        root, args.split, args.limit
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
    similarity_scores, crop_failure_count = _score_candidate_crops(
        args,
        source_ids,
        sample_records,
        candidate_boxes,
        region_mask,
        text_features,
        image_processor,
        model,
        torch,
        image_module,
        device,
        dtype,
    )
    return score_precomputed_clip_crop_same_candidates(
        args,
        similarity_scores=similarity_scores,
        crop_failure_count=crop_failure_count,
        source_ids=source_ids,
        sample_records=sample_records,
        candidate_boxes=candidate_boxes,
        bbox_targets=bbox_targets,
        region_targets=region_targets,
        region_mask=region_mask,
    )


def score_precomputed_clip_crop_same_candidates(
    args: argparse.Namespace,
    *,
    similarity_scores: np.ndarray,
    crop_failure_count: int,
    source_ids: list[str] | None = None,
    sample_records: dict[str, dict[str, Any]] | None = None,
    candidate_boxes: np.ndarray | None = None,
    bbox_targets: np.ndarray | None = None,
    region_targets: np.ndarray | None = None,
    region_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if source_ids is None or sample_records is None or candidate_boxes is None or bbox_targets is None or region_targets is None or region_mask is None:
        layout = MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version)
        source_ids, sample_records, candidate_boxes, bbox_targets, region_targets, region_mask = _load_cache_inputs(
            layout.root, args.split, getattr(args, "limit", None)
        )
    if similarity_scores.shape != candidate_boxes.shape[:2]:
        raise ValueError("similarity_scores must match candidate_region_boxes sample/candidate axes")

    per_sample_rows = []
    metric_rows = []
    hit_at_1_count = 0
    hit_at_5_count = 0
    acc_at_0_5_count = 0
    acc_at_0_7_count = 0
    reciprocal_rank_sum = 0.0
    selected_iou_sum = 0.0
    no_valid_score_count = 0

    for row_index, source_id in enumerate(source_ids):
        valid_mask = region_mask[row_index].astype(bool, copy=False)
        scores = similarity_scores[row_index].astype(np.float32, copy=True)
        scores[~valid_mask] = -1.0
        has_valid_score = bool(np.any(valid_mask & np.isfinite(scores) & (scores > -1.0)))
        if not has_valid_score:
            no_valid_score_count += 1
        score_list = [float(value) if np.isfinite(value) else -1.0 for value in scores]
        target_index = int(region_targets[row_index])
        selected_index, ranked_indices = _rank_candidates(score_list, valid_mask, has_predictions=has_valid_score)
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
                "candidate_scores": score_list,
                "sample_record": sample_records.get(source_id, {}),
            }
        )

    sample_count = len(source_ids)
    model_name = getattr(args, "model_name", "clip_crop_similarity")
    summary = {
        "artifact_type": "clip_crop_same_candidate_summary",
        "comparison_scope": "clip_crop_same_candidate_scorer",
        "dataset": args.dataset_name,
        "split": args.split,
        "version": args.version,
        "sample_count": sample_count,
        "model": model_name,
        "clip_model": getattr(args, "clip_model", None),
        "crop_failure_count": int(crop_failure_count),
        "no_valid_score_count": no_valid_score_count,
        "recall_at_1": _safe_rate(hit_at_1_count, sample_count),
        "recall_at_5": _safe_rate(hit_at_5_count, sample_count),
        "acc_at_0_5": _safe_rate(acc_at_0_5_count, sample_count),
        "acc_at_0_7": _safe_rate(acc_at_0_7_count, sample_count),
        "mean_iou": _safe_rate(selected_iou_sum, sample_count),
        "mrr": _safe_rate(reciprocal_rank_sum, sample_count),
        "scoring_rule": "candidate_score=cosine(CLIP(crop(candidate_box)), CLIP(expression))",
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
                "evidence_scope": "clip_crop_same_candidate_alignment",
                "dataset": args.dataset_name,
                "task": "phrase_region_grounding",
                "model": model_name,
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


def _load_cache_inputs(
    root: Path,
    split: str,
    limit: int | None,
) -> tuple[list[str], dict[str, dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_ids = _load_source_ids(root, split)
    sample_records = _load_sample_records(root, split)
    candidate_boxes = np.load(root / "supervision" / f"candidate_region_boxes_{split}.npy")
    bbox_targets = np.load(root / "supervision" / f"bbox_targets_{split}.npy")
    region_targets = np.load(root / "supervision" / f"region_targets_{split}.npy").reshape(-1)
    region_mask = _load_region_mask(root, split, candidate_boxes)
    _validate_cache_shapes(source_ids, candidate_boxes, bbox_targets, region_targets, region_mask)
    if limit is not None:
        source_ids = source_ids[:limit]
        candidate_boxes = candidate_boxes[:limit]
        bbox_targets = bbox_targets[:limit]
        region_targets = region_targets[:limit]
        region_mask = region_mask[:limit]
    return source_ids, sample_records, candidate_boxes, bbox_targets, region_targets, region_mask


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if getattr(args, "text_batch_size", 1) <= 0:
        raise ValueError("--text-batch-size must be positive")
    if getattr(args, "crop_batch_size", 1) <= 0:
        raise ValueError("--crop-batch-size must be positive")
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")


def _load_clip_dependencies():
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError("CLIP crop same-candidate scoring requires torch, transformers, and pillow") from exc
    return torch, transformers, Image


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _load_expressions(path: Path) -> dict[str, str]:
    expressions: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        source_id = row.get("source_id")
        expression = row.get("expression", row.get("text", row.get("sentence")))
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{path} line {line_number} missing source_id")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"{path} line {line_number} missing expression/text")
        if source_id in expressions:
            raise ValueError(f"duplicate expression source_id: {source_id}")
        expressions[source_id] = expression.strip()
    return expressions


def _encode_text_features(args: argparse.Namespace, texts: list[str], tokenizer: Any, model: Any, torch: Any, device: Any) -> Any:
    features = []
    with torch.no_grad():
        for start in range(0, len(texts), args.text_batch_size):
            batch = texts[start : start + args.text_batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            batch_features = model.get_text_features(**inputs)
            features.append(torch.nn.functional.normalize(batch_features.float(), dim=-1).cpu())
    return torch.cat(features, dim=0).to(device)


def _score_candidate_crops(
    args: argparse.Namespace,
    source_ids: list[str],
    sample_records: dict[str, dict[str, Any]],
    candidate_boxes: np.ndarray,
    region_mask: np.ndarray,
    text_features: Any,
    image_processor: Any,
    model: Any,
    torch: Any,
    image_module: Any,
    device: Any,
    dtype: Any,
) -> tuple[np.ndarray, int]:
    similarity_scores = np.full(candidate_boxes.shape[:2], -1.0, dtype=np.float32)
    crop_batch = []
    crop_keys: list[tuple[int, int]] = []
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
            row_indices = torch.tensor([row_index for row_index, _ in crop_keys], dtype=torch.long, device=device)
            batch_text_features = text_features.index_select(0, row_indices)
            similarities = (image_features * batch_text_features).sum(dim=-1).detach().cpu().numpy()
        for (_, (row_index, candidate_index)), score in zip(enumerate(crop_keys), similarities):
            similarity_scores[row_index, candidate_index] = float(score)
        crop_batch = []
        crop_keys = []

    for row_index, source_id in enumerate(source_ids):
        record = sample_records[source_id]
        image_path = _image_path_from_record(record, image_root=args.image_root, image_template=args.image_template)
        with image_module.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
            for candidate_index, is_valid in enumerate(region_mask[row_index]):
                if not bool(is_valid):
                    continue
                crop = _crop_normalized_box(image, candidate_boxes[row_index, candidate_index])
                if crop is None:
                    crop_failure_count += 1
                    continue
                crop_batch.append(crop)
                crop_keys.append((row_index, candidate_index))
                if len(crop_batch) >= args.crop_batch_size:
                    flush_batch()
    flush_batch()
    return similarity_scores, crop_failure_count


def _crop_normalized_box(image: Any, box: np.ndarray) -> Any | None:
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in box]
    left = max(0, min(width, math.floor(x1 * width)))
    upper = max(0, min(height, math.floor(y1 * height)))
    right = max(0, min(width, math.ceil(x2 * width)))
    lower = max(0, min(height, math.ceil(y2 * height)))
    if right <= left or lower <= upper:
        return None
    return image.crop((left, upper, right, lower)).copy()


if __name__ == "__main__":
    raise SystemExit(main())
