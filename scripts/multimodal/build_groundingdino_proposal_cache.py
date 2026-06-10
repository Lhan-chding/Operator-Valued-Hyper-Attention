#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from moat_ovha_torch.data.multimodal.cache_schema import default_data_card, file_sha256
from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task
from scripts.multimodal.run_groundingdino_refcoco_predictions import _image_path_from_record
from scripts.multimodal.score_clip_crop_same_candidates import (
    _crop_normalized_box,
    _load_clip_dependencies,
    _load_expressions,
    _resolve_device,
)
from scripts.multimodal.extract_refcoco_clip_features import _project_text_tokens
from scripts.multimodal.score_groundingdino_proposal_clip_similarity import _sort_predictions_by_score
from scripts.multimodal.score_groundingdino_same_candidates import (
    _load_predictions,
    _load_sample_records,
    _load_source_ids,
    _pairwise_iou,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a RefCOCO public cache whose region candidates are GroundingDINO top-K proposals."
    )
    parser.add_argument("--source-cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--source-dataset-name", default="refcoco")
    parser.add_argument("--source-version", default="v0.1")
    parser.add_argument("--output-cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--output-dataset-name", default="refcoco_gdino_proposals")
    parser.add_argument("--output-version", default="v0.1")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "testA", "testB"])
    parser.add_argument("--predictions-template", required=True)
    parser.add_argument("--expressions-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-template", default="COCO_train2014_{image_number:012d}.jpg")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--crop-batch-size", type=int, default=192)
    parser.add_argument("--max-text-length", type=int, default=77)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--max-proposals-per-sample", type=int, default=32)
    parser.add_argument("--prediction-box-format", choices=("xyxy_normalized", "cxcywh_normalized"), default="xyxy_normalized")
    parser.add_argument("--allow-missing-predictions", action="store_true")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Update only text token fields in an existing proposal cache; use this to repair token-axis validation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(normalize=True)
    payload = build_groundingdino_proposal_cache(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_groundingdino_proposal_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_proposals_per_sample <= 0:
        raise ValueError("--max-proposals-per-sample must be positive")
    source_root = Path(args.source_cache_root) / args.source_dataset_name / args.source_version
    output_root = Path(args.output_cache_root) / args.output_dataset_name / args.output_version
    if output_root.exists() and not args.text_only:
        if not args.overwrite:
            raise ValueError(f"output cache already exists; pass --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    for folder in ("masks", "positions", "provenance", "supervision", "token_fields"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    split_source_ids = {split: _load_source_ids(source_root, split) for split in args.splits}
    expressions = _load_expressions(args.expressions_jsonl)
    missing_expressions = [
        source_id
        for split in args.splits
        for source_id in split_source_ids[split]
        if source_id not in expressions
    ]
    if missing_expressions:
        raise ValueError("missing expressions for source_id values: " + ", ".join(missing_expressions[:20]))

    torch, transformers, image_module = _load_clip_dependencies()
    device = _resolve_device(torch, args.device)
    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32
    model = transformers.CLIPModel.from_pretrained(args.clip_model, revision=args.revision).to(device=device, dtype=dtype)
    model.eval()
    tokenizer = transformers.CLIPTokenizer.from_pretrained(args.clip_model, revision=args.revision)
    image_processor = transformers.CLIPImageProcessor.from_pretrained(args.clip_model, revision=args.revision)

    summaries = []
    for split in args.splits:
        summaries.append(
            _write_split(
                args,
                split=split,
                source_root=source_root,
                output_root=output_root,
                source_ids=split_source_ids[split],
                expressions=expressions,
                tokenizer=tokenizer,
                image_processor=image_processor,
                model=model,
                torch=torch,
                image_module=image_module,
                device=device,
                dtype=dtype,
            )
        )
    if not args.text_only:
        _write_common_files(args, output_root, split_source_ids)
    _write_checksums(output_root)
    return {
        "ok": True,
        "artifact_type": "groundingdino_proposal_public_cache",
        "source_dataset": args.source_dataset_name,
        "source_version": args.source_version,
        "dataset": args.output_dataset_name,
        "version": args.output_version,
        "output_root": str(output_root),
        "splits": summaries,
    }


def _write_split(
    args: argparse.Namespace,
    *,
    split: str,
    source_root: Path,
    output_root: Path,
    source_ids: list[str],
    expressions: dict[str, str],
    tokenizer: Any,
    image_processor: Any,
    model: Any,
    torch: Any,
    image_module: Any,
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    sample_records = _load_sample_records(source_root, split)
    bbox_targets = np.load(source_root / "supervision" / f"bbox_targets_{split}.npy").astype(np.float32, copy=False)
    predictions_path = Path(str(args.predictions_template).format(split=split))
    predictions = _load_predictions(
        predictions_path,
        box_format=args.prediction_box_format,
        max_predictions_per_sample=args.max_proposals_per_sample,
    )
    missing_predictions = [source_id for source_id in source_ids if source_id not in predictions]
    if missing_predictions and not args.allow_missing_predictions:
        raise ValueError(
            f"{split} predictions missing {len(missing_predictions)} source_id values; "
            f"first missing: {', '.join(missing_predictions[:10])}"
        )

    texts = [expressions[source_id] for source_id in source_ids]
    text_features, text_mask = _encode_text_token_features(args, texts, tokenizer, model, torch, device)
    if args.text_only:
        _write_array(output_root / "token_fields" / f"text_{split}.npy", text_features)
        _write_array(output_root / "positions" / f"text_pos_{split}.npy", _text_positions(text_features.shape[0], text_features.shape[1]))
        _write_array(output_root / "masks" / f"text_mask_{split}.npy", text_mask)
        _patch_text_manifest(output_root / "token_fields" / f"manifest_{split}.json", split)
        return {
            "split": split,
            "sample_count": len(source_ids),
            "text_token_count": int(text_features.shape[1]),
            "mode": "text_only",
        }
    proposal_boxes, proposal_detector_scores, region_mask, proposal_counts = _proposal_arrays(
        source_ids,
        predictions,
        max_k=int(args.max_proposals_per_sample),
    )
    region_features, crop_failure_count = _encode_proposal_crops(
        args,
        split=split,
        source_ids=source_ids,
        sample_records=sample_records,
        proposal_boxes=proposal_boxes,
        region_mask=region_mask,
        image_processor=image_processor,
        model=model,
        torch=torch,
        image_module=image_module,
        device=device,
        dtype=dtype,
    )
    best_ious = _best_ious(proposal_boxes, region_mask, bbox_targets)
    target_indices = np.argmax(best_ious, axis=1).astype(np.int64)
    target_labels = np.zeros((len(source_ids), int(args.max_proposals_per_sample)), dtype=np.float32)
    target_labels[np.arange(len(source_ids)), target_indices] = 1.0

    _write_array(output_root / "token_fields" / f"text_{split}.npy", text_features)
    _write_array(output_root / "positions" / f"text_pos_{split}.npy", _text_positions(text_features.shape[0], text_features.shape[1]))
    _write_array(output_root / "masks" / f"text_mask_{split}.npy", text_mask)
    _write_array(output_root / "token_fields" / f"region_{split}.npy", region_features)
    _write_array(output_root / "positions" / f"region_pos_{split}.npy", _region_positions(proposal_boxes))
    _write_array(output_root / "masks" / f"region_mask_{split}.npy", region_mask)
    _write_array(output_root / "supervision" / f"task_labels_{split}.npy", target_labels)
    _write_array(output_root / "supervision" / f"bbox_targets_{split}.npy", bbox_targets)
    _write_array(output_root / "supervision" / f"candidate_region_boxes_{split}.npy", proposal_boxes)
    _write_array(output_root / "supervision" / f"region_targets_{split}.npy", target_indices.reshape(-1, 1))
    (output_root / "token_fields" / f"manifest_{split}.json").write_text(
        json.dumps(
            {
                "text": {
                    "x": f"token_fields/text_{split}.npy",
                    "pos": f"positions/text_pos_{split}.npy",
                    "mask": f"masks/text_mask_{split}.npy",
                },
                "region": {
                    "x": f"token_fields/region_{split}.npy",
                    "pos": f"positions/region_pos_{split}.npy",
                    "mask": f"masks/region_mask_{split}.npy",
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    (output_root / "provenance" / f"source_ids_{split}.txt").write_text("\n".join(source_ids) + "\n")
    _write_sample_records(
        output_root / "provenance" / f"sample_records_{split}.jsonl",
        split,
        source_ids,
        sample_records,
        proposal_boxes,
        proposal_detector_scores,
        target_indices,
    )
    (output_root / "provenance" / f"failed_samples_{split}.jsonl").write_text("")
    _write_alignment_pairs(output_root / "supervision" / f"alignment_pairs_{split}.parquet", split, source_ids, sample_records, target_indices)
    _write_target_histogram(output_root / "supervision" / f"target_slot_histogram_by_valid_count_{split}.json", target_indices, region_mask)
    _write_corruption(output_root / "supervision" / f"corruption_{split}.parquet", split, source_ids)
    positive = best_ious[np.arange(len(source_ids)), target_indices] >= 0.5
    return {
        "split": split,
        "sample_count": len(source_ids),
        "missing_prediction_count": len(missing_predictions),
        "empty_proposal_count": int(np.count_nonzero(proposal_counts == 0)),
        "crop_failure_count": int(crop_failure_count),
        "positive_candidate_rate": float(np.mean(positive)) if len(source_ids) else 0.0,
        "oracle_best_iou": float(np.mean(best_ious[np.arange(len(source_ids)), target_indices])) if len(source_ids) else 0.0,
    }


def _proposal_arrays(source_ids: list[str], predictions: dict[str, Any], *, max_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes_out = np.zeros((len(source_ids), max_k, 4), dtype=np.float32)
    scores_out = np.zeros((len(source_ids), max_k), dtype=np.float32)
    mask_out = np.zeros((len(source_ids), max_k), dtype=bool)
    counts = np.zeros((len(source_ids),), dtype=np.int64)
    for row_index, source_id in enumerate(source_ids):
        prediction = predictions.get(source_id)
        boxes = prediction.boxes if prediction is not None else np.zeros((0, 4), dtype=np.float32)
        scores = prediction.scores if prediction is not None else np.zeros((0,), dtype=np.float32)
        boxes, scores = _sort_predictions_by_score(boxes, scores)
        count = min(int(boxes.shape[0]), max_k)
        counts[row_index] = count
        if count:
            boxes_out[row_index, :count] = boxes[:count]
            scores_out[row_index, :count] = scores[:count]
            mask_out[row_index, :count] = True
        else:
            mask_out[row_index, 0] = True
    return boxes_out, scores_out, mask_out, counts


def _encode_text_token_features(
    args: argparse.Namespace,
    texts: list[str],
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    outputs = []
    masks = []
    with torch.no_grad():
        for start in range(0, len(texts), int(args.text_batch_size)):
            batch = texts[start : start + int(args.text_batch_size)]
            encoded = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=int(args.max_text_length),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            features = model.text_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
            ).last_hidden_state
            features = _project_text_tokens(model, features)
            if bool(args.normalize):
                features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
            features = features * attention_mask.unsqueeze(-1).to(dtype=features.dtype)
            outputs.append(features.detach().cpu().float().numpy())
            masks.append(attention_mask.detach().cpu().bool().numpy())
    return (
        np.concatenate(outputs, axis=0).astype(np.float32, copy=False),
        np.concatenate(masks, axis=0).astype(bool, copy=False),
    )


def _text_positions(sample_count: int, token_count: int) -> np.ndarray:
    return np.broadcast_to(
        np.arange(token_count, dtype=np.float32).reshape(1, token_count, 1),
        (sample_count, token_count, 1),
    ).copy()


def _patch_text_manifest(path: Path, split: str) -> None:
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        manifest = {}
    manifest["text"] = {
        "x": f"token_fields/text_{split}.npy",
        "pos": f"positions/text_pos_{split}.npy",
        "mask": f"masks/text_mask_{split}.npy",
    }
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")


def _encode_proposal_crops(
    args: argparse.Namespace,
    *,
    split: str,
    source_ids: list[str],
    sample_records: dict[str, dict[str, Any]],
    proposal_boxes: np.ndarray,
    region_mask: np.ndarray,
    image_processor: Any,
    model: Any,
    torch: Any,
    image_module: Any,
    device: Any,
    dtype: Any,
) -> tuple[np.ndarray, int]:
    feature_dim = int(model.config.projection_dim)
    features = np.zeros((len(source_ids), int(args.max_proposals_per_sample), feature_dim), dtype=np.float32)
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
            image_features = torch.nn.functional.normalize(image_features.float(), dim=-1).detach().cpu().numpy().astype(np.float32)
        for (row_index, proposal_index), feature in zip(crop_keys, image_features):
            features[row_index, proposal_index] = feature
        crop_batch = []
        crop_keys = []

    for row_index, source_id in enumerate(source_ids):
        record = sample_records[source_id]
        image_path = _image_path_from_record(record, image_root=args.image_root, image_template=args.image_template)
        with image_module.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
            for proposal_index, is_valid in enumerate(region_mask[row_index]):
                if not bool(is_valid):
                    continue
                crop = _crop_normalized_box(image, proposal_boxes[row_index, proposal_index])
                if crop is None:
                    crop_failure_count += 1
                    continue
                crop_batch.append(crop)
                crop_keys.append((row_index, proposal_index))
                if len(crop_batch) >= int(args.crop_batch_size):
                    flush_batch()
    flush_batch()
    return features, crop_failure_count


def _best_ious(proposal_boxes: np.ndarray, region_mask: np.ndarray, bbox_targets: np.ndarray) -> np.ndarray:
    best = np.zeros(proposal_boxes.shape[:2], dtype=np.float32)
    for row_index in range(proposal_boxes.shape[0]):
        ious = _pairwise_iou(proposal_boxes[row_index], bbox_targets[row_index].reshape(1, 4)).reshape(-1)
        best[row_index] = np.where(region_mask[row_index], ious, -1.0)
    return best


def _region_positions(boxes: np.ndarray) -> np.ndarray:
    x1 = boxes[:, :, 0]
    y1 = boxes[:, :, 1]
    x2 = boxes[:, :, 2]
    y2 = boxes[:, :, 3]
    width = np.clip(x2 - x1, 0.0, 1.0)
    height = np.clip(y2 - y1, 0.0, 1.0)
    cx = x1 + 0.5 * width
    cy = y1 + 0.5 * height
    area = width * height
    return np.stack([x1, y1, x2, y2, cx, cy, width, height, area], axis=-1).astype(np.float32)


def _write_sample_records(
    path: Path,
    split: str,
    source_ids: list[str],
    records: dict[str, dict[str, Any]],
    boxes: np.ndarray,
    detector_scores: np.ndarray,
    target_indices: np.ndarray,
) -> None:
    rows = []
    for row_index, source_id in enumerate(source_ids):
        record = records[source_id]
        row = dict(record)
        row.update(
            {
                "split": split,
                "candidate_region_source": "groundingdino_topk_proposals",
                "candidate_region_boxes": [[float(value) for value in box] for box in boxes[row_index].tolist()],
                "candidate_region_detector_scores": [float(value) for value in detector_scores[row_index].tolist()],
                "target_region_index": int(target_indices[row_index]),
                "box_coordinate_convention": "xyxy_normalized",
            }
        )
        rows.append(row)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_alignment_pairs(path: Path, split: str, source_ids: list[str], records: dict[str, dict[str, Any]], target_indices: np.ndarray) -> None:
    rows = [
        {
            "source_id": source_id,
            "split": split,
            "phrase_span": records[source_id].get("phrase_span", [0, 1]),
            "target_region_index": int(target_indices[row_index]),
            "row_index": row_index,
        }
        for row_index, source_id in enumerate(source_ids)
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_target_histogram(path: Path, target_indices: np.ndarray, region_mask: np.ndarray) -> None:
    histogram: dict[str, dict[str, int]] = {}
    for target_index, mask_row in zip(target_indices.tolist(), region_mask):
        valid_k = str(int(np.count_nonzero(mask_row)))
        histogram.setdefault(valid_k, {})
        histogram[valid_k][str(int(target_index))] = histogram[valid_k].get(str(int(target_index)), 0) + 1
    path.write_text(json.dumps(histogram, indent=2, sort_keys=True) + "\n")


def _write_corruption(path: Path, split: str, source_ids: list[str]) -> None:
    path.write_text(json.dumps({"split": split, "source_ids": source_ids, "corruption": "none"}, sort_keys=True) + "\n")


def _write_common_files(args: argparse.Namespace, output_root: Path, split_source_ids: dict[str, list[str]]) -> None:
    data_card = default_data_card(args.output_dataset_name, args.output_version, ["text", "region"], ["phrase_region_grounding"])
    data_card["candidate_protocol"] = {
        "name": "groundingdino_topk_proposal_reranking_v0.1",
        "candidate_region_source": "groundingdino_topk_proposals",
        "fixed_k": int(args.max_proposals_per_sample),
        "proposal_model": "GroundingDINO",
        "proposal_predictions_template": str(args.predictions_template),
    }
    (output_root / "data_card.json").write_text(json.dumps(data_card, sort_keys=True) + "\n")
    (output_root / "splits.json").write_text(json.dumps(split_source_ids, sort_keys=True) + "\n")
    (output_root / "samples.parquet").write_text(
        "\n".join(
            json.dumps({"source_id": source_id, "split": split}, sort_keys=True)
            for split, source_ids in sorted(split_source_ids.items())
            for source_id in source_ids
        )
        + "\n"
    )
    (output_root / "provenance" / "pseudo_label_versions.json").write_text(
        json.dumps({"generated_from_splits": [], "version": "none"}, sort_keys=True) + "\n"
    )
    text_version = f"clip_text:{args.clip_model}:{args.revision}"
    region_version = f"clip_crop_groundingdino_proposals:{args.clip_model}:{args.revision}"
    reference = {"text": text_version, "region": region_version}
    baselines = {"ovha_full": dict(reference)}
    for baseline_name in baseline_names_for_task("phrase_region_grounding"):
        baselines[str(baseline_name)] = dict(reference)
    (output_root / "provenance" / "feature_versions.json").write_text(
        json.dumps(
            {
                "text": text_version,
                "region": region_version,
                "baselines": baselines,
                "proposal_source": str(args.predictions_template),
            },
            sort_keys=True,
        )
        + "\n"
    )


def _write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _write_checksums(root: Path) -> None:
    checksums = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
