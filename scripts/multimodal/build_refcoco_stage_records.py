#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
from typing import Any


SPLIT_ORDER = ("train", "val", "testA", "testB", "test")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build RefCOCO stage records/splits from UNC referring-expression annotations "
            "and COCO 2014 instances metadata."
        )
    )
    parser.add_argument("dataset_name", choices=("refcoco", "refcoco_plus", "refcocog"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--candidate-region-source", default="coco_gt_box")
    parser.add_argument("--box-coordinate-convention", default="xyxy_normalized")
    parser.add_argument("--max-candidate-regions", type=int, default=32)
    parser.add_argument("--candidate-count-policy", choices=("variable_k", "fixed_k"), default="variable_k")
    args = parser.parse_args()

    try:
        payload = build_refcoco_stage_records(args)
    except (OSError, ValueError, json.JSONDecodeError, pickle.UnpicklingError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "dataset_name": args.dataset_name,
                    "output_dir": str(args.output_dir),
                    "errors": [str(exc)],
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_refcoco_stage_records(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_candidate_regions <= 1:
        raise ValueError("--max-candidate-regions must be greater than 1 for non-trivial grounding")
    instances = _read_instances(args.instances)
    refs = _read_refs(args.refs)
    records: list[dict[str, Any]] = []
    splits: dict[str, list[str]] = {split: [] for split in SPLIT_ORDER}

    for ref in refs:
        base = _ref_base(ref, instances)
        split = _normalize_split(_required_str(ref, "split"))
        for sentence in _sentences(ref):
            record = _record_for_sentence(
                args.dataset_name,
                base,
                sentence,
                split,
                candidate_region_source=args.candidate_region_source,
                box_coordinate_convention=args.box_coordinate_convention,
                max_candidate_regions=args.max_candidate_regions,
                candidate_count_policy=args.candidate_count_policy,
            )
            records.append(record)
            splits[split].append(record["source_id"])

    records_by_ordered_splits = _records_by_split_order(records, splits)
    splits = {split: source_ids for split, source_ids in splits.items() if source_ids}
    _validate_splits(splits)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{args.dataset_name}_phrase_region_records.json"
    splits_path = output_dir / f"{args.dataset_name}_splits.json"
    manifest_path = output_dir / f"{args.dataset_name}_stage_records_manifest.json"
    histogram_path = output_dir / f"{args.dataset_name}_target_slot_histogram_by_valid_count.json"
    records_path.write_text(json.dumps({"records": records_by_ordered_splits}, sort_keys=True) + "\n")
    splits_path.write_text(json.dumps(splits, sort_keys=True) + "\n")
    histogram = _target_slot_histogram_by_valid_count(records_by_ordered_splits)
    histogram_path.write_text(json.dumps(histogram, indent=2, sort_keys=True) + "\n")
    manifest = {
        "dataset_name": args.dataset_name,
        "source_paths": {"refs": str(args.refs), "instances": str(args.instances)},
        "records": str(records_path),
        "splits": str(splits_path),
        "target_slot_histogram_by_valid_count": str(histogram_path),
        "sample_count": len(records_by_ordered_splits),
        "split_counts": {split: len(source_ids) for split, source_ids in sorted(splits.items())},
        "candidate_region_source": args.candidate_region_source,
        "max_candidate_regions": args.max_candidate_regions,
        "candidate_count_policy": args.candidate_count_policy,
        "box_coordinate_convention": args.box_coordinate_convention,
        "candidate_permutation_policy": "stable_source_id_seeded_target_slot_randomization",
        "candidate_protocol": (
            "fixed_k_balanced_v0.2"
            if args.candidate_count_policy == "fixed_k"
            else "variable_k_balanced_stratified_v0.2"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "ok": True,
        **manifest,
        "manifest": str(manifest_path),
        "next": (
            f"python scripts/multimodal/stage_refcoco_raw.py refcoco data/raw_multimodal/refcoco "
            f"--splits {splits_path} --records {records_path} "
            f"--text-features {output_dir / 'refcoco_text_features.npy'} "
            f"--region-features {output_dir / 'refcoco_region_features.npy'} "
            f"--license-tag refcoco-coco2014 --preprocessing-version refcoco-frozen-features-v0.1"
        ),
    }


def _read_instances(path: Path) -> dict[str, dict[int, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("instances must be a COCO-style JSON object")
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("instances JSON must contain images and annotations lists")
    image_by_id = {}
    for image in images:
        if not isinstance(image, dict):
            continue
        image_id = _int_value(image.get("id"), field="image id")
        width = _positive_float(image.get("width"), field=f"image {image_id} width")
        height = _positive_float(image.get("height"), field=f"image {image_id} height")
        image_by_id[image_id] = {"id": image_id, "width": width, "height": height}
    annotation_by_id = {}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        ann_id = _int_value(annotation.get("id"), field="annotation id")
        image_id = _int_value(annotation.get("image_id"), field=f"annotation {ann_id} image_id")
        if image_id not in image_by_id:
            raise ValueError(f"annotation {ann_id} references missing image_id {image_id}")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"annotation {ann_id} bbox must be COCO [x,y,w,h]")
        normalized = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": annotation.get("category_id"),
            "bbox": [float(value) for value in bbox],
        }
        annotation_by_id[ann_id] = normalized
        annotations_by_image.setdefault(image_id, []).append(normalized)
    for image_annotations in annotations_by_image.values():
        image_annotations.sort(key=lambda item: int(item["id"]))
    return {"images": image_by_id, "annotations": annotation_by_id, "annotations_by_image": annotations_by_image}


def _read_refs(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".p", ".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        payload = json.loads(path.read_text())
    refs = payload.get("refs") if isinstance(payload, dict) else payload
    if not isinstance(refs, list) or not refs:
        raise ValueError("refs must contain a non-empty list")
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict):
            raise ValueError(f"refs[{index}] must be an object")
    return refs


def _ref_base(ref: dict[str, Any], instances: dict[str, dict[int, Any]]) -> dict[str, Any]:
    ann_id = _int_value(ref.get("ann_id"), field="ref ann_id")
    annotations = instances["annotations"]
    if ann_id not in annotations:
        raise ValueError(f"ref ann_id {ann_id} missing from COCO instances")
    annotation = annotations[ann_id]
    image_id = _int_value(ref.get("image_id", annotation["image_id"]), field=f"ref ann_id {ann_id} image_id")
    if image_id != annotation["image_id"]:
        raise ValueError(f"ref ann_id {ann_id} image_id disagrees with COCO annotation")
    image = instances["images"][image_id]
    candidate_annotations = instances["annotations_by_image"].get(image_id, [])
    if not candidate_annotations:
        raise ValueError(f"image_id {image_id} has no COCO candidate annotations")
    return {
        "ref_id": _int_value(ref.get("ref_id", ann_id), field="ref_id"),
        "ann_id": ann_id,
        "image_id": image_id,
        "bbox": _normalize_coco_bbox(annotation["bbox"], image["width"], image["height"]),
        "candidate_annotations": candidate_annotations,
        "image_width": image["width"],
        "image_height": image["height"],
    }


def _sentences(ref: dict[str, Any]) -> list[dict[str, Any]]:
    sentences = ref.get("sentences")
    if isinstance(sentences, list) and sentences:
        return [sentence for sentence in sentences if isinstance(sentence, dict)]
    sent_ids = ref.get("sent_ids")
    if isinstance(sent_ids, list) and sent_ids:
        return [{"sent_id": sent_id, "raw": str(sent_id), "tokens": [str(sent_id)]} for sent_id in sent_ids]
    raise ValueError(f"ref {ref.get('ref_id', ref.get('ann_id', '?'))} missing sentences")


def _record_for_sentence(
    dataset_name: str,
    base: dict[str, Any],
    sentence: dict[str, Any],
    split: str,
    *,
    candidate_region_source: str,
    box_coordinate_convention: str,
    max_candidate_regions: int,
    candidate_count_policy: str = "variable_k",
) -> dict[str, Any]:
    sent_id = _int_value(sentence.get("sent_id"), field="sentence sent_id")
    tokens = sentence.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raw = sentence.get("raw") or sentence.get("sent") or ""
        tokens = str(raw).split()
    token_count = len(tokens)
    if token_count <= 0:
        raise ValueError(f"sentence {sent_id} has no tokens")
    source_id = f"{dataset_name}::image{base['image_id']}::ann{base['ann_id']}::sent{sent_id}"
    candidate_boxes, candidate_ann_ids, target_region_index, candidate_permutation_seed = _candidate_regions_balanced(
        base,
        max_candidate_regions=max_candidate_regions,
        source_id=source_id,
        candidate_count_policy=candidate_count_policy,
    )
    return {
        "source_id": source_id,
        "image_id": f"image{base['image_id']}",
        "caption_id": f"sent{sent_id}",
        "phrase_span": {"start": 0, "end": token_count},
        "region_box": base["bbox"],
        "candidate_region_boxes": candidate_boxes,
        "candidate_region_annotation_ids": candidate_ann_ids,
        "target_region_index": target_region_index,
        "candidate_permutation_seed": candidate_permutation_seed,
        "candidate_region_source": candidate_region_source,
        "candidate_count_policy": candidate_count_policy,
        "box_coordinate_convention": box_coordinate_convention,
        "original_split": split,
        "raw_ref": f"{dataset_name}://ref{base['ref_id']}/sent{sent_id}",
    }


def _candidate_regions_balanced(
    base: dict[str, Any],
    *,
    max_candidate_regions: int,
    source_id: str,
    hard_negative_policy: str = "same_category_then_spatial_then_random",
    candidate_count_policy: str = "variable_k",
) -> tuple[list[list[float]], list[int], int, int]:
    target_ann_id = int(base["ann_id"])
    width = float(base["image_width"])
    height = float(base["image_height"])
    annotations = list(base["candidate_annotations"])
    target = [annotation for annotation in annotations if int(annotation["id"]) == target_ann_id]
    if not target:
        raise ValueError(f"target annotation {target_ann_id} missing from image candidate annotations")
    target_annotation = target[0]
    distractors = [annotation for annotation in annotations if int(annotation["id"]) != target_ann_id]
    if candidate_count_policy not in {"variable_k", "fixed_k"}:
        raise ValueError(f"unsupported candidate_count_policy: {candidate_count_policy}")
    if candidate_count_policy == "fixed_k" and len(distractors) + 1 < max_candidate_regions:
        raise ValueError(
            "fixed_k candidate policy requires enough same-image COCO candidates; "
            f"target image has {len(distractors) + 1}, requested {max_candidate_regions}"
        )
    slot_count = max_candidate_regions if candidate_count_policy == "fixed_k" else min(max_candidate_regions, len(distractors) + 1)
    rng, seed = _stable_rng(source_id)
    distractors = _select_hard_distractors(
        target_annotation,
        distractors,
        k=slot_count - 1,
        rng=rng,
        policy=hard_negative_policy,
    )
    target_region_index = rng.randrange(slot_count)
    selected: list[dict[str, Any] | None] = [None] * slot_count
    selected[target_region_index] = target_annotation
    distractor_index = 0
    for slot in range(slot_count):
        if selected[slot] is None:
            selected[slot] = distractors[distractor_index]
            distractor_index += 1
    concrete_selected = [annotation for annotation in selected if annotation is not None]
    if len(concrete_selected) != slot_count:
        raise ValueError("balanced candidate construction produced an empty candidate slot")
    selected = concrete_selected
    candidate_ann_ids = [int(annotation["id"]) for annotation in selected]
    candidate_boxes = [_normalize_coco_bbox(annotation["bbox"], width, height) for annotation in selected]
    return candidate_boxes, candidate_ann_ids, target_region_index, seed


def _stable_rng(source_id: str) -> tuple[random.Random, int]:
    seed = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed), seed


def _select_hard_distractors(
    target: dict[str, Any],
    distractors: list[dict[str, Any]],
    *,
    k: int,
    rng: random.Random,
    policy: str,
) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    if policy != "same_category_then_spatial_then_random":
        shuffled = list(distractors)
        rng.shuffle(shuffled)
        return shuffled[:k]
    selected: list[dict[str, Any]] = []
    remaining = list(distractors)

    same_category = [
        annotation
        for annotation in remaining
        if annotation.get("category_id") is not None and annotation.get("category_id") == target.get("category_id")
    ]
    same_category.sort(key=lambda item: (_bbox_center_distance(target["bbox"], item["bbox"]), int(item["id"])))
    selected.extend(same_category[:k])
    selected_ids = {int(annotation["id"]) for annotation in selected}
    remaining = [annotation for annotation in remaining if int(annotation["id"]) not in selected_ids]

    if len(selected) < k:
        spatial = sorted(remaining, key=lambda item: (_bbox_center_distance(target["bbox"], item["bbox"]), int(item["id"])))
        selected.extend(spatial[: k - len(selected)])
        selected_ids = {int(annotation["id"]) for annotation in selected}
        remaining = [annotation for annotation in remaining if int(annotation["id"]) not in selected_ids]

    if len(selected) < k:
        rng.shuffle(remaining)
        selected.extend(remaining[: k - len(selected)])
    return selected[:k]


def _bbox_center_distance(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    lcx, lcy = lx + 0.5 * lw, ly + 0.5 * lh
    rcx, rcy = rx + 0.5 * rw, ry + 0.5 * rh
    return (lcx - rcx) ** 2 + (lcy - rcy) ** 2


def _normalize_coco_bbox(bbox: list[float], width: float, height: float) -> list[float]:
    x, y, box_width, box_height = bbox
    if box_width <= 0 or box_height <= 0:
        raise ValueError("COCO bbox width/height must be positive")
    x1 = x / width
    y1 = y / height
    x2 = (x + box_width) / width
    y2 = (y + box_height) / height
    return [round(value, 6) for value in (x1, y1, x2, y2)]


def _records_by_split_order(records: list[dict[str, Any]], splits: dict[str, list[str]]) -> list[dict[str, Any]]:
    by_source_id = {record["source_id"]: record for record in records}
    ordered = []
    for split in [name for name in SPLIT_ORDER if name in splits] + [name for name in splits if name not in set(SPLIT_ORDER)]:
        ordered.extend(by_source_id[source_id] for source_id in splits[split])
    return ordered


def _validate_splits(splits: dict[str, list[str]]) -> None:
    if not splits:
        raise ValueError("no splits were produced")
    seen = {}
    for split, source_ids in splits.items():
        if not source_ids:
            raise ValueError(f"split {split} is empty")
        for source_id in source_ids:
            previous = seen.get(source_id)
            if previous is not None:
                raise ValueError(f"source_id appears in multiple splits: {source_id} ({previous}, {split})")
            seen[source_id] = split


def _normalize_split(value: str) -> str:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered in {"train", "training"}:
        return "train"
    if lowered in {"val", "valid", "validation"}:
        return "val"
    if lowered in {"testa", "test_a"}:
        return "testA"
    if lowered in {"testb", "test_b"}:
        return "testB"
    if lowered == "test":
        return "test"
    raise ValueError(f"unsupported RefCOCO split: {value}")


def _target_slot_histogram_by_valid_count(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_valid_count: dict[str, dict[str, Any]] = {}
    for record in records:
        ann_ids = record.get("candidate_region_annotation_ids")
        target_index = record.get("target_region_index")
        if not isinstance(ann_ids, list) or not ann_ids:
            raise ValueError(f"record {record.get('source_id', '?')} missing candidate_region_annotation_ids")
        if not isinstance(target_index, int) or isinstance(target_index, bool) or target_index < 0 or target_index >= len(ann_ids):
            raise ValueError(f"record {record.get('source_id', '?')} has invalid target_region_index")
        key = str(len(ann_ids))
        row = by_valid_count.setdefault(
            key,
            {
                "sample_count": 0,
                "target_slot_counts": [0 for _ in ann_ids],
            },
        )
        if len(row["target_slot_counts"]) != len(ann_ids):
            raise ValueError(f"inconsistent valid count bucket for record {record.get('source_id', '?')}")
        row["sample_count"] += 1
        row["target_slot_counts"][target_index] += 1

    for row in by_valid_count.values():
        sample_count = int(row["sample_count"])
        slot_count = len(row["target_slot_counts"])
        expected = sample_count / max(1, slot_count)
        row["expected_per_slot"] = expected
        row["max_deviation"] = max((abs(int(count) - expected) for count in row["target_slot_counts"]), default=0.0)
        row["max_fraction"] = max((int(count) / sample_count for count in row["target_slot_counts"]), default=0.0) if sample_count else 0.0
    return {
        "audit_name": "target_slot_histogram_by_valid_count",
        "sample_count": len(records),
        "by_valid_count": by_valid_count,
    }


def _int_value(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _positive_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _required_str(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{key} must be a non-empty normalized string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
