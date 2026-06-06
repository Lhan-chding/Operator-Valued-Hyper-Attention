#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SPLIT_ORDER = ("train", "val", "testA", "testB", "test")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage extracted RefCOCO/Flickr-style phrase-region frozen features "
            "into the raw manifest layout consumed by scripts/multimodal/build_cache.py."
        )
    )
    parser.add_argument("dataset_name", choices=("refcoco", "flickr30k_entities"))
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--region-features", type=Path, required=True)
    parser.add_argument("--text-mask", type=Path)
    parser.add_argument("--region-mask", type=Path)
    parser.add_argument("--failed-samples", type=Path)
    parser.add_argument("--license-tag", required=True)
    parser.add_argument("--preprocessing-version", required=True)
    args = parser.parse_args()

    try:
        payload = stage_refcoco_raw(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "dataset_name": args.dataset_name,
                    "raw_root": str(args.raw_root),
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


def stage_refcoco_raw(args: argparse.Namespace) -> dict[str, Any]:
    splits = _read_splits(args.splits)
    source_ids = _ordered_source_ids(splits)
    sample_count = len(source_ids)
    raw_root = args.raw_root
    for folder in ("annotations", "features", "provenance"):
        (raw_root / folder).mkdir(parents=True, exist_ok=True)

    text_shape = _copy_npy(args.text_features, raw_root / "features" / "text_features.npy", sample_count)
    region_shape = _copy_npy(args.region_features, raw_root / "features" / "region_features.npy", sample_count)
    text_mask_shape = _copy_optional_npy(args.text_mask, raw_root / "features" / "text_mask.npy", sample_count)
    region_mask_shape = _copy_optional_npy(args.region_mask, raw_root / "features" / "region_mask.npy", sample_count)
    records = _records(
        args.records,
        splits,
        license_tag=args.license_tag,
        preprocessing_version=args.preprocessing_version,
    )
    annotation_name = "refs.json" if args.dataset_name == "refcoco" else "phrase_regions.json"
    (raw_root / "annotations" / annotation_name).write_text(json.dumps({"records": records}, sort_keys=True) + "\n")
    if args.dataset_name == "refcoco":
        (raw_root / "annotations" / "instances.json").write_text(
            json.dumps({"records": [{"source_id": record["source_id"], "image_id": record["image_id"]} for record in records]}, sort_keys=True)
            + "\n"
        )
    else:
        (raw_root / "annotations" / "captions.json").write_text(
            json.dumps(
                {"records": [{"source_id": record["source_id"], "caption_id": record["caption_id"]} for record in records]},
                sort_keys=True,
            )
            + "\n"
        )
    if args.failed_samples is not None:
        _copy_text(args.failed_samples, raw_root / "provenance" / "failed_samples.jsonl")
    (raw_root / "splits.json").write_text(json.dumps(splits, sort_keys=True) + "\n")
    return {
        "ok": True,
        "dataset_name": args.dataset_name,
        "raw_root": str(raw_root),
        "sample_count": sample_count,
        "feature_shapes": {"text": text_shape, "region": region_shape},
        "mask_shapes": {"text": text_mask_shape, "region": region_mask_shape},
        "annotation_records": len(records),
        "next": (
            f"python scripts/multimodal/build_cache.py {args.dataset_name} "
            f"{raw_root} data/multimodal_cache --version v0.1"
        ),
    }


def _read_splits(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError("splits must be a non-empty JSON object keyed by split")
    splits: dict[str, list[str]] = {}
    for split, raw_source_ids in payload.items():
        if not isinstance(split, str) or not split.strip() or split != split.strip():
            raise ValueError("split names must be non-empty normalized strings")
        if not isinstance(raw_source_ids, list) or not raw_source_ids:
            raise ValueError(f"split {split} must contain a non-empty source_id list")
        source_ids = []
        for source_id in raw_source_ids:
            if not isinstance(source_id, str) or not source_id.strip() or source_id != source_id.strip():
                raise ValueError(f"split {split} contains an invalid source_id")
            source_ids.append(source_id)
        splits[split] = source_ids
    return splits


def _ordered_source_ids(splits: dict[str, list[str]]) -> list[str]:
    ordered_splits = [split for split in SPLIT_ORDER if split in splits]
    ordered_splits.extend(split for split in splits if split not in set(SPLIT_ORDER))
    source_ids = [source_id for split in ordered_splits for source_id in splits[split]]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source_id values must be unique across splits")
    return source_ids


def _copy_npy(source: Path, destination: Path, sample_count: int) -> tuple[int, ...]:
    array = np.load(source, allow_pickle=False)
    if array.ndim == 0 or int(array.shape[0]) != sample_count:
        raise ValueError(f"{source} first axis must match split sample count {sample_count}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, array)
    return tuple(int(dim) for dim in array.shape)


def _copy_optional_npy(source: Path | None, destination: Path, sample_count: int) -> tuple[int, ...] | None:
    if source is None:
        return None
    return _copy_npy(source, destination, sample_count)


def _records(
    path: Path,
    splits: dict[str, list[str]],
    *,
    license_tag: str,
    preprocessing_version: str,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    raw_records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("records must contain a non-empty list")
    by_source_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be an object")
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip() or source_id != source_id.strip():
            raise ValueError(f"records[{index}] source_id must be a non-empty normalized string")
        if source_id in by_source_id:
            raise ValueError(f"duplicate record source_id: {source_id}")
        by_source_id[source_id] = record
    staged = []
    for split in [name for name in SPLIT_ORDER if name in splits] + [name for name in splits if name not in set(SPLIT_ORDER)]:
        for source_id in splits[split]:
            if source_id not in by_source_id:
                raise ValueError(f"records missing source_id from splits: {source_id}")
            staged.append(_stage_record(by_source_id[source_id], split, license_tag, preprocessing_version))
    return staged


def _stage_record(record: dict[str, Any], split: str, license_tag: str, preprocessing_version: str) -> dict[str, Any]:
    source_id = _required_str(record, "source_id")
    staged = {
        "source_id": source_id,
        "split": split,
        "original_split": str(record.get("original_split") or split),
        "raw_ref": str(record.get("raw_ref") or f"refcoco://{source_id}"),
        "license_tag": license_tag,
        "preprocessing_version": preprocessing_version,
        "image_id": _required_str(record, "image_id"),
        "caption_id": _required_str(record, "caption_id"),
        "phrase_span": _phrase_span(record.get("phrase_span")),
        "region_box": _region_box(record.get("region_box")),
        "target_region_index": _target_region_index(record.get("target_region_index", 0)),
        "candidate_region_source": str(record.get("candidate_region_source") or "region_features"),
        "box_coordinate_convention": str(record.get("box_coordinate_convention") or "xyxy_normalized"),
    }
    candidate_boxes = _candidate_region_boxes(record.get("candidate_region_boxes"))
    if candidate_boxes:
        staged["candidate_region_boxes"] = candidate_boxes
    candidate_ann_ids = _candidate_region_annotation_ids(record.get("candidate_region_annotation_ids"))
    if candidate_ann_ids:
        staged["candidate_region_annotation_ids"] = candidate_ann_ids
    candidate_seed = _candidate_permutation_seed(record.get("candidate_permutation_seed"))
    if candidate_seed is not None:
        staged["candidate_permutation_seed"] = candidate_seed
    return staged


def _required_str(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"record {record.get('source_id', '?')} {key} must be a non-empty normalized string")
    return value


def _phrase_span(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("phrase_span must be an object with start/end")
    start = value.get("start")
    end = value.get("end")
    if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or not (0 <= start < end):
        raise ValueError("phrase_span must contain integer start/end with 0 <= start < end")
    return {"start": start, "end": end}


def _region_box(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("region_box must contain four numeric coordinates")
    try:
        return [float(coordinate) for coordinate in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("region_box must contain four numeric coordinates") from exc


def _candidate_region_boxes(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("candidate_region_boxes must be a list when provided")
    return [_region_box(box) for box in value]


def _candidate_region_annotation_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("candidate_region_annotation_ids must be a list when provided")
    ids: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError("candidate_region_annotation_ids must contain non-negative integers")
        ids.append(item)
    return ids


def _candidate_permutation_seed(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("candidate_permutation_seed must be a non-negative integer when provided")
    return value


def _target_region_index(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("target_region_index must be a non-negative integer")
    return value


def _copy_text(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text())


if __name__ == "__main__":
    raise SystemExit(main())
