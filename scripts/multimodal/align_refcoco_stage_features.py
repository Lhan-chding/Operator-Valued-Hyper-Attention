#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SPLIT_ORDER = ("train", "val", "testA", "testB", "test")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Align externally frozen RefCOCO text/region feature banks to the exact "
            "source_id order required by stage_refcoco_raw.py."
        )
    )
    parser.add_argument("dataset_name", choices=("refcoco", "refcoco_plus", "refcocog", "flickr30k_entities"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--text-source-ids", type=Path, required=True)
    parser.add_argument("--region-features", type=Path, required=True)
    parser.add_argument("--region-source-ids", type=Path, required=True)
    parser.add_argument("--feature-version", required=True)
    args = parser.parse_args()

    try:
        payload = align_refcoco_stage_features(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
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


def align_refcoco_stage_features(args: argparse.Namespace) -> dict[str, Any]:
    splits = _read_splits(args.splits)
    ordered_source_ids = _ordered_source_ids(splits)
    _validate_records_cover_source_ids(args.records, ordered_source_ids)
    text_bank = _load_feature_bank(args.text_features, args.text_source_ids, bank_name="text")
    region_bank = _load_feature_bank(args.region_features, args.region_source_ids, bank_name="region")

    text = _align_bank(text_bank, ordered_source_ids, bank_name="text")
    region = _align_bank(region_bank, ordered_source_ids, bank_name="region")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / f"{args.dataset_name}_text_features.npy"
    region_path = output_dir / f"{args.dataset_name}_region_features.npy"
    manifest_path = output_dir / f"{args.dataset_name}_feature_alignment_manifest.json"
    np.save(text_path, text)
    np.save(region_path, region)
    manifest = {
        "dataset_name": args.dataset_name,
        "feature_version": args.feature_version,
        "sample_count": len(ordered_source_ids),
        "ordered_source_ids": ordered_source_ids,
        "splits": str(args.splits),
        "records": str(args.records),
        "inputs": {
            "text_features": _artifact_descriptor(args.text_features),
            "text_source_ids": _artifact_descriptor(args.text_source_ids),
            "region_features": _artifact_descriptor(args.region_features),
            "region_source_ids": _artifact_descriptor(args.region_source_ids),
        },
        "outputs": {
            "text_features": str(text_path),
            "region_features": str(region_path),
        },
        "feature_shapes": {
            "text": list(text.shape),
            "region": list(region.shape),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "ok": True,
        "dataset_name": args.dataset_name,
        "output_dir": str(output_dir),
        "sample_count": len(ordered_source_ids),
        "feature_shapes": manifest["feature_shapes"],
        "manifest": str(manifest_path),
        "next": (
            f"python scripts/multimodal/stage_refcoco_raw.py {args.dataset_name} "
            f"data/raw_multimodal/{args.dataset_name} "
            f"--splits {args.splits} --records {args.records} "
            f"--text-features {text_path} --region-features {region_path} "
            f"--license-tag refcoco-coco2014 --preprocessing-version {args.feature_version}"
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


def _validate_records_cover_source_ids(records_path: Path, ordered_source_ids: list[str]) -> None:
    payload = json.loads(records_path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("records must contain a non-empty list")
    observed = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be an object")
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip() or source_id != source_id.strip():
            raise ValueError(f"records[{index}] source_id must be a normalized string")
        observed.append(source_id)
    missing = sorted(set(ordered_source_ids) - set(observed))
    extra = sorted(set(observed) - set(ordered_source_ids))
    if missing or extra:
        raise ValueError(f"records/source_id set mismatch; missing={missing[:10]} extra={extra[:10]}")


def _load_feature_bank(features_path: Path, source_ids_path: Path, *, bank_name: str) -> dict[str, np.ndarray]:
    features = np.load(features_path, allow_pickle=False)
    if features.ndim == 0:
        raise ValueError(f"{bank_name} features must have sample axis")
    source_ids = _read_source_ids(source_ids_path)
    if len(source_ids) != int(features.shape[0]):
        raise ValueError(
            f"{bank_name} feature row count {features.shape[0]} does not match source id count {len(source_ids)}"
        )
    if len(source_ids) != len(set(source_ids)):
        duplicates = sorted(source_id for source_id in set(source_ids) if source_ids.count(source_id) > 1)
        raise ValueError(f"{bank_name} source ids contain duplicates: {', '.join(duplicates)}")
    return {source_id: np.asarray(features[index]).copy() for index, source_id in enumerate(source_ids)}


def _read_source_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        values = payload.get("source_ids") if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise ValueError(f"{path} JSON must be a source_id list or contain source_ids")
        raw_source_ids = values
    else:
        raw_source_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    source_ids = []
    for value in raw_source_ids:
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{path} contains an invalid source_id")
        source_ids.append(value)
    if not source_ids:
        raise ValueError(f"{path} must contain at least one source_id")
    return source_ids


def _align_bank(bank: dict[str, np.ndarray], ordered_source_ids: list[str], *, bank_name: str) -> np.ndarray:
    missing = [source_id for source_id in ordered_source_ids if source_id not in bank]
    if missing:
        raise ValueError(f"{bank_name} feature bank missing source_id values: {', '.join(missing[:10])}")
    rows = [bank[source_id] for source_id in ordered_source_ids]
    shapes = {tuple(row.shape) for row in rows}
    if len(shapes) != 1:
        raise ValueError(f"{bank_name} feature rows have inconsistent shapes: {sorted(shapes)}")
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
