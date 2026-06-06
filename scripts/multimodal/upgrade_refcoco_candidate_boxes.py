#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade an existing RefCOCO cache with candidate region boxes from staged raw refs."
    )
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--dataset-name", default="refcoco")
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    try:
        payload = upgrade_refcoco_candidate_boxes(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": "upgrade_refcoco_candidate_boxes",
                    "raw_root": str(args.raw_root),
                    "cache_root": str(MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version).root),
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


def upgrade_refcoco_candidate_boxes(args: argparse.Namespace) -> dict[str, Any]:
    layout = MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version)
    root = layout.root
    if not root.exists():
        raise ValueError(f"cache root does not exist: {root}")
    records_by_source_id = _records_by_source_id(args.raw_root / "annotations" / "refs.json")
    updated: dict[str, Any] = {}
    changed_paths: list[Path] = []

    for split in tuple(args.splits):
        source_ids = _read_source_ids(root / "provenance" / f"source_ids_{split}.txt")
        records = _records_for_source_ids(records_by_source_id, source_ids, split)
        candidate_count = _candidate_count(root, split, records)
        candidate_boxes = _candidate_region_boxes(records, candidate_count)
        region_pos = _region_positions(candidate_boxes)
        valid_box_mask = _valid_box_mask(candidate_boxes)

        boxes_path = root / "supervision" / f"candidate_region_boxes_{split}.npy"
        pos_path = root / "positions" / f"region_pos_{split}.npy"
        mask_path = root / "masks" / f"region_mask_{split}.npy"
        existing_mask = _optional_array(mask_path)
        region_mask = valid_box_mask if existing_mask is None else existing_mask.astype(bool) & valid_box_mask

        _write_array(boxes_path, candidate_boxes)
        _write_array(pos_path, region_pos)
        _write_array(mask_path, region_mask)
        changed_paths.extend((boxes_path, pos_path, mask_path))
        updated[split] = {
            "sample_count": int(candidate_boxes.shape[0]),
            "candidate_region_count": int(candidate_boxes.shape[1]),
            "candidate_region_boxes": str(boxes_path),
            "region_pos": str(pos_path),
            "region_mask": str(mask_path),
            "valid_box_fraction": float(region_mask.mean()) if region_mask.size else 0.0,
        }

    _update_checksums(root, changed_paths)
    return {
        "ok": True,
        "mode": "upgrade_refcoco_candidate_boxes",
        "policy": "existing cache feature tensors are reused; candidate boxes and region geometry are upgraded from raw refs",
        "cache_root": str(root),
        "splits": updated,
    }


def _records_by_source_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError(f"raw refs must contain non-empty records: {path}")
    by_source_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"raw refs records[{index}] must be an object")
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"raw refs records[{index}] missing source_id")
        by_source_id[source_id] = record
    return by_source_id


def _read_source_ids(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"cache source id file does not exist: {path}")
    source_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not source_ids:
        raise ValueError(f"cache source id file is empty: {path}")
    return source_ids


def _records_for_source_ids(records_by_source_id: dict[str, dict[str, Any]], source_ids: list[str], split: str) -> list[dict[str, Any]]:
    missing = [source_id for source_id in source_ids if source_id not in records_by_source_id]
    if missing:
        raise ValueError(f"raw refs missing {len(missing)} {split} source_ids; first missing: {missing[0]}")
    return [records_by_source_id[source_id] for source_id in source_ids]


def _candidate_count(root: Path, split: str, records: list[dict[str, Any]]) -> int:
    region_path = root / "token_fields" / f"region_{split}.npy"
    if region_path.exists():
        region = np.load(region_path, mmap_mode="r")
        if region.ndim < 2:
            raise ValueError(f"region feature shard must have a candidate axis: {region_path}")
        return int(region.shape[1])
    return max(len(record.get("candidate_region_boxes", [])) for record in records)


def _candidate_region_boxes(records: list[dict[str, Any]], candidate_count: int) -> np.ndarray:
    boxes = np.zeros((len(records), candidate_count, 4), dtype=np.float32)
    for row, record in enumerate(records):
        raw_boxes = record.get("candidate_region_boxes")
        if not isinstance(raw_boxes, list) or not raw_boxes:
            raise ValueError(f"record missing candidate_region_boxes: {record.get('source_id')}")
        for col, raw_box in enumerate(raw_boxes[:candidate_count]):
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                raise ValueError(f"candidate_region_boxes must contain four coordinates: {record.get('source_id')}")
            boxes[row, col] = np.asarray([float(value) for value in raw_box], dtype=np.float32)
    return boxes


def _region_positions(candidate_boxes: np.ndarray) -> np.ndarray:
    x1y1 = candidate_boxes[..., :2]
    x2y2 = candidate_boxes[..., 2:]
    width_height = np.clip(x2y2 - x1y1, 0.0, None)
    centers = 0.5 * (x1y1 + x2y2)
    area = (width_height[..., 0] * width_height[..., 1])[..., None]
    return np.concatenate([candidate_boxes, centers, width_height, area], axis=-1).astype(np.float32)


def _valid_box_mask(candidate_boxes: np.ndarray) -> np.ndarray:
    width_height = np.clip(candidate_boxes[..., 2:] - candidate_boxes[..., :2], 0.0, None)
    return (width_height[..., 0] > 0.0) & (width_height[..., 1] > 0.0)


def _optional_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def _write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _update_checksums(root: Path, changed_paths: list[Path]) -> None:
    checksums_path = root / "checksums.json"
    checksums: dict[str, str] = {}
    if checksums_path.exists():
        payload = json.loads(checksums_path.read_text())
        if isinstance(payload, dict):
            checksums = {str(key): str(value) for key, value in payload.items()}
    for path in sorted(set(changed_paths)):
        checksums[str(path.relative_to(root))] = file_sha256(path)
    checksums_path.write_text(json.dumps(checksums, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
