#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RefCOCO candidate order randomization and target-slot audit contract.")
    parser.add_argument("records", type=Path, help="RefCOCO phrase-region records JSON produced by build_refcoco_stage_records.py")
    parser.add_argument("--max-sorted-fraction", type=float, default=0.95)
    parser.add_argument("--fail-on-sorted", action="store_true", default=True)
    args = parser.parse_args()
    try:
        payload = validate_refcoco_candidate_order(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {"ok": False, "records": str(args.records), "errors": [str(exc)], "warnings": []}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


def validate_refcoco_candidate_order(args: argparse.Namespace) -> dict[str, Any]:
    records = _read_records(Path(args.records))
    errors: list[str] = []
    warnings: list[str] = []
    sorted_count = 0
    missing_seed_count = 0
    histogram: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(records):
        source_id = record.get("source_id", f"records[{index}]")
        ann_ids = record.get("candidate_region_annotation_ids")
        target_index = record.get("target_region_index")
        seed = record.get("candidate_permutation_seed")
        if not isinstance(ann_ids, list) or not ann_ids:
            errors.append(f"{source_id}: missing candidate_region_annotation_ids")
            continue
        if any(not isinstance(value, int) or isinstance(value, bool) for value in ann_ids):
            errors.append(f"{source_id}: candidate_region_annotation_ids must be integers")
            continue
        if ann_ids == sorted(ann_ids):
            sorted_count += 1
        if not isinstance(target_index, int) or isinstance(target_index, bool) or target_index < 0 or target_index >= len(ann_ids):
            errors.append(f"{source_id}: target_region_index must point inside candidate_region_annotation_ids")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            missing_seed_count += 1
            errors.append(f"{source_id}: missing candidate_permutation_seed")
        bucket = histogram.setdefault(
            str(len(ann_ids)),
            {"sample_count": 0, "target_slot_counts": [0 for _ in ann_ids]},
        )
        bucket["sample_count"] += 1
        if isinstance(target_index, int) and not isinstance(target_index, bool) and 0 <= target_index < len(ann_ids):
            bucket["target_slot_counts"][target_index] += 1

    sorted_fraction = sorted_count / len(records)
    if bool(getattr(args, "fail_on_sorted", True)) and sorted_fraction >= float(args.max_sorted_fraction):
        errors.append(
            "sorted candidate_region_annotation_ids fraction exceeds gate: "
            f"{sorted_fraction:.6f} >= {float(args.max_sorted_fraction):.6f}"
        )
    for bucket in histogram.values():
        sample_count = int(bucket["sample_count"])
        slot_count = len(bucket["target_slot_counts"])
        expected = sample_count / max(1, slot_count)
        bucket["expected_per_slot"] = expected
        bucket["max_deviation"] = max((abs(int(count) - expected) for count in bucket["target_slot_counts"]), default=0.0)
        bucket["max_fraction"] = max((int(count) / sample_count for count in bucket["target_slot_counts"]), default=0.0) if sample_count else 0.0
        if sample_count >= slot_count * 20 and bucket["max_fraction"] > 0.45:
            warnings.append(f"valid_count={slot_count} target slot max fraction is high: {bucket['max_fraction']:.6f}")

    return {
        "ok": not errors,
        "records": str(args.records),
        "sample_count": len(records),
        "sorted_count": sorted_count,
        "sorted_fraction": sorted_fraction,
        "missing_seed_count": missing_seed_count,
        "target_slot_histogram_by_valid_count": {
            "audit_name": "target_slot_histogram_by_valid_count",
            "sample_count": len(records),
            "by_valid_count": histogram,
        },
        "errors": errors,
        "warnings": warnings,
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be an object")
    return records


if __name__ == "__main__":
    raise SystemExit(main())
