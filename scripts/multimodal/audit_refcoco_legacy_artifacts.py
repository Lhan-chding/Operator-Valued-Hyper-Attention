#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit RefCOCO staged raw/cache/output artifacts that can leak legacy sorted "
            "candidate records into a new balanced grounding experiment. This command is "
            "read-only and prints explicit deletion/rebuild commands."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_multimodal/refcoco"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--dataset-name", default="refcoco")
    parser.add_argument("--version", default="v0.1")
    parser.add_argument(
        "--stage-dir",
        action="append",
        type=Path,
        default=[
            Path("data/raw_multimodal/_downloads/refcoco_stage_inputs"),
            Path("data/raw_multimodal/_downloads/refcoco_balanced_rebuild"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        action="append",
        type=Path,
        default=[
            Path("outputs/multimodal/refcoco_main"),
            Path("outputs/multimodal/refcoco_public_smoke"),
        ],
    )
    args = parser.parse_args()
    try:
        payload = audit_refcoco_legacy_artifacts(args)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        payload = {"ok": False, "mode": "audit_refcoco_legacy_artifacts", "errors": [str(exc)], "warnings": []}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


def audit_refcoco_legacy_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = Path(args.raw_root)
    cache_root = Path(args.cache_root)
    dataset_name = str(args.dataset_name)
    version = str(args.version)
    raw_records = raw_root / "annotations" / "refs.json"
    cache_version_root = cache_root / dataset_name / version
    candidates = [
        raw_root / "annotations",
        raw_root / "features",
        raw_root / "provenance",
        raw_root / "splits.json",
        cache_version_root,
        *[Path(path) for path in args.stage_dir],
        *[Path(path) for path in args.output_dir],
    ]
    existing = [path for path in candidates if path.exists()]
    record_audit = _record_audit(raw_records) if raw_records.exists() else None
    stale_reasons = []
    if record_audit is None:
        stale_reasons.append("raw RefCOCO refs.json is absent; rebuild is required before formal training")
    else:
        if int(record_audit["missing_candidate_permutation_seed"]) > 0:
            stale_reasons.append("raw refs.json contains records missing candidate_permutation_seed")
        if float(record_audit["sorted_fraction"]) >= 0.95:
            stale_reasons.append("raw refs.json candidate_region_annotation_ids are globally sorted")
        if not set(record_audit["splits"]).issuperset({"train", "val", "testA", "testB"}):
            stale_reasons.append("raw refs.json/splits do not expose train/val/testA/testB")
    if cache_version_root.exists():
        stale_reasons.append(f"cache version exists and should be rebuilt under the new protocol: {cache_version_root}")
    delete_targets = [path for path in existing if _is_deletable_project_artifact(path, raw_root, cache_root, dataset_name, version)]
    return {
        "ok": True,
        "mode": "audit_refcoco_legacy_artifacts",
        "policy": "read_only_audit; delete only generated staging/cache/output artifacts, never downloaded archives or extracted COCO images",
        "raw_root": str(raw_root),
        "cache_version_root": str(cache_version_root),
        "record_audit": record_audit,
        "existing_artifacts": [str(path) for path in existing],
        "stale_reasons": stale_reasons,
        "delete_targets": [str(path) for path in delete_targets],
        "delete_command": _rm_command(delete_targets),
        "rebuild_command": _rebuild_command(dataset_name, version),
        "warnings": [],
        "errors": [],
    }


def _record_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} records must be a list or object with records list")
    sorted_count = 0
    missing_seed = 0
    splits = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        ann_ids = record.get("candidate_region_annotation_ids")
        if isinstance(ann_ids, list) and ann_ids == sorted(ann_ids):
            sorted_count += 1
        if not isinstance(record.get("candidate_permutation_seed"), int):
            missing_seed += 1
        split = record.get("split") or record.get("original_split")
        if isinstance(split, str):
            splits.add(split)
    sample_count = len(records)
    return {
        "path": str(path),
        "sample_count": sample_count,
        "sorted_count": sorted_count,
        "sorted_fraction": sorted_count / max(1, sample_count),
        "missing_candidate_permutation_seed": missing_seed,
        "splits": sorted(splits),
    }


def _is_deletable_project_artifact(path: Path, raw_root: Path, cache_root: Path, dataset_name: str, version: str) -> bool:
    allowed = {
        raw_root / "annotations",
        raw_root / "features",
        raw_root / "provenance",
        raw_root / "splits.json",
        cache_root / dataset_name / version,
        Path("data/raw_multimodal/_downloads/refcoco_stage_inputs"),
        Path("data/raw_multimodal/_downloads/refcoco_balanced_rebuild"),
        Path("outputs/multimodal/refcoco_main"),
        Path("outputs/multimodal/refcoco_public_smoke"),
    }
    return path in allowed


def _rm_command(paths: list[Path]) -> str:
    if not paths:
        return "true  # no generated RefCOCO artifacts found to delete"
    joined = " \\\n       ".join(str(path) for path in paths)
    return f"rm -rf {joined}"


def _rebuild_command(dataset_name: str, version: str) -> str:
    return (
        "python scripts/multimodal/rebuild_refcoco_balanced_cache.py \\\n"
        f"  {dataset_name} \\\n"
        "  data/raw_multimodal/_downloads/refcoco_balanced_rebuild \\\n"
        "  --refs data/raw_multimodal/_downloads/refcoco/extracted/replace_with_refcoco_refs.json \\\n"
        "  --instances data/raw_multimodal/_downloads/refcoco/extracted/annotations/instances_train2014.json \\\n"
        "  --raw-root data/raw_multimodal/refcoco \\\n"
        "  --cache-root data/multimodal_cache \\\n"
        f"  --version {version} \\\n"
        "  --text-features data/raw_multimodal/_downloads/refcoco_stage_inputs/refcoco_text_features.npy \\\n"
        "  --region-features data/raw_multimodal/_downloads/refcoco_stage_inputs/refcoco_region_features.npy \\\n"
        "  --feature-version refcoco-frozen-features-v0.1 \\\n"
        "  --candidate-count-policy variable_k \\\n"
        "  --purge-existing-raw-and-cache"
    )


if __name__ == "__main__":
    raise SystemExit(main())
