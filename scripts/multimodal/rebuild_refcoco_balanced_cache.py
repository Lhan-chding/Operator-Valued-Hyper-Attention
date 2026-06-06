#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.data.multimodal.adapters.base import RawDatasetManifest
from moat_ovha_torch.data.multimodal.adapters.refcoco import RefCOCOAdapter
from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
from scripts.multimodal.align_refcoco_stage_features import align_refcoco_stage_features
from scripts.multimodal.build_refcoco_stage_records import build_refcoco_stage_records
from scripts.multimodal.stage_refcoco_raw import stage_refcoco_raw
from scripts.multimodal.validate_refcoco_candidate_order import validate_refcoco_candidate_order


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild RefCOCO/COCO balanced candidate records, stage raw features, "
            "write cache artifacts, and validate target-slot/candidate-order gates."
        )
    )
    parser.add_argument("dataset_name", choices=("refcoco", "refcoco_plus", "refcocog"))
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--region-features", type=Path, required=True)
    parser.add_argument("--text-source-ids", type=Path)
    parser.add_argument("--region-source-ids", type=Path)
    parser.add_argument("--text-mask", type=Path)
    parser.add_argument("--region-mask", type=Path)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--license-tag", default="refcoco-coco2014")
    parser.add_argument("--max-candidate-regions", type=int, default=32)
    args = parser.parse_args()

    try:
        payload = rebuild_refcoco_balanced_cache(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "ok": False,
            "mode": "rebuild_refcoco_balanced_cache",
            "dataset_name": args.dataset_name,
            "errors": [str(exc)],
            "warnings": [],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


def rebuild_refcoco_balanced_cache(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = Path(args.work_dir)
    stage_dir = work_dir / "stage_records"
    feature_dir = work_dir / "aligned_features"
    stage_payload = build_refcoco_stage_records(
        argparse.Namespace(
            dataset_name=args.dataset_name,
            output_dir=stage_dir,
            refs=args.refs,
            instances=args.instances,
            candidate_region_source="coco_gt_box",
            box_coordinate_convention="xyxy_normalized",
            max_candidate_regions=int(args.max_candidate_regions),
        )
    )
    records_path = Path(stage_payload["records"])
    splits_path = Path(stage_payload["splits"])
    order_report = validate_refcoco_candidate_order(
        argparse.Namespace(records=records_path, fail_on_sorted=True, max_sorted_fraction=0.95)
    )
    if not order_report["ok"]:
        raise ValueError("candidate order validation failed: " + "; ".join(order_report["errors"]))

    text_features = Path(args.text_features)
    region_features = Path(args.region_features)
    alignment_payload: dict[str, Any] | None = None
    if args.text_source_ids is not None or args.region_source_ids is not None:
        if args.text_source_ids is None or args.region_source_ids is None:
            raise ValueError("--text-source-ids and --region-source-ids must be provided together")
        alignment_payload = align_refcoco_stage_features(
            argparse.Namespace(
                dataset_name=args.dataset_name,
                output_dir=feature_dir,
                splits=splits_path,
                records=records_path,
                text_features=text_features,
                text_source_ids=args.text_source_ids,
                region_features=region_features,
                region_source_ids=args.region_source_ids,
                feature_version=args.feature_version,
            )
        )
        text_features = Path(alignment_payload["outputs"]["text_features"])
        region_features = Path(alignment_payload["outputs"]["region_features"])

    stage_raw_payload = stage_refcoco_raw(
        argparse.Namespace(
            dataset_name="refcoco",
            raw_root=args.raw_root,
            splits=splits_path,
            records=records_path,
            text_features=text_features,
            region_features=region_features,
            text_mask=args.text_mask,
            region_mask=args.region_mask,
            failed_samples=None,
            license_tag=args.license_tag,
            preprocessing_version=args.feature_version,
        )
    )
    adapter = RefCOCOAdapter()
    manifest = RawDatasetManifest(
        dataset_name="refcoco",
        raw_root=args.raw_root,
        files={
            "annotations/instances.json": args.raw_root / "annotations" / "instances.json",
            "annotations/refs.json": args.raw_root / "annotations" / "refs.json",
            "features/text_features.npy": args.raw_root / "features" / "text_features.npy",
            "features/region_features.npy": args.raw_root / "features" / "region_features.npy",
            "splits.json": args.raw_root / "splits.json",
        },
    )
    splits = _read_splits(args.raw_root / "splits.json")
    for split in _ordered_splits(splits):
        adapter.write_cache(manifest, args.cache_root, split, args.version)

    layout = MultimodalCacheLayout(args.cache_root, "refcoco", args.version)
    cache_report = validate_cache_layout(layout, splits=tuple(_ordered_splits(splits)))
    return {
        "ok": cache_report.ok,
        "mode": "rebuild_refcoco_balanced_cache",
        "dataset_name": args.dataset_name,
        "records": stage_payload,
        "candidate_order_report": order_report,
        "alignment": alignment_payload,
        "raw_stage": stage_raw_payload,
        "cache_root": str(layout.root),
        "cache_validation": {
            "ok": cache_report.ok,
            "errors": cache_report.errors,
            "warnings": cache_report.warnings,
        },
        "errors": cache_report.errors,
        "warnings": cache_report.warnings,
    }


def _read_splits(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError("splits.json must be a non-empty object")
    return {str(split): list(source_ids) for split, source_ids in payload.items()}


def _ordered_splits(splits: dict[str, list[str]]) -> list[str]:
    order = ("train", "val", "testA", "testB", "test")
    ordered = [split for split in order if split in splits]
    ordered.extend(split for split in splits if split not in set(order))
    return ordered


if __name__ == "__main__":
    raise SystemExit(main())
