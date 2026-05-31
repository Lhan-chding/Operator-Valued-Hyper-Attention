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

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
    CacheValidationTarget,
    validate_topconf_main_experiment_entry,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that controlled, public gate, and cache artifacts are strong enough "
            "to enter top-conference multimodal main experiments."
        )
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--controlled-report", type=Path)
    parser.add_argument("--region-gate-report", type=Path)
    parser.add_argument("--region-gate-bundle", type=Path)
    parser.add_argument("--sentiment-gate-report", type=Path)
    parser.add_argument("--sentiment-gate-bundle", type=Path)
    parser.add_argument(
        "--cache-target",
        nargs=5,
        action="append",
        metavar=("NAME", "CACHE_ROOT", "DATASET", "VERSION", "SPLITS"),
        default=[],
        help="Cache target as: NAME CACHE_ROOT DATASET VERSION comma,separated,splits",
    )
    args = parser.parse_args()

    try:
        spec = _entry_spec(args)
        controlled_report = _read_json(spec["controlled_report"])
        region_gate_path = _gate_report_path(
            report_path=spec["region_gate_report"],
            bundle_root=spec["region_gate_bundle"],
            bundle_filename="region_text_gate_report.json",
            label="region gate",
        )
        sentiment_gate_path = _gate_report_path(
            report_path=spec["sentiment_gate_report"],
            bundle_root=spec["sentiment_gate_bundle"],
            bundle_filename="sentiment_gate_report.json",
            label="sentiment gate",
        )
        region_gate_report = _read_json(region_gate_path)
        sentiment_gate_report = _read_json(sentiment_gate_path)
        cache_targets = _cache_targets(spec["cache_target"])
        report = validate_topconf_main_experiment_entry(
            controlled_report=controlled_report,
            region_gate_report=region_gate_report,
            sentiment_gate_report=sentiment_gate_report,
            cache_targets=cache_targets,
        )
        payload = {
            "ok": report.ok,
            "mode": "topconf_main_entry_validation",
            "policy": "all controlled, public-gate, robustness, statistics, and cache evidence must validate",
            "errors": report.errors,
            "warnings": report.warnings,
            "cache_targets": sorted(cache_targets),
            "gate_sources": {
                "region_text_public": str(region_gate_path),
                "sentiment_emotion_public": str(sentiment_gate_path),
            },
        }
        if spec["manifest"] is not None:
            payload["manifest"] = str(spec["manifest"])
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if report.ok else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": "topconf_main_entry_validation",
                    "policy": "fail-fast: top-conference entry artifacts must be readable and well-formed",
                    "errors": [str(exc)],
                    "warnings": [],
                    "cache_targets": [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


def _entry_spec(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest is not None:
        return _entry_spec_from_manifest(args)
    if args.controlled_report is None:
        raise ValueError("--controlled-report is required unless --manifest is provided")
    return {
        "manifest": None,
        "controlled_report": args.controlled_report,
        "region_gate_report": args.region_gate_report,
        "region_gate_bundle": args.region_gate_bundle,
        "sentiment_gate_report": args.sentiment_gate_report,
        "sentiment_gate_bundle": args.sentiment_gate_bundle,
        "cache_target": args.cache_target,
    }


def _entry_spec_from_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if any(
        value
        for value in (
            args.controlled_report,
            args.region_gate_report,
            args.region_gate_bundle,
            args.sentiment_gate_report,
            args.sentiment_gate_bundle,
            args.cache_target,
        )
    ):
        raise ValueError("--manifest cannot be combined with direct report, bundle, or cache-target arguments")
    manifest = _read_json(args.manifest)
    base_dir = args.manifest.parent
    return {
        "manifest": args.manifest,
        "controlled_report": _manifest_path(manifest, "controlled_report", base_dir=base_dir),
        "region_gate_report": _optional_manifest_path(manifest, "region_gate_report", base_dir=base_dir),
        "region_gate_bundle": _optional_manifest_path(manifest, "region_gate_bundle", base_dir=base_dir),
        "sentiment_gate_report": _optional_manifest_path(manifest, "sentiment_gate_report", base_dir=base_dir),
        "sentiment_gate_bundle": _optional_manifest_path(manifest, "sentiment_gate_bundle", base_dir=base_dir),
        "cache_target": _manifest_cache_targets(manifest, base_dir=base_dir),
    }


def _manifest_path(manifest: dict[str, Any], key: str, *, base_dir: Path) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest missing required path: {key}")
    return _resolve_manifest_path(value, base_dir=base_dir)


def _optional_manifest_path(manifest: dict[str, Any], key: str, *, base_dir: Path) -> Path | None:
    value = manifest.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest path must be a non-empty string: {key}")
    return _resolve_manifest_path(value, base_dir=base_dir)


def _resolve_manifest_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _manifest_cache_targets(manifest: dict[str, Any], *, base_dir: Path) -> list[list[str]]:
    raw_targets = manifest.get("cache_targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("manifest cache_targets must be a non-empty list")
    targets: list[list[str]] = []
    for index, item in enumerate(raw_targets):
        if not isinstance(item, dict):
            raise ValueError(f"manifest cache_targets[{index}] must be an object")
        name = _required_manifest_text(item, "name", index)
        cache_root = _resolve_manifest_path(_required_manifest_text(item, "cache_root", index), base_dir=base_dir)
        dataset = str(item.get("dataset", item.get("dataset_name", ""))).strip()
        if not dataset:
            raise ValueError(f"manifest cache_targets[{index}] missing dataset")
        version = _required_manifest_text(item, "version", index)
        splits = item.get("splits")
        if isinstance(splits, str):
            split_text = splits
        elif isinstance(splits, list) and all(isinstance(split, str) and split.strip() for split in splits):
            split_text = ",".join(str(split).strip() for split in splits)
        else:
            raise ValueError(f"manifest cache_targets[{index}].splits must be a non-empty string list or comma string")
        targets.append([name, str(cache_root), dataset, version, split_text])
    return targets


def _required_manifest_text(item: dict[str, Any], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest cache_targets[{index}] missing {key}")
    return value.strip()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _gate_report_path(
    *,
    report_path: Path | None,
    bundle_root: Path | None,
    bundle_filename: str,
    label: str,
) -> Path:
    if report_path is not None and bundle_root is not None:
        raise ValueError(f"{label} must use either direct report path or bundle directory, not both")
    if report_path is not None:
        return report_path
    if bundle_root is not None:
        return bundle_root / bundle_filename
    raise ValueError(f"{label} requires either direct report path or bundle directory")


def _cache_targets(values: list[list[str]]) -> dict[str, CacheValidationTarget]:
    targets: dict[str, CacheValidationTarget] = {}
    for item in values:
        name, cache_root, dataset_name, version, splits_text = item
        if not name.strip():
            raise ValueError("cache target NAME must be non-empty")
        if name in targets:
            raise ValueError(f"duplicate cache target: {name}")
        splits = tuple(split.strip() for split in splits_text.split(",") if split.strip())
        if not splits:
            raise ValueError(f"cache target {name} must include at least one split")
        targets[name] = CacheValidationTarget(
            layout=MultimodalCacheLayout(Path(cache_root), dataset_name, version),
            splits=splits,
        )
    return targets


if __name__ == "__main__":
    raise SystemExit(main())
