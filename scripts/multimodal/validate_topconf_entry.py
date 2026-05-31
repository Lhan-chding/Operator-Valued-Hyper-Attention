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
    parser.add_argument("--controlled-report", type=Path, required=True)
    parser.add_argument("--region-gate-report", type=Path, required=True)
    parser.add_argument("--sentiment-gate-report", type=Path, required=True)
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
        controlled_report = _read_json(args.controlled_report)
        region_gate_report = _read_json(args.region_gate_report)
        sentiment_gate_report = _read_json(args.sentiment_gate_report)
        cache_targets = _cache_targets(args.cache_target)
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
        }
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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
