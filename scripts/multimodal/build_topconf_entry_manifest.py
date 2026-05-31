#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a reproducible top-conference entry manifest for multimodal main experiments."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controlled-report", type=Path, required=True)
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
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    try:
        payload, exit_code = build_manifest(args)
    except ValueError as exc:
        payload = {
            "ok": False,
            "mode": "topconf_entry_manifest_build",
            "policy": "fail-fast: top-conference entry manifest inputs must be complete",
            "errors": [str(exc)],
            "warnings": [],
        }
        exit_code = 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def build_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if not args.cache_target:
        raise ValueError("at least one --cache-target is required")
    _require_exclusive_gate_source(args.region_gate_report, args.region_gate_bundle, "region gate")
    _require_exclusive_gate_source(args.sentiment_gate_report, args.sentiment_gate_bundle, "sentiment gate")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    base_dir = output.parent
    manifest: dict[str, Any] = {
        "controlled_report": _manifest_path(args.controlled_report, base_dir=base_dir),
        "cache_targets": [_cache_target_entry(item, base_dir=base_dir) for item in args.cache_target],
    }
    if args.region_gate_report is not None:
        manifest["region_gate_report"] = _manifest_path(args.region_gate_report, base_dir=base_dir)
    if args.region_gate_bundle is not None:
        manifest["region_gate_bundle"] = _manifest_path(args.region_gate_bundle, base_dir=base_dir)
    if args.sentiment_gate_report is not None:
        manifest["sentiment_gate_report"] = _manifest_path(args.sentiment_gate_report, base_dir=base_dir)
    if args.sentiment_gate_bundle is not None:
        manifest["sentiment_gate_bundle"] = _manifest_path(args.sentiment_gate_bundle, base_dir=base_dir)

    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validation = _run_validation(output) if args.validate else {
        "ok": True,
        "skipped": True,
        "reason": "--validate was not provided",
    }
    payload = {
        "ok": bool(validation.get("ok", False)),
        "mode": "topconf_entry_manifest_build",
        "policy": "top-conference entry paths, cache targets, and validation entrypoint are archived in one manifest",
        "manifest": str(output),
        "manifest_fields": sorted(manifest),
        "cache_targets": sorted(str(item["name"]) for item in manifest["cache_targets"]),
        "validation": validation,
    }
    return payload, 0 if payload["ok"] else 2


def _require_exclusive_gate_source(report_path: Path | None, bundle_path: Path | None, label: str) -> None:
    if report_path is not None and bundle_path is not None:
        raise ValueError(f"{label} must use either direct report path or bundle directory, not both")
    if report_path is None and bundle_path is None:
        raise ValueError(f"{label} requires either direct report path or bundle directory")


def _cache_target_entry(item: list[str], *, base_dir: Path) -> dict[str, Any]:
    name, cache_root, dataset, version, splits_text = item
    splits = [split.strip() for split in splits_text.split(",") if split.strip()]
    if not name.strip():
        raise ValueError("cache target NAME must be non-empty")
    if not dataset.strip():
        raise ValueError(f"cache target {name} dataset must be non-empty")
    if not version.strip():
        raise ValueError(f"cache target {name} version must be non-empty")
    if not splits:
        raise ValueError(f"cache target {name} must include at least one split")
    return {
        "name": name.strip(),
        "cache_root": _manifest_path(Path(cache_root), base_dir=base_dir),
        "dataset": dataset.strip(),
        "version": version.strip(),
        "splits": splits,
    }


def _manifest_path(path: Path, *, base_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return Path(os.path.relpath(resolved, start=base_dir.resolve())).as_posix()
    except ValueError:
        return str(resolved)


def _run_validation(manifest_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "multimodal" / "validate_topconf_entry.py"),
            "--manifest",
            str(manifest_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {
            "ok": False,
            "errors": ["validate_topconf_entry.py did not emit valid JSON"],
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    payload["returncode"] = result.returncode
    if result.stderr.strip():
        payload["stderr"] = result.stderr
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
