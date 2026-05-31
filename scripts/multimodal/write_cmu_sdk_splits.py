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
            "Write project-ready train/val/test splits from CMU MultimodalSDK "
            "standard folds or an equivalent fold JSON export."
        )
    )
    parser.add_argument("dataset_name", choices=("cmu_mosei", "cmu_mosi"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--folds-json", type=Path)
    parser.add_argument("--sequence", action="append", default=[], type=Path)
    args = parser.parse_args()

    try:
        payload = write_cmu_sdk_splits(args)
    except (OSError, ValueError, json.JSONDecodeError, ImportError, AttributeError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "dataset_name": args.dataset_name,
                    "output": str(args.output),
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


def write_cmu_sdk_splits(args: argparse.Namespace) -> dict[str, Any]:
    raw_folds, source = _load_raw_folds(args.dataset_name, args.folds_json)
    splits = _normalize_folds(raw_folds)
    sequence_validation = _validate_sequences(splits, args.sequence)
    if not sequence_validation["ok"]:
        raise ValueError("; ".join(sequence_validation["errors"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(splits, sort_keys=True) + "\n")
    return {
        "ok": True,
        "dataset_name": args.dataset_name,
        "output": str(args.output),
        "source": source,
        "split_counts": {split: len(source_ids) for split, source_ids in sorted(splits.items())},
        "sequence_validation": sequence_validation,
    }


def _load_raw_folds(dataset_name: str, folds_json: Path | None) -> tuple[dict[str, Any], str]:
    if folds_json is not None:
        payload = json.loads(folds_json.read_text())
        if not isinstance(payload, dict):
            raise ValueError("--folds-json must contain an object")
        return payload, "folds_json"
    try:
        import mmdatasdk  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("mmdatasdk is required unless --folds-json is provided") from exc
    dataset = getattr(mmdatasdk, dataset_name)
    standard_folds = getattr(dataset, "standard_folds")
    return {
        "standard_train_fold": getattr(standard_folds, "standard_train_fold"),
        "standard_valid_fold": getattr(standard_folds, "standard_valid_fold"),
        "standard_test_fold": getattr(standard_folds, "standard_test_fold"),
    }, "mmdatasdk.standard_folds"


def _normalize_folds(raw_folds: dict[str, Any]) -> dict[str, list[str]]:
    aliases = {
        "train": ("train", "standard_train_fold", "train_fold"),
        "val": ("val", "valid", "validation", "dev", "standard_valid_fold", "valid_fold"),
        "test": ("test", "standard_test_fold", "test_fold"),
    }
    splits = {}
    for split, keys in aliases.items():
        values = None
        for key in keys:
            if key in raw_folds:
                values = raw_folds[key]
                break
        if values is None:
            raise ValueError(f"folds missing required {split} split")
        splits[split] = _normalize_source_ids(values, split)
    _validate_no_overlap(splits)
    return splits


def _normalize_source_ids(values: Any, split: str) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{split} fold must be a non-empty source_id list")
    source_ids = []
    for value in values:
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{split} fold contains an invalid source_id")
        source_ids.append(value)
    if len(source_ids) != len(set(source_ids)):
        duplicates = sorted(source_id for source_id in set(source_ids) if source_ids.count(source_id) > 1)
        raise ValueError(f"{split} fold contains duplicate source_id: {', '.join(duplicates)}")
    return source_ids


def _validate_no_overlap(splits: dict[str, list[str]]) -> None:
    seen = {}
    for split, source_ids in splits.items():
        for source_id in source_ids:
            previous = seen.get(source_id)
            if previous is not None:
                raise ValueError(f"source_id appears in multiple folds: {source_id} ({previous}, {split})")
            seen[source_id] = split


def _validate_sequences(splits: dict[str, list[str]], sequence_paths: list[Path]) -> dict[str, Any]:
    if not sequence_paths:
        return {"ok": True, "checked_sequences": 0, "errors": []}
    expected = {source_id for source_ids in splits.values() for source_id in source_ids}
    errors = []
    sequence_reports = []
    for path in sequence_paths:
        observed = _read_sequence_source_ids(path)
        missing = sorted(expected - observed)
        sequence_reports.append(
            {
                "path": str(path),
                "source_id_count": len(observed),
                "missing_count": len(missing),
                "missing_preview": missing[:10],
            }
        )
        if missing:
            errors.append(f"{path} missing {len(missing)} split source_id values; first missing: {', '.join(missing[:10])}")
    return {"ok": not errors, "checked_sequences": len(sequence_paths), "sequences": sequence_reports, "errors": errors}


def _read_sequence_source_ids(path: Path) -> set[str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        raw_data = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(raw_data, dict):
            raise ValueError(f"{path} must contain a data object or source_id-keyed object")
        return {source_id for source_id in raw_data if isinstance(source_id, str)}
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("sequence validation for .csd/.h5 files requires h5py") from exc
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"{path} missing root data group")
        return {str(source_id) for source_id in handle["data"].keys()}


if __name__ == "__main__":
    raise SystemExit(main())
