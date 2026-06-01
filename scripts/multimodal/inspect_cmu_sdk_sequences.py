#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SUPPORTED_SUFFIXES = {".csd", ".h5", ".hdf5", ".json"}
ROLE_TOKENS = {
    "text": ("text", "bert", "glove", "word", "language", "transcript"),
    "audio": ("audio", "covarep", "openface2_audio", "acoustic", "voice"),
    "vision": ("visual", "vision", "facet", "openface", "image"),
    "labels": ("label", "labels", "opinion", "sentiment", "emotion"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a CMU MultimodalSDK download directory and suggest explicit "
            "extract_cmu_sdk_stage_inputs.py arguments."
        )
    )
    parser.add_argument("dataset_name", choices=("cmu_mosei", "cmu_mosi"))
    parser.add_argument("sdk_root", type=Path)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--stage-output-dir", default=None)
    parser.add_argument("--temporal-policy", choices=("strict", "mean", "first"), default="mean")
    parser.add_argument("--sentiment-column", type=int, default=0)
    parser.add_argument("--emotion-columns", default="1:")
    parser.add_argument("--license-tag", default="cmu-multimodal-sdk")
    parser.add_argument("--preprocessing-version", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        payload = inspect_cmu_sdk_sequences(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "ok": False,
            "dataset_name": args.dataset_name,
            "sdk_root": str(args.sdk_root),
            "errors": [str(exc)],
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def inspect_cmu_sdk_sequences(args: argparse.Namespace) -> dict[str, Any]:
    sdk_root = args.sdk_root
    if not sdk_root.exists():
        raise ValueError(f"sdk_root does not exist: {sdk_root}")
    splits_path = args.splits.resolve()
    sequence_paths = sorted(
        path
        for path in sdk_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and path.resolve() != splits_path
    )
    if not sequence_paths:
        raise ValueError(f"no CMU sequence files found under {sdk_root}")
    sequences = [_sequence_descriptor(path, sdk_root) for path in sequence_paths]
    suggested_roles = _suggest_roles(sequences)
    required_roles = ("text", "audio", "vision", "labels")
    missing = [role for role in required_roles if role not in suggested_roles]
    stage_output_dir = args.stage_output_dir or f"data/raw_multimodal/_downloads/{args.dataset_name}_stage_inputs"
    payload = {
        "ok": not missing,
        "dataset_name": args.dataset_name,
        "sdk_root": str(sdk_root),
        "sequence_count": len(sequences),
        "sequences": sequences,
        "suggested_roles": suggested_roles,
        "missing_suggested_roles": missing,
        "suggested_extract_command": (
            _extract_command(args, suggested_roles, stage_output_dir) if not missing else None
        ),
    }
    return payload


def _sequence_descriptor(path: Path, sdk_root: Path) -> dict[str, Any]:
    sequence = _read_sequence(path)
    source_ids = sorted(sequence)
    shapes = [tuple(int(dim) for dim in sequence[source_id].shape) for source_id in source_ids]
    unique_shapes = sorted({shape for shape in shapes})
    descriptor = {
        "path": str(path),
        "relative_path": str(path.relative_to(sdk_root)),
        "sample_count": len(source_ids),
        "first_source_id": source_ids[0] if source_ids else None,
        "feature_shapes": [list(shape) for shape in unique_shapes],
        "rank": int(len(unique_shapes[0])) if len(unique_shapes) == 1 and unique_shapes else None,
        "role_scores": _role_scores(path),
    }
    return descriptor


def _read_sequence(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".json":
        return _read_sequence_json(path)
    return _read_sequence_hdf5(path)


def _read_sequence_json(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text())
    raw_data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(raw_data, dict) or not raw_data:
        raise ValueError(f"{path} must contain a non-empty data object")
    data: dict[str, np.ndarray] = {}
    for source_id, raw_record in raw_data.items():
        if not isinstance(source_id, str) or not source_id.strip() or source_id != source_id.strip():
            raise ValueError(f"{path} contains an invalid source_id")
        features = raw_record.get("features") if isinstance(raw_record, dict) else raw_record
        data[source_id] = _as_array(features, source_id=source_id, source_name=str(path))
    return data


def _read_sequence_hdf5(path: Path) -> dict[str, np.ndarray]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError("reading CMU SDK .csd/.h5 files requires h5py; install it in the active .venv") from exc
    data: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        group = _hdf5_sample_group(handle, path, h5py)
        for source_id in group.keys():
            sample = group[source_id]
            if "features" not in sample:
                raise ValueError(f"{path} sample {source_id} missing features dataset")
            data[source_id] = _as_array(sample["features"][()], source_id=source_id, source_name=str(path))
    if not data:
        raise ValueError(f"{path} contains no sequence samples")
    return data


def _hdf5_sample_group(handle: Any, path: Path, h5py: Any) -> Any:
    candidates = []
    if "data" in handle and isinstance(handle["data"], h5py.Group):
        candidates.append(handle["data"])
    for value in handle.values():
        if isinstance(value, h5py.Group) and "data" in value and isinstance(value["data"], h5py.Group):
            candidates.append(value["data"])

    def collect(name: str, value: Any) -> None:
        if isinstance(value, h5py.Group):
            candidates.append(value)

    handle.visititems(collect)
    seen = set()
    unique_candidates = []
    for group in candidates:
        group_name = group.name
        if group_name not in seen:
            seen.add(group_name)
            unique_candidates.append(group)
    for group in unique_candidates:
        if _looks_like_cmu_sample_group(group, h5py):
            return group
    root_groups = ", ".join(str(key) for key in handle.keys())
    raise ValueError(f"{path} missing HDF5 sample group with per-source features; root groups: {root_groups}")


def _looks_like_cmu_sample_group(group: Any, h5py: Any) -> bool:
    checked = 0
    for key in group.keys():
        child = group[key]
        if isinstance(child, h5py.Group):
            checked += 1
            if "features" in child:
                return True
            if checked >= 20:
                return False
    return False


def _as_array(value: Any, *, source_id: str, source_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError(f"{source_name} sample {source_id} features must have at least one axis")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if not np.isfinite(array).all():
        raise ValueError(f"{source_name} sample {source_id} features contain non-finite values")
    return array


def _role_scores(path: Path) -> dict[str, int]:
    name = path.name.lower().replace("-", "_")
    scores = {}
    for role, tokens in ROLE_TOKENS.items():
        scores[role] = sum(1 for token in tokens if token in name)
    return scores


def _suggest_roles(sequences: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    suggestions: dict[str, dict[str, Any]] = {}
    for role in ("text", "audio", "vision", "labels"):
        ranked = sorted(
            sequences,
            key=lambda item: (int(item["role_scores"].get(role, 0)), -len(str(item["relative_path"]))),
            reverse=True,
        )
        if not ranked or int(ranked[0]["role_scores"].get(role, 0)) <= 0:
            continue
        top = ranked[0]
        suggestions[role] = {
            "path": top["path"],
            "relative_path": top["relative_path"],
            "sample_count": top["sample_count"],
            "feature_shapes": top["feature_shapes"],
            "score": int(top["role_scores"][role]),
        }
    return suggestions


def _extract_command(args: argparse.Namespace, roles: dict[str, dict[str, Any]], stage_output_dir: str) -> str:
    return (
        f"python scripts/multimodal/extract_cmu_sdk_stage_inputs.py {args.dataset_name} {stage_output_dir} "
        f"--splits {args.splits} "
        f"--text-sequence {roles['text']['path']} "
        f"--audio-sequence {roles['audio']['path']} "
        f"--visual-sequence {roles['vision']['path']} "
        f"--label-sequence {roles['labels']['path']} "
        f"--temporal-policy {args.temporal_policy} "
        f"--sentiment-column {args.sentiment_column} "
        f"--emotion-columns {args.emotion_columns} "
        f"--license-tag {args.license_tag} "
        f"--preprocessing-version {args.preprocessing_version}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
