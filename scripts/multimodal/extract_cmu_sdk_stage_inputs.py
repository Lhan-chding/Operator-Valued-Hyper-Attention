#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SPLIT_ORDER = ("train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract CMU-MOSEI/MOSI SDK computational sequences into the .npy "
            "stage inputs consumed by scripts/multimodal/stage_cmu_sentiment_raw.py."
        )
    )
    parser.add_argument("dataset_name", choices=("cmu_mosei", "cmu_mosi"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--text-sequence", type=Path, required=True)
    parser.add_argument("--audio-sequence", type=Path, required=True)
    parser.add_argument("--visual-sequence", type=Path, required=True)
    parser.add_argument("--label-sequence", type=Path, required=True)
    parser.add_argument("--temporal-policy", choices=("strict", "mean", "first"), default="strict")
    parser.add_argument("--sentiment-column", type=int, default=0)
    parser.add_argument("--emotion-columns", default="1:")
    parser.add_argument("--license-tag", default="cmu-multimodal-sdk")
    parser.add_argument("--preprocessing-version", required=True)
    args = parser.parse_args()

    try:
        payload = extract_cmu_sdk_stage_inputs(args)
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


def extract_cmu_sdk_stage_inputs(args: argparse.Namespace) -> dict[str, Any]:
    splits = _read_splits(args.splits)
    source_ids = _ordered_source_ids(splits)
    sequences = {
        "text": _read_sequence(args.text_sequence),
        "audio": _read_sequence(args.audio_sequence),
        "vision": _read_sequence(args.visual_sequence),
        "labels": _read_sequence(args.label_sequence),
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_outputs = {
        "text": _feature_tensor(sequences["text"], source_ids, args.temporal_policy, sequence_name="text"),
        "audio": _feature_tensor(sequences["audio"], source_ids, args.temporal_policy, sequence_name="audio"),
        "vision": _feature_tensor(sequences["vision"], source_ids, args.temporal_policy, sequence_name="vision"),
    }
    labels = _label_matrix(sequences["labels"], source_ids, args.temporal_policy)
    emotion_columns = _column_indices(args.emotion_columns, labels.shape[1], option_name="--emotion-columns")
    sentiment_column = _single_column(args.sentiment_column, labels.shape[1], option_name="--sentiment-column")

    split_path = output_dir / f"{args.dataset_name}_splits.json"
    text_path = output_dir / f"{args.dataset_name}_text_features.npy"
    audio_path = output_dir / f"{args.dataset_name}_audio_features.npy"
    visual_path = output_dir / f"{args.dataset_name}_visual_features.npy"
    sentiment_path = output_dir / f"{args.dataset_name}_sentiment.npy"
    emotion_path = output_dir / f"{args.dataset_name}_emotion.npy"
    manifest_path = output_dir / f"{args.dataset_name}_stage_input_manifest.json"

    split_path.write_text(json.dumps(splits, sort_keys=True) + "\n")
    np.save(text_path, feature_outputs["text"])
    np.save(audio_path, feature_outputs["audio"])
    np.save(visual_path, feature_outputs["vision"])
    np.save(sentiment_path, labels[:, sentiment_column : sentiment_column + 1])
    np.save(emotion_path, labels[:, emotion_columns])

    manifest = {
        "dataset_name": args.dataset_name,
        "sample_count": len(source_ids),
        "splits": splits,
        "source_paths": {
            "text": str(args.text_sequence),
            "audio": str(args.audio_sequence),
            "vision": str(args.visual_sequence),
            "labels": str(args.label_sequence),
        },
        "temporal_policy": args.temporal_policy,
        "sentiment_column": sentiment_column,
        "emotion_columns": emotion_columns,
        "license_tag": args.license_tag,
        "preprocessing_version": args.preprocessing_version,
        "outputs": {
            "splits": str(split_path),
            "text_features": str(text_path),
            "audio_features": str(audio_path),
            "visual_features": str(visual_path),
            "sentiment_labels": str(sentiment_path),
            "emotion_labels": str(emotion_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "ok": True,
        "dataset_name": args.dataset_name,
        "output_dir": str(output_dir),
        "sample_count": len(source_ids),
        "feature_shapes": {name: list(array.shape) for name, array in feature_outputs.items()},
        "label_shapes": {
            "sentiment": list(np.load(sentiment_path, allow_pickle=False).shape),
            "emotion": list(np.load(emotion_path, allow_pickle=False).shape),
        },
        "manifest": str(manifest_path),
        "next": _stage_command(args, manifest["outputs"]),
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


def _read_sequence(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_sequence_json(path)
    return _read_sequence_hdf5(path)


def _read_sequence_json(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text())
    raw_data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(raw_data, dict) or not raw_data:
        raise ValueError(f"{path} must contain a non-empty data object")
    data = {}
    for source_id, raw_record in raw_data.items():
        if not isinstance(source_id, str) or not source_id.strip() or source_id != source_id.strip():
            raise ValueError(f"{path} contains an invalid source_id")
        features = raw_record.get("features") if isinstance(raw_record, dict) else raw_record
        data[source_id] = _as_feature_array(features, source_id=source_id, source_name=str(path))
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
            data[source_id] = _as_feature_array(sample["features"][()], source_id=source_id, source_name=str(path))
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


def _as_feature_array(value: Any, *, source_id: str, source_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError(f"{source_name} sample {source_id} features must have at least one axis")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if not np.isfinite(array).all():
        raise ValueError(f"{source_name} sample {source_id} features contain non-finite values")
    return array


def _feature_tensor(
    sequence: dict[str, np.ndarray],
    source_ids: list[str],
    temporal_policy: str,
    *,
    sequence_name: str,
) -> np.ndarray:
    rows = [_sequence_row(sequence, source_id, sequence_name=sequence_name) for source_id in source_ids]
    reduced = [_apply_temporal_policy(row, temporal_policy) for row in rows]
    shapes = {tuple(row.shape) for row in reduced}
    if len(shapes) != 1:
        raise ValueError(
            f"{sequence_name} features have variable shapes after temporal policy {temporal_policy}: "
            + ", ".join(str(shape) for shape in sorted(shapes))
        )
    return np.stack(reduced, axis=0).astype(np.float32, copy=False)


def _label_matrix(sequence: dict[str, np.ndarray], source_ids: list[str], temporal_policy: str) -> np.ndarray:
    rows = [_sequence_row(sequence, source_id, sequence_name="labels") for source_id in source_ids]
    reduced = [_apply_label_policy(row, temporal_policy) for row in rows]
    shapes = {tuple(row.shape) for row in reduced}
    if len(shapes) != 1:
        raise ValueError("label features have variable dimensions after temporal reduction")
    return np.stack(reduced, axis=0).astype(np.float32, copy=False)


def _sequence_row(sequence: dict[str, np.ndarray], source_id: str, *, sequence_name: str) -> np.ndarray:
    if source_id not in sequence:
        raise ValueError(f"{sequence_name} sequence missing source_id from splits: {source_id}")
    return sequence[source_id]


def _apply_temporal_policy(array: np.ndarray, temporal_policy: str) -> np.ndarray:
    if temporal_policy == "strict":
        return array
    if temporal_policy == "mean":
        return array.mean(axis=0, keepdims=True)
    if temporal_policy == "first":
        return array[:1]
    raise ValueError(f"unsupported temporal policy: {temporal_policy}")


def _apply_label_policy(array: np.ndarray, temporal_policy: str) -> np.ndarray:
    if temporal_policy == "strict" and array.shape[0] != 1:
        raise ValueError("strict label extraction requires one label row per source_id")
    if temporal_policy == "first" or temporal_policy == "strict":
        return np.asarray(array[0]).reshape(-1)
    if temporal_policy == "mean":
        return np.asarray(array.mean(axis=0)).reshape(-1)
    raise ValueError(f"unsupported temporal policy: {temporal_policy}")


def _single_column(index: int, column_count: int, *, option_name: str) -> int:
    if index < 0:
        index += column_count
    if index < 0 or index >= column_count:
        raise ValueError(f"{option_name} index {index} is outside label width {column_count}")
    return index


def _column_indices(value: str, column_count: int, *, option_name: str) -> list[int]:
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else column_count
        if start < 0:
            start += column_count
        if end < 0:
            end += column_count
        indices = list(range(start, end))
    else:
        indices = [int(piece) for piece in value.split(",") if piece.strip()]
    if not indices:
        raise ValueError(f"{option_name} must select at least one column")
    return [_single_column(index, column_count, option_name=option_name) for index in indices]


def _stage_command(args: argparse.Namespace, outputs: dict[str, str]) -> str:
    return (
        f"python scripts/multimodal/stage_cmu_sentiment_raw.py {args.dataset_name} "
        f"data/raw_multimodal/{args.dataset_name} "
        f"--splits {outputs['splits']} "
        f"--text-features {outputs['text_features']} "
        f"--audio-features {outputs['audio_features']} "
        f"--visual-features {outputs['visual_features']} "
        f"--sentiment-labels {outputs['sentiment_labels']} "
        f"--emotion-labels {outputs['emotion_labels']} "
        f"--feature-version text={args.preprocessing_version}:text "
        f"--feature-version audio={args.preprocessing_version}:audio "
        f"--feature-version vision={args.preprocessing_version}:vision "
        f"--license-tag {args.license_tag} "
        f"--preprocessing-version {args.preprocessing_version}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
