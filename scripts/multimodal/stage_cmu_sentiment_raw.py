#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SPLIT_ORDER = ("train", "val", "test")
MODALITIES = ("text", "audio", "vision")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage extracted CMU-MOSEI/MOSI frozen features into the raw manifest "
            "layout consumed by scripts/multimodal/build_cache.py."
        )
    )
    parser.add_argument("dataset_name", choices=("cmu_mosei", "cmu_mosi"))
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--visual-features", type=Path, required=True)
    parser.add_argument("--sentiment-labels", type=Path, required=True)
    parser.add_argument("--emotion-labels", type=Path, required=True)
    parser.add_argument("--missing-modality-mask", type=Path)
    parser.add_argument("--corruption-transforms", type=Path)
    parser.add_argument("--failed-samples", type=Path)
    parser.add_argument("--feature-version", action="append", default=[], metavar="MODALITY=VERSION")
    parser.add_argument("--license-tag", default="cmu-multimodal-sdk")
    parser.add_argument("--preprocessing-version", required=True)
    parser.add_argument("--transcript-source", default="official_transcript")
    args = parser.parse_args()

    try:
        payload = stage_cmu_sentiment_raw(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "dataset_name": args.dataset_name,
                    "raw_root": str(args.raw_root),
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


def stage_cmu_sentiment_raw(args: argparse.Namespace) -> dict[str, Any]:
    splits = _read_splits(args.splits)
    source_ids = _ordered_source_ids(splits)
    sample_count = len(source_ids)
    raw_root = args.raw_root
    for folder in ("features", "labels", "metadata", "provenance"):
        (raw_root / folder).mkdir(parents=True, exist_ok=True)

    feature_paths = {
        "text": args.text_features,
        "audio": args.audio_features,
        "vision": args.visual_features,
    }
    feature_shapes = {
        modality: _copy_npy(source, raw_root / "features" / f"{artifact_name}_features.npy", sample_count)
        for modality, source, artifact_name in (
            ("text", feature_paths["text"], "text"),
            ("audio", feature_paths["audio"], "audio"),
            ("vision", feature_paths["vision"], "visual"),
        )
    }
    sentiment_shape = _copy_npy(args.sentiment_labels, raw_root / "labels" / "sentiment.npy", sample_count)
    emotion_shape = _copy_npy(args.emotion_labels, raw_root / "labels" / "emotion.npy", sample_count)

    if args.missing_modality_mask is None:
        np.save(raw_root / "metadata" / "missing_modality_mask.npy", np.zeros((sample_count, len(MODALITIES)), dtype=bool))
    else:
        _copy_npy(args.missing_modality_mask, raw_root / "metadata" / "missing_modality_mask.npy", sample_count)

    if args.corruption_transforms is None:
        (raw_root / "metadata" / "corruption_transforms.json").write_text(
            json.dumps({"version": "none", "transforms": []}, sort_keys=True) + "\n"
        )
    else:
        _copy_json(args.corruption_transforms, raw_root / "metadata" / "corruption_transforms.json")

    if args.failed_samples is not None:
        _copy_text(args.failed_samples, raw_root / "provenance" / "failed_samples.jsonl")

    (raw_root / "splits.json").write_text(json.dumps(splits, sort_keys=True) + "\n")
    (raw_root / "metadata" / "utterances.json").write_text(
        json.dumps(
            {
                "records": _records(
                    splits,
                    dataset_name=args.dataset_name,
                    license_tag=args.license_tag,
                    preprocessing_version=args.preprocessing_version,
                    transcript_source=args.transcript_source,
                )
            },
            sort_keys=True,
        )
        + "\n"
    )
    (raw_root / "metadata" / "dialogues.json").write_text(json.dumps({"records": []}, sort_keys=True) + "\n")
    (raw_root / "metadata" / "feature_versions.json").write_text(
        json.dumps(_feature_versions(args.feature_version), sort_keys=True) + "\n"
    )
    return {
        "ok": True,
        "dataset_name": args.dataset_name,
        "raw_root": str(raw_root),
        "sample_count": sample_count,
        "feature_shapes": feature_shapes,
        "label_shapes": {"sentiment": sentiment_shape, "emotion": emotion_shape},
        "next": (
            f"python scripts/multimodal/build_cache.py {args.dataset_name} "
            f"{raw_root} data/multimodal_cache --version v0.1"
        ),
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


def _copy_npy(source: Path, destination: Path, sample_count: int) -> tuple[int, ...]:
    array = np.load(source, allow_pickle=False)
    if array.ndim == 0 or int(array.shape[0]) != sample_count:
        raise ValueError(f"{source} first axis must match split sample count {sample_count}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, array)
    return tuple(int(dim) for dim in array.shape)


def _copy_json(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _copy_text(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text())


def _records(
    splits: dict[str, list[str]],
    *,
    dataset_name: str,
    license_tag: str,
    preprocessing_version: str,
    transcript_source: str,
) -> list[dict[str, str]]:
    records = []
    for split in [name for name in SPLIT_ORDER if name in splits] + [name for name in splits if name not in set(SPLIT_ORDER)]:
        for source_id in splits[split]:
            records.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "original_split": split,
                    "raw_ref": f"{dataset_name}://{source_id}",
                    "license_tag": license_tag,
                    "preprocessing_version": preprocessing_version,
                    "utterance_id": source_id,
                    "dialogue_id": _dialogue_id(source_id),
                    "speaker_id": "unknown_speaker",
                    "transcript_source": transcript_source,
                }
            )
    return records


def _dialogue_id(source_id: str) -> str:
    if "[" in source_id:
        video_id = source_id.split("[", 1)[0].strip()
        if video_id:
            return video_id
    for separator in ("::", "/", "#"):
        if separator in source_id:
            return source_id.split(separator, 1)[0]
    if "-" in source_id:
        candidate = source_id.rsplit("-", 1)[0]
        if candidate.strip() and any(character.isalnum() for character in candidate):
            return candidate
    return source_id


def _feature_versions(values: list[str]) -> dict[str, str]:
    versions = {modality: "unspecified" for modality in MODALITIES}
    for value in values:
        if "=" not in value:
            raise ValueError("--feature-version values must be MODALITY=VERSION")
        key, version = value.split("=", 1)
        key = "vision" if key.strip() == "visual" else key.strip()
        if key not in versions:
            raise ValueError(f"unknown feature-version modality: {key}")
        if not version.strip():
            raise ValueError("--feature-version values must include a non-empty VERSION")
        versions[key] = version.strip()
    return versions


if __name__ == "__main__":
    raise SystemExit(main())
