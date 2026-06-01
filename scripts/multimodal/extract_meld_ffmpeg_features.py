#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


SPLIT_ORDER = ("train", "val", "test")
MODALITIES = ("text", "audio", "vision")
TEXT_TOKEN_RE = re.compile(r"[a-z0-9']+")
AUDIO_FEATURE_DIM = 12


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract deterministic diagnostic MELD features from prepared MELD.Raw CSV/mp4 files "
            "into the raw manifest layout consumed by scripts/multimodal/build_cache.py. "
            "Use extract_meld_transformer_features.py for formal public-main experiments."
        )
    )
    parser.add_argument("meld_root", type=Path)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--text-dim", type=int, default=768)
    parser.add_argument("--audio-steps", type=int, default=32)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--visual-frames", type=int, default=16)
    parser.add_argument("--visual-size", type=int, default=8)
    parser.add_argument("--visual-fps", type=float, default=2.0)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument(
        "--min-video-coverage",
        type=float,
        default=0.95,
        help="Fail if fewer than this fraction of retained samples have an mp4 path.",
    )
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args()

    try:
        payload = extract_meld_ffmpeg_features(args)
    except (OSError, ValueError, json.JSONDecodeError, csv.Error, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "meld_root": str(args.meld_root),
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


def extract_meld_ffmpeg_features(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    meld_root = args.meld_root
    raw_root = args.raw_root
    if not meld_root.exists():
        raise ValueError(f"meld_root does not exist: {meld_root}")
    if not (raw_root / "splits.json").exists():
        raise ValueError(f"raw_root missing splits.json: {raw_root}")
    if not (raw_root / "metadata" / "dialogues.json").exists():
        raise ValueError(f"raw_root missing metadata/dialogues.json: {raw_root}")

    splits = _read_splits(raw_root / "splits.json")
    ordered_source_ids = _ordered_source_ids(splits)
    records_by_source_id = _read_records(raw_root / "metadata" / "dialogues.json")
    csv_text_by_source_id = _read_meld_csv_text(meld_root)
    video_index = _video_index(meld_root)
    samples = [
        _sample_spec(source_id, records_by_source_id, csv_text_by_source_id, video_index)
        for source_id in ordered_source_ids
    ]
    missing_records = [source_id for source_id in ordered_source_ids if source_id not in records_by_source_id]
    if missing_records:
        raise ValueError("metadata/dialogues.json missing source_id records: " + ", ".join(missing_records[:10]))

    video_coverage = sum(1 for sample in samples if sample["video_path"]) / max(1, len(samples))
    if video_coverage < args.min_video_coverage:
        raise ValueError(
            f"MELD mp4 coverage {video_coverage:.4f} is below --min-video-coverage {args.min_video_coverage:.4f}; "
            "extract nested train/dev/test tarballs first or lower the threshold only for debugging."
        )
    if any(sample["video_path"] for sample in samples) and shutil.which(args.ffmpeg_bin) is None:
        raise ValueError(f"ffmpeg executable not found: {args.ffmpeg_bin}; install ffmpeg before feature extraction")

    feature_dir = raw_root / "features"
    metadata_dir = raw_root / "metadata"
    feature_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    sample_count = len(samples)
    text_path = feature_dir / "text_features.npy"
    audio_path = feature_dir / "audio_features.npy"
    visual_path = feature_dir / "visual_features.npy"
    text = np.lib.format.open_memmap(text_path, mode="w+", dtype=np.float32, shape=(sample_count, 1, args.text_dim))
    audio = np.lib.format.open_memmap(
        audio_path,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, args.audio_steps, AUDIO_FEATURE_DIM),
    )
    visual_dim = args.visual_size * args.visual_size * 3
    visual = np.lib.format.open_memmap(
        visual_path,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, args.visual_frames, visual_dim),
    )
    missing_mask = np.zeros((sample_count, len(MODALITIES)), dtype=bool)

    for index, sample in enumerate(samples):
        text[index, 0, :] = _hash_text_features(str(sample["text"]), args.text_dim)
        if not str(sample["text"]).strip():
            missing_mask[index, 0] = True

    extraction_tasks = [
        {
            "index": index,
            "source_id": str(sample["source_id"]),
            "video_path": str(sample["video_path"]) if sample["video_path"] else "",
            "ffmpeg_bin": args.ffmpeg_bin,
            "audio_steps": args.audio_steps,
            "audio_sample_rate": args.audio_sample_rate,
            "visual_frames": args.visual_frames,
            "visual_size": args.visual_size,
            "visual_fps": args.visual_fps,
            "timeout_sec": args.timeout_sec,
        }
        for index, sample in enumerate(samples)
    ]
    failures: list[dict[str, str]] = []
    for result in _iter_video_results(extraction_tasks, args.workers):
        index = int(result["index"])
        audio[index, :, :] = result["audio"]
        visual[index, :, :] = result["visual"]
        missing_mask[index, 1] = bool(result["audio_missing"])
        missing_mask[index, 2] = bool(result["visual_missing"])
        for error in result["errors"]:
            failures.append({"source_id": str(result["source_id"]), "reason": str(error)})

    text.flush()
    audio.flush()
    visual.flush()
    np.save(metadata_dir / "missing_modality_mask.npy", missing_mask)
    _write_feature_versions(metadata_dir / "feature_versions.json", args)
    manifest = _manifest(args, samples, failures, video_coverage, visual_dim)
    manifest_path = args.output_manifest or metadata_dir / "meld_ffmpeg_feature_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "ok": True,
        **manifest,
        "next": "python scripts/multimodal/build_cache.py meld data/raw_multimodal/meld data/multimodal_cache --version v0.1",
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "workers": args.workers,
        "text_dim": args.text_dim,
        "audio_steps": args.audio_steps,
        "audio_sample_rate": args.audio_sample_rate,
        "visual_frames": args.visual_frames,
        "visual_size": args.visual_size,
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    if args.visual_fps <= 0:
        raise ValueError("--visual-fps must be positive")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be positive")
    if not (0 <= args.min_video_coverage <= 1):
        raise ValueError("--min-video-coverage must be in [0, 1]")


def _read_splits(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError("splits.json must be a non-empty JSON object")
    splits: dict[str, list[str]] = {}
    for split, source_ids in payload.items():
        if not isinstance(split, str) or not split.strip() or split != split.strip():
            raise ValueError("split names must be non-empty normalized strings")
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError(f"split {split} must contain a non-empty source_id list")
        splits[split] = [_normalized_source_id(source_id, f"split {split}") for source_id in source_ids]
    return splits


def _ordered_source_ids(splits: dict[str, list[str]]) -> list[str]:
    ordered_splits = [split for split in SPLIT_ORDER if split in splits]
    ordered_splits.extend(split for split in splits if split not in set(SPLIT_ORDER))
    source_ids = [source_id for split in ordered_splits for source_id in splits[split]]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source_id values must be unique across splits")
    return source_ids


def _read_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a non-empty records list")
    by_source_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path} records[{index}] must be an object")
        source_id = _normalized_source_id(record.get("source_id"), f"{path} records[{index}]")
        if source_id in by_source_id:
            raise ValueError(f"{path} contains duplicate source_id: {source_id}")
        by_source_id[source_id] = record
    return by_source_id


def _normalized_source_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{context} contains an invalid source_id")
    return value


def _read_meld_csv_text(meld_root: Path) -> dict[str, str]:
    text_by_source_id: dict[str, str] = {}
    for csv_path in sorted(meld_root.rglob("*_sent_emo.csv")):
        split = _split_from_csv_name(csv_path.name)
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                dialogue_id = str(row.get("Dialogue_ID", "")).strip()
                utterance_id = str(row.get("Utterance_ID", "")).strip()
                if not dialogue_id or not utterance_id:
                    continue
                text_by_source_id[f"{split}/dia{dialogue_id}_utt{utterance_id}"] = str(row.get("Utterance", ""))
    return text_by_source_id


def _split_from_csv_name(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("train_"):
        return "train"
    if lowered.startswith("dev_"):
        return "val"
    if lowered.startswith("test_"):
        return "test"
    raise ValueError(f"cannot infer MELD split from CSV file name: {name}")


def _video_index(meld_root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    fallback: dict[tuple[str, str], Path] = {}
    for path in sorted(meld_root.rglob("*.mp4")):
        stem = path.stem
        split = _split_tag_for_path(path)
        if split is not None:
            index.setdefault((split, stem), path)
        fallback.setdefault(("", stem), path)
    index.update({key: value for key, value in fallback.items() if key not in index})
    return index


def _split_tag_for_path(path: Path) -> str | None:
    lowered = "/".join(part.lower() for part in path.parts)
    if "train" in lowered:
        return "train"
    if "dev" in lowered or "val" in lowered:
        return "dev"
    if "test" in lowered:
        return "test"
    return None


def _sample_spec(
    source_id: str,
    records_by_source_id: dict[str, dict[str, Any]],
    csv_text_by_source_id: dict[str, str],
    video_index: dict[tuple[str, str], Path],
) -> dict[str, Any]:
    record = records_by_source_id.get(source_id)
    if record is None:
        return {"source_id": source_id, "text": "", "video_path": None}
    dialogue_id = str(record.get("dialogue_id", "")).strip()
    utterance_id = str(record.get("utterance_id", "")).strip()
    original_split = str(record.get("original_split", "")).strip()
    split = str(record.get("split", "")).strip()
    stem = f"dia{dialogue_id}_utt{utterance_id}"
    video_path = (
        video_index.get((original_split, stem))
        or video_index.get((split, stem))
        or video_index.get(("", stem))
    )
    return {
        "source_id": source_id,
        "split": split,
        "original_split": original_split,
        "dialogue_id": dialogue_id,
        "utterance_id": utterance_id,
        "text": csv_text_by_source_id.get(source_id) or str(record.get("utterance_text", "")),
        "video_path": video_path,
    }


def _hash_text_features(text: str, dim: int) -> np.ndarray:
    vector = np.zeros((dim,), dtype=np.float32)
    tokens = TEXT_TOKEN_RE.findall(text.lower())
    for position, token in enumerate(tokens):
        for feature_token in (token, f"{position % 17}:{token}"):
            digest = hashlib.blake2b(feature_token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little", signed=False)
            sign = 1.0 if value & (1 << 63) == 0 else -1.0
            vector[value % dim] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _iter_video_results(tasks: list[dict[str, Any]], workers: int):
    if workers == 1:
        for task in tasks:
            yield _extract_video_features(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_extract_video_features, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def _extract_video_features(task: dict[str, Any]) -> dict[str, Any]:
    video_path = str(task["video_path"])
    if not video_path:
        return {
            "index": task["index"],
            "source_id": task["source_id"],
            "audio": np.zeros((int(task["audio_steps"]), AUDIO_FEATURE_DIM), dtype=np.float32),
            "visual": np.zeros(
                (int(task["visual_frames"]), int(task["visual_size"]) * int(task["visual_size"]) * 3),
                dtype=np.float32,
            ),
            "audio_missing": True,
            "visual_missing": True,
            "errors": ["missing_video"],
        }
    errors = []
    audio, audio_missing, audio_error = _ffmpeg_audio_features(task, video_path)
    visual, visual_missing, visual_error = _ffmpeg_visual_features(task, video_path)
    if audio_error:
        errors.append(audio_error)
    if visual_error:
        errors.append(visual_error)
    return {
        "index": task["index"],
        "source_id": task["source_id"],
        "audio": audio,
        "visual": visual,
        "audio_missing": audio_missing,
        "visual_missing": visual_missing,
        "errors": errors,
    }


def _ffmpeg_audio_features(task: dict[str, Any], video_path: str) -> tuple[np.ndarray, bool, str | None]:
    steps = int(task["audio_steps"])
    sample_rate = int(task["audio_sample_rate"])
    command = [
        str(task["ffmpeg_bin"]),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = _run_ffmpeg(command, float(task["timeout_sec"]))
    if result.returncode != 0 or not result.stdout:
        return np.zeros((steps, AUDIO_FEATURE_DIM), dtype=np.float32), True, "audio_decode_failed"
    waveform = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if waveform.size == 0 or not np.isfinite(waveform).all():
        return np.zeros((steps, AUDIO_FEATURE_DIM), dtype=np.float32), True, "audio_decode_empty"
    return _audio_segment_features(waveform, steps, sample_rate), False, None


def _audio_segment_features(waveform: np.ndarray, steps: int, sample_rate: int) -> np.ndarray:
    features = np.zeros((steps, AUDIO_FEATURE_DIM), dtype=np.float32)
    boundaries = np.linspace(0, waveform.size, steps + 1, dtype=np.int64)
    for index in range(steps):
        segment = waveform[boundaries[index] : boundaries[index + 1]]
        if segment.size == 0:
            continue
        abs_segment = np.abs(segment)
        rms = float(np.sqrt(np.mean(np.square(segment))))
        zero_crossing = float(np.mean(segment[:-1] * segment[1:] < 0)) if segment.size > 1 else 0.0
        quantiles = np.quantile(segment, [0.25, 0.5, 0.75]).astype(np.float32)
        features[index] = np.asarray(
            [
                float(np.mean(segment)),
                float(np.std(segment)),
                rms,
                float(np.max(segment)),
                float(np.min(segment)),
                float(quantiles[0]),
                float(quantiles[1]),
                float(quantiles[2]),
                float(np.mean(abs_segment)),
                float(np.log1p(np.mean(np.square(segment)))),
                zero_crossing,
                float(segment.size / sample_rate),
            ],
            dtype=np.float32,
        )
    return features


def _ffmpeg_visual_features(task: dict[str, Any], video_path: str) -> tuple[np.ndarray, bool, str | None]:
    frame_count = int(task["visual_frames"])
    size = int(task["visual_size"])
    frame_dim = size * size * 3
    command = [
        str(task["ffmpeg_bin"]),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        video_path,
        "-vf",
        f"fps={float(task['visual_fps'])},scale={size}:{size}",
        "-frames:v",
        str(frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = _run_ffmpeg(command, float(task["timeout_sec"]))
    if result.returncode != 0 or not result.stdout:
        return np.zeros((frame_count, frame_dim), dtype=np.float32), True, "visual_decode_failed"
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    usable = (raw.size // frame_dim) * frame_dim
    if usable == 0:
        return np.zeros((frame_count, frame_dim), dtype=np.float32), True, "visual_decode_empty"
    decoded = raw[:usable].reshape(-1, frame_dim).astype(np.float32) / 255.0
    visual = np.zeros((frame_count, frame_dim), dtype=np.float32)
    take = min(frame_count, decoded.shape[0])
    visual[:take, :] = decoded[:take, :]
    return visual, False, None


def _run_ffmpeg(command: list[str], timeout_sec: float) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
        check=False,
    )


def _write_feature_versions(path: Path, args: argparse.Namespace) -> None:
    existing = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {}
    versions = {
        **existing,
        "text": f"meld-hashed-text-v0.1:dim={args.text_dim}",
        "audio": (
            f"meld-ffmpeg-audio-stats-v0.1:steps={args.audio_steps}:"
            f"sr={args.audio_sample_rate}:dim={AUDIO_FEATURE_DIM}"
        ),
        "visual": (
            f"meld-ffmpeg-rgb-grid-v0.1:frames={args.visual_frames}:"
            f"size={args.visual_size}:fps={args.visual_fps}"
        ),
    }
    path.write_text(json.dumps(versions, sort_keys=True) + "\n")


def _manifest(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    failures: list[dict[str, str]],
    video_coverage: float,
    visual_dim: int,
) -> dict[str, Any]:
    return {
        "dataset_name": "meld",
        "meld_root": str(args.meld_root),
        "raw_root": str(args.raw_root),
        "sample_count": len(samples),
        "video_coverage": video_coverage,
        "missing_video_count": sum(1 for sample in samples if not sample["video_path"]),
        "failure_count": len(failures),
        "failure_preview": failures[:20],
        "feature_shapes": {
            "text": [len(samples), 1, args.text_dim],
            "audio": [len(samples), args.audio_steps, AUDIO_FEATURE_DIM],
            "visual": [len(samples), args.visual_frames, visual_dim],
        },
        "feature_extractor_versions": {
            "text": f"meld-hashed-text-v0.1:dim={args.text_dim}",
            "audio": f"meld-ffmpeg-audio-stats-v0.1:steps={args.audio_steps}:sr={args.audio_sample_rate}",
            "visual": f"meld-ffmpeg-rgb-grid-v0.1:frames={args.visual_frames}:size={args.visual_size}",
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
