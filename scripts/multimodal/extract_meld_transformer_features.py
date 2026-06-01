#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


SPLIT_ORDER = ("train", "val", "test")
MODALITIES = ("text", "audio", "vision")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract formal frozen MELD features with transformer encoders: "
            "RoBERTa-style text, Wav2Vec2-style speech, and CLIP-style video frame features."
        )
    )
    parser.add_argument("meld_root", type=Path)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--text-model", default="FacebookAI/roberta-base")
    parser.add_argument("--audio-model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--text-revision", default="main")
    parser.add_argument("--audio-revision", default="main")
    parser.add_argument("--vision-revision", default="main")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument("--text-max-length", type=int, default=128)
    parser.add_argument("--text-tokens", type=int, default=16)
    parser.add_argument("--audio-steps", type=int, default=32)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--visual-frames", type=int, default=8)
    parser.add_argument("--visual-fps", type=float, default=2.0)
    parser.add_argument("--visual-size", type=int, default=224)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--timeout-sec", type=float, default=90.0)
    parser.add_argument("--min-video-coverage", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args()

    try:
        payload = extract_meld_transformer_features(args)
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


def extract_meld_transformer_features(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    raw_root = args.raw_root
    metadata_dir = raw_root / "metadata"
    feature_dir = raw_root / "features"
    splits = _read_splits(raw_root / "splits.json")
    ordered_source_ids = _ordered_source_ids(splits)
    records_by_source_id = _read_records(metadata_dir / "dialogues.json")
    csv_text_by_source_id = _read_meld_csv_text(args.meld_root)
    video_index = _video_index(args.meld_root)
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
            "extract nested train/dev/test tarballs first."
        )
    if any(sample["video_path"] for sample in samples) and shutil.which(args.ffmpeg_bin) is None:
        raise ValueError(f"ffmpeg executable not found: {args.ffmpeg_bin}; install ffmpeg before feature extraction")
    if args.dry_run:
        return _dry_run_payload(args, samples, video_coverage)

    torch, transformers = _load_feature_dependencies()
    device = _resolve_device(torch, args.device)
    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32
    feature_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    text_features = _extract_text_features(args, samples, feature_dir / "text_features.npy", torch, transformers, device, dtype)
    audio_features, audio_missing = _extract_audio_features(
        args,
        samples,
        feature_dir / "audio_features.npy",
        torch,
        transformers,
        device,
        dtype,
    )
    visual_features, visual_missing = _extract_visual_features(
        args,
        samples,
        feature_dir / "visual_features.npy",
        torch,
        transformers,
        device,
        dtype,
    )
    missing_mask = np.zeros((len(samples), len(MODALITIES)), dtype=bool)
    missing_mask[:, 0] = [not str(sample["text"]).strip() for sample in samples]
    missing_mask[:, 1] = audio_missing
    missing_mask[:, 2] = visual_missing
    np.save(metadata_dir / "missing_modality_mask.npy", missing_mask)
    _write_feature_versions(metadata_dir / "feature_versions.json", args)
    manifest = _manifest(
        args,
        samples,
        video_coverage,
        text_shape=list(text_features.shape),
        audio_shape=list(audio_features.shape),
        visual_shape=list(visual_features.shape),
        audio_missing_count=int(np.sum(audio_missing)),
        visual_missing_count=int(np.sum(visual_missing)),
    )
    manifest_path = args.output_manifest or metadata_dir / "meld_transformer_feature_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "ok": True,
        **manifest,
        "next": "python scripts/multimodal/build_cache.py meld data/raw_multimodal/meld data/multimodal_cache --version v0.1",
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "text_batch_size": args.text_batch_size,
        "vision_batch_size": args.vision_batch_size,
        "text_max_length": args.text_max_length,
        "text_tokens": args.text_tokens,
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


def _load_feature_dependencies():
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "formal MELD transformer feature extraction requires torch and transformers in the active .venv"
        ) from exc
    return torch, transformers


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but torch.cuda.is_available() is false")
    return torch.device(requested)


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
        split = _split_tag_for_path(path)
        index.setdefault((split or "", path.stem), path)
        fallback.setdefault(("", path.stem), path)
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
    split = str(record.get("split", "")).strip()
    original_split = str(record.get("original_split", "")).strip()
    stem = f"dia{dialogue_id}_utt{utterance_id}"
    return {
        "source_id": source_id,
        "split": split,
        "original_split": original_split,
        "dialogue_id": dialogue_id,
        "utterance_id": utterance_id,
        "text": csv_text_by_source_id.get(source_id) or str(record.get("utterance_text", "")),
        "video_path": video_index.get((original_split, stem))
        or video_index.get((split, stem))
        or video_index.get(("", stem)),
    }


def _extract_text_features(args, samples, destination: Path, torch, transformers, device, dtype):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_model, revision=args.text_revision)
    model = transformers.AutoModel.from_pretrained(args.text_model, revision=args.text_revision)
    model.to(device=device, dtype=dtype)
    model.eval()
    output = None
    texts = [str(sample["text"]) for sample in samples]
    with torch.inference_mode():
        for start in range(0, len(texts), args.text_batch_size):
            batch_texts = texts[start : start + args.text_batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.text_max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state.detach().float().cpu()
            mask = inputs["attention_mask"].detach().cpu().bool()
            pooled = [_pool_valid_tokens(hidden[index, mask[index]], args.text_tokens) for index in range(hidden.shape[0])]
            batch = np.stack(pooled).astype(np.float32)
            if output is None:
                output = np.lib.format.open_memmap(
                    destination,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(samples), args.text_tokens, batch.shape[-1]),
                )
            output[start : start + len(batch_texts), :, :] = batch
    if output is None:
        raise ValueError("no MELD samples available for text feature extraction")
    output.flush()
    return np.asarray(output)


def _pool_valid_tokens(valid_hidden, target_tokens: int) -> np.ndarray:
    import torch

    if valid_hidden.numel() == 0:
        return np.zeros((target_tokens, 1), dtype=np.float32)
    sequence = valid_hidden.transpose(0, 1).unsqueeze(0)
    pooled = torch.nn.functional.adaptive_avg_pool1d(sequence, target_tokens).squeeze(0).transpose(0, 1)
    return pooled.numpy()


def _extract_audio_features(args, samples, destination: Path, torch, transformers, device, dtype):
    processor = transformers.AutoProcessor.from_pretrained(args.audio_model, revision=args.audio_revision)
    model = transformers.AutoModel.from_pretrained(args.audio_model, revision=args.audio_revision)
    model.to(device=device, dtype=dtype)
    model.eval()
    output = None
    missing = np.zeros((len(samples),), dtype=bool)
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            video_path = sample["video_path"]
            if video_path is None:
                missing[index] = True
                continue
            waveform = _decode_audio(args, video_path)
            if waveform.size == 0:
                missing[index] = True
                continue
            inputs = processor(waveform, sampling_rate=args.audio_sample_rate, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state.detach().float().cpu()[0]
            pooled = _pool_valid_tokens(hidden, args.audio_steps).astype(np.float32)
            if output is None:
                output = np.lib.format.open_memmap(
                    destination,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(samples), args.audio_steps, pooled.shape[-1]),
                )
                output[:] = 0
            output[index, :, :] = pooled
    if output is None:
        output = np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=np.float32,
            shape=(len(samples), args.audio_steps, 1),
        )
        output[:] = 0
        missing[:] = True
    output.flush()
    return np.asarray(output), missing


def _decode_audio(args, video_path: Path) -> np.ndarray:
    command = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(args.audio_sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.timeout_sec, check=False)
    if result.returncode != 0 or not result.stdout:
        return np.empty((0,), dtype=np.float32)
    waveform = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return waveform if np.isfinite(waveform).all() else np.empty((0,), dtype=np.float32)


def _extract_visual_features(args, samples, destination: Path, torch, transformers, device, dtype):
    from PIL import Image

    processor = transformers.CLIPImageProcessor.from_pretrained(args.vision_model, revision=args.vision_revision)
    model = transformers.CLIPVisionModel.from_pretrained(args.vision_model, revision=args.vision_revision)
    model.to(device=device, dtype=dtype)
    model.eval()
    output = None
    missing = np.zeros((len(samples),), dtype=bool)
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            video_path = sample["video_path"]
            if video_path is None:
                missing[index] = True
                continue
            frames = _decode_frames(args, video_path)
            if not frames:
                missing[index] = True
                continue
            embeddings = []
            for start in range(0, len(frames), args.vision_batch_size):
                batch_frames = frames[start : start + args.vision_batch_size]
                images = [Image.fromarray(frame, mode="RGB") for frame in batch_frames]
                inputs = processor(images=images, return_tensors="pt")
                inputs = {key: value.to(device=device, dtype=dtype) for key, value in inputs.items()}
                batch = model(**inputs).pooler_output.detach().float().cpu().numpy()
                embeddings.append(batch)
            frame_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
            padded = np.zeros((args.visual_frames, frame_embeddings.shape[-1]), dtype=np.float32)
            take = min(args.visual_frames, frame_embeddings.shape[0])
            padded[:take, :] = frame_embeddings[:take, :]
            if output is None:
                output = np.lib.format.open_memmap(
                    destination,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(samples), args.visual_frames, padded.shape[-1]),
                )
                output[:] = 0
            output[index, :, :] = padded
    if output is None:
        output = np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=np.float32,
            shape=(len(samples), args.visual_frames, 1),
        )
        output[:] = 0
        missing[:] = True
    output.flush()
    return np.asarray(output), missing


def _decode_frames(args, video_path: Path) -> list[np.ndarray]:
    frame_dim = args.visual_size * args.visual_size * 3
    command = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-vf",
        f"fps={args.visual_fps},scale={args.visual_size}:{args.visual_size}",
        "-frames:v",
        str(args.visual_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.timeout_sec, check=False)
    if result.returncode != 0 or not result.stdout:
        return []
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    usable = (raw.size // frame_dim) * frame_dim
    if usable == 0:
        return []
    return list(raw[:usable].reshape(-1, args.visual_size, args.visual_size, 3))


def _write_feature_versions(path: Path, args: argparse.Namespace) -> None:
    existing = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                existing = payload
        except json.JSONDecodeError:
            existing = {}
    versions = {
        **existing,
        "text": f"{args.text_model}@{args.text_revision}:tokens={args.text_tokens}:max_length={args.text_max_length}",
        "audio": f"{args.audio_model}@{args.audio_revision}:steps={args.audio_steps}:sr={args.audio_sample_rate}",
        "visual": f"{args.vision_model}@{args.vision_revision}:frames={args.visual_frames}:fps={args.visual_fps}",
    }
    path.write_text(json.dumps(versions, sort_keys=True) + "\n")


def _dry_run_payload(args: argparse.Namespace, samples: list[dict[str, Any]], video_coverage: float) -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "dry_run",
        "dataset_name": "meld",
        "sample_count": len(samples),
        "video_coverage": video_coverage,
        "missing_video_count": sum(1 for sample in samples if not sample["video_path"]),
        "feature_extractor_versions": _feature_version_payload(args),
    }


def _manifest(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    video_coverage: float,
    *,
    text_shape: list[int],
    audio_shape: list[int],
    visual_shape: list[int],
    audio_missing_count: int,
    visual_missing_count: int,
) -> dict[str, Any]:
    return {
        "dataset_name": "meld",
        "meld_root": str(args.meld_root),
        "raw_root": str(args.raw_root),
        "sample_count": len(samples),
        "video_coverage": video_coverage,
        "audio_missing_count": audio_missing_count,
        "visual_missing_count": visual_missing_count,
        "feature_shapes": {
            "text": text_shape,
            "audio": audio_shape,
            "visual": visual_shape,
        },
        "feature_extractor_versions": _feature_version_payload(args),
    }


def _feature_version_payload(args: argparse.Namespace) -> dict[str, str]:
    return {
        "text": f"{args.text_model}@{args.text_revision}",
        "audio": f"{args.audio_model}@{args.audio_revision}",
        "visual": f"{args.vision_model}@{args.vision_revision}",
    }


if __name__ == "__main__":
    raise SystemExit(main())
