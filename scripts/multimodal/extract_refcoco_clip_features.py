#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np


SPLIT_ORDER = ("train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract formal frozen RefCOCO phrase/region features with CLIP. "
            "The output arrays are written in the exact source_id order required "
            "by stage_refcoco_raw.py."
        )
    )
    parser.add_argument("dataset_name", choices=("refcoco", "refcoco_plus", "refcocog"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, action="append", required=True)
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--max-text-length", type=int, default=77)
    parser.add_argument("--min-image-coverage", type=float, default=1.0)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-manifest", type=Path)
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    try:
        payload = extract_refcoco_clip_features(args)
    except (OSError, ValueError, json.JSONDecodeError, pickle.UnpicklingError) as exc:
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


def extract_refcoco_clip_features(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    splits = _read_splits(args.splits)
    ordered_source_ids = _ordered_source_ids(splits)
    records_by_source_id = _read_records(args.records)
    texts_by_source_id = _read_ref_texts(args.refs, args.dataset_name)
    samples = _samples(ordered_source_ids, records_by_source_id, texts_by_source_id, args.image_root)
    missing_images = [sample["source_id"] for sample in samples if sample["image_path"] is None]
    image_coverage = 1.0 - (len(missing_images) / max(1, len(samples)))
    if image_coverage < args.min_image_coverage:
        raise ValueError(
            f"RefCOCO image coverage {image_coverage:.6f} is below --min-image-coverage "
            f"{args.min_image_coverage:.6f}; first image missing source_id={missing_images[0]}"
        )
    if args.dry_run:
        return _dry_run_payload(args, samples, image_coverage, missing_images)

    torch, transformers, image_module = _load_dependencies()
    device = _resolve_device(torch, args.device)
    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32
    model = transformers.CLIPModel.from_pretrained(args.model, revision=args.revision).to(device=device, dtype=dtype)
    model.eval()
    tokenizer = transformers.CLIPTokenizer.from_pretrained(args.model, revision=args.revision)
    image_processor = transformers.CLIPImageProcessor.from_pretrained(args.model, revision=args.revision)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    text_features, text_mask = _extract_text_features(args, samples, tokenizer, model, torch, device)
    region_features, region_mask, region_missing = _extract_region_features(args, samples, image_processor, model, torch, image_module, device)
    text_path = output_dir / f"{args.dataset_name}_text_features.npy"
    region_path = output_dir / f"{args.dataset_name}_region_features.npy"
    text_mask_path = output_dir / f"{args.dataset_name}_text_mask.npy"
    region_mask_path = output_dir / f"{args.dataset_name}_region_mask.npy"
    np.save(text_path, text_features)
    np.save(region_path, region_features)
    np.save(text_mask_path, text_mask)
    np.save(region_mask_path, region_mask)
    failed_path = output_dir / f"{args.dataset_name}_failed_samples.jsonl"
    if any(region_missing):
        failed_path.write_text(
            "".join(
                json.dumps({"source_id": sample["source_id"], "reason": "missing_or_invalid_region_image"}, sort_keys=True) + "\n"
                for sample, missing in zip(samples, region_missing)
                if missing
            )
        )
    manifest = _manifest(
        args,
        samples,
        image_coverage,
        missing_images,
        text_shape=list(text_features.shape),
        region_shape=list(region_features.shape),
    )
    manifest_path = args.output_manifest or output_dir / f"{args.dataset_name}_clip_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "ok": True,
        **manifest,
        "outputs": {
            "text_features": str(text_path),
            "region_features": str(region_path),
            "text_mask": str(text_mask_path),
            "region_mask": str(region_mask_path),
            "failed_samples": str(failed_path) if failed_path.exists() else None,
        },
        "next": (
            f"python scripts/multimodal/stage_refcoco_raw.py {args.dataset_name} "
            f"data/raw_multimodal/{args.dataset_name} "
            f"--splits {args.splits} --records {args.records} "
            f"--text-features {text_path} --region-features {region_path} "
            f"--text-mask {text_mask_path} --region-mask {region_mask_path} "
            f"--license-tag refcoco-coco2014 --preprocessing-version {args.dataset_name}-clip-vit-large-p14-v0.1"
        ),
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "text_batch_size": args.text_batch_size,
        "region_batch_size": args.region_batch_size,
        "max_text_length": args.max_text_length,
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    if not (0 <= args.min_image_coverage <= 1):
        raise ValueError("--min-image-coverage must be in [0, 1]")


def _load_dependencies():
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError("formal RefCOCO CLIP feature extraction requires torch, transformers, and pillow") from exc
    return torch, transformers, Image


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _read_splits(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError("splits must be a non-empty JSON object keyed by split")
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
        raise ValueError("records must contain a non-empty list")
    by_source_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be an object")
        source_id = _normalized_source_id(record.get("source_id"), f"records[{index}]")
        if source_id in by_source_id:
            raise ValueError(f"duplicate record source_id: {source_id}")
        by_source_id[source_id] = record
    return by_source_id


def _read_ref_texts(path: Path, dataset_name: str) -> dict[str, str]:
    payload = _read_pickle_or_json(path)
    refs = payload.get("refs") if isinstance(payload, dict) else payload
    if not isinstance(refs, list) or not refs:
        raise ValueError("refs must contain a non-empty list")
    texts: dict[str, str] = {}
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        ann_id = _int_value(ref.get("ann_id"), field="ref ann_id")
        image_id = _int_value(ref.get("image_id"), field=f"ref ann_id {ann_id} image_id")
        for sentence in _sentences(ref):
            sent_id = _int_value(sentence.get("sent_id"), field=f"ref ann_id {ann_id} sentence sent_id")
            text = _sentence_text(sentence)
            source_id = f"{dataset_name}::image{image_id}::ann{ann_id}::sent{sent_id}"
            texts[source_id] = text
    return texts


def _read_pickle_or_json(path: Path) -> Any:
    if path.suffix.lower() in {".p", ".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    return json.loads(path.read_text())


def _sentences(ref: dict[str, Any]) -> list[dict[str, Any]]:
    sentences = ref.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError(f"ref {ref.get('ref_id', ref.get('ann_id', '?'))} missing sentences")
    return [sentence for sentence in sentences if isinstance(sentence, dict)]


def _sentence_text(sentence: dict[str, Any]) -> str:
    raw = sentence.get("raw") or sentence.get("sent") or sentence.get("sentence")
    if isinstance(raw, str) and raw.strip():
        return " ".join(raw.split())
    tokens = sentence.get("tokens")
    if isinstance(tokens, list) and tokens:
        text = " ".join(str(token) for token in tokens).strip()
        if text:
            return text
    raise ValueError(f"sentence {sentence.get('sent_id', '?')} has no text")


def _samples(
    ordered_source_ids: list[str],
    records_by_source_id: dict[str, dict[str, Any]],
    texts_by_source_id: dict[str, str],
    image_roots: list[Path],
) -> list[dict[str, Any]]:
    samples = []
    for index, source_id in enumerate(ordered_source_ids):
        record = records_by_source_id.get(source_id)
        if record is None:
            raise ValueError(f"records missing source_id from splits: {source_id}")
        text = texts_by_source_id.get(source_id) or _record_text(record)
        if not text:
            raise ValueError(f"refs missing expression text for source_id: {source_id}")
        image_number = _image_number(record.get("image_id"), source_id)
        samples.append(
            {
                "index": index,
                "source_id": source_id,
                "text": text,
                "image_id": str(record.get("image_id")),
                "image_number": image_number,
                "image_path": _find_image(image_roots, image_number),
                "region_box": _region_box(record.get("region_box"), source_id),
                "candidate_region_boxes": _candidate_region_boxes(record, source_id),
            }
        )
    return samples


def _record_text(record: dict[str, Any]) -> str | None:
    for key in ("text", "phrase", "sentence", "raw_sentence", "referring_expression"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return None


def _image_number(value: Any, source_id: str) -> int:
    if isinstance(value, str) and value.startswith("image"):
        suffix = value[len("image") :]
        if suffix.isdigit():
            return int(suffix)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"record {source_id} image_id must be image<int>")


def _find_image(image_roots: list[Path], image_number: int) -> Path | None:
    names = (
        f"COCO_train2014_{image_number:012d}.jpg",
        f"COCO_val2014_{image_number:012d}.jpg",
        f"{image_number:012d}.jpg",
        f"{image_number}.jpg",
    )
    for root in image_roots:
        for name in names:
            path = root / name
            if path.exists():
                return path
    return None


def _region_box(value: Any, source_id: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"record {source_id} region_box must contain four coordinates")
    try:
        x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {source_id} region_box must contain numeric coordinates") from exc
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"record {source_id} region_box must be normalized xyxy with 0 <= x1 < x2 <= 1")
    return x1, y1, x2, y2


def _candidate_region_boxes(record: dict[str, Any], source_id: str) -> list[tuple[float, float, float, float]]:
    raw_boxes = record.get("candidate_region_boxes")
    if raw_boxes is None:
        return [_region_box(record.get("region_box"), source_id)]
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError(f"record {source_id} candidate_region_boxes must be a non-empty list when provided")
    boxes = [_region_box(box, source_id) for box in raw_boxes]
    target_region_index = record.get("target_region_index", 0)
    if not isinstance(target_region_index, int) or isinstance(target_region_index, bool):
        raise ValueError(f"record {source_id} target_region_index must be an integer")
    if target_region_index < 0 or target_region_index >= len(boxes):
        raise ValueError(f"record {source_id} target_region_index is outside candidate_region_boxes")
    return boxes


def _extract_text_features(args: argparse.Namespace, samples, tokenizer, model, torch, device) -> tuple[np.ndarray, np.ndarray]:
    outputs = []
    masks = []
    with torch.no_grad():
        for batch in _chunks(samples, args.text_batch_size):
            encoded = tokenizer(
                [sample["text"] for sample in batch],
                padding="max_length",
                truncation=True,
                max_length=args.max_text_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            features = model.text_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
            ).last_hidden_state
            features = _project_text_tokens(model, features)
            if args.normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
            features = features * attention_mask.unsqueeze(-1).to(dtype=features.dtype)
            outputs.append(features.detach().cpu().float().numpy())
            masks.append(attention_mask.detach().cpu().bool().numpy())
    return (
        np.concatenate(outputs, axis=0).astype(np.float32, copy=False),
        np.concatenate(masks, axis=0).astype(bool, copy=False),
    )


def _extract_region_features(args: argparse.Namespace, samples, image_processor, model, torch, image_module, device):
    max_regions = max(len(sample["candidate_region_boxes"]) for sample in samples)
    if max_regions <= 1:
        raise ValueError("RefCOCO records must provide more than one candidate region for non-trivial grounding")
    outputs = []
    masks = []
    missing = np.zeros((len(samples),), dtype=bool)
    with torch.no_grad():
        for batch in _chunks(samples, args.region_batch_size):
            images = []
            slots: list[tuple[int, int]] = []
            for sample in batch:
                for region_index, region_box in enumerate(sample["candidate_region_boxes"]):
                    crop = _crop_region(sample, region_box, image_module)
                    slots.append((int(sample["index"]), region_index))
                    if crop is None:
                        images.append(_blank_image(image_module))
                        missing[int(sample["index"])] = True
                    else:
                        images.append(_as_rgb_image(crop, image_module))
            encoded = image_processor(images=[_as_clip_image(image, image_module) for image in images], return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            features = model.get_image_features(**encoded)
            if args.normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            values = features.detach().cpu().float().numpy()
            batch_features = np.zeros((len(batch), max_regions, values.shape[-1]), dtype=np.float32)
            batch_mask = np.zeros((len(batch), max_regions), dtype=bool)
            sample_position_by_index = {int(sample["index"]): position for position, sample in enumerate(batch)}
            for value_index, (sample_index, region_index) in enumerate(slots):
                batch_position = sample_position_by_index[sample_index]
                batch_features[batch_position, region_index, :] = values[value_index]
                batch_mask[batch_position, region_index] = True
            outputs.append(batch_features)
            masks.append(batch_mask)
    return (
        np.concatenate(outputs, axis=0).astype(np.float32, copy=False),
        np.concatenate(masks, axis=0).astype(bool, copy=False),
        missing,
    )


def _project_text_tokens(model, features):
    projection = getattr(model, "text_projection", None)
    if projection is None:
        return features
    if callable(projection):
        return projection(features)
    return features @ projection


def _crop_region(sample: dict[str, Any], region_box: tuple[float, float, float, float], image_module):
    path = sample["image_path"]
    if path is None:
        return None
    try:
        with image_module.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            x1, y1, x2, y2 = region_box
            pixel_box = (
                max(0, min(width - 1, int(round(x1 * width)))),
                max(0, min(height - 1, int(round(y1 * height)))),
                max(1, min(width, int(round(x2 * width)))),
                max(1, min(height, int(round(y2 * height)))),
            )
            if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
                return None
            return _as_rgb_image(rgb.crop(pixel_box), image_module)
    except OSError:
        return None


def _blank_image(image_module):
    return image_module.new("RGB", (224, 224), color=(0, 0, 0))


def _as_rgb_image(image: Any, image_module):
    array = _as_rgb_array(image)
    return image_module.fromarray(array, mode="RGB")


def _as_clip_image(image: Any, image_module):
    rgb = _as_rgb_image(image, image_module)
    if rgb.size == (224, 224):
        return rgb
    resampling = getattr(image_module, "Resampling", image_module)
    return rgb.resize((224, 224), resample=resampling.BICUBIC)


def _as_rgb_array(image: Any) -> np.ndarray:
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    if array.size == 0 or array.shape[0] <= 0 or array.shape[1] <= 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return np.ascontiguousarray(array, dtype=np.uint8)


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _dry_run_payload(args: argparse.Namespace, samples, image_coverage: float, missing_images: list[str]) -> dict[str, Any]:
    max_regions = max((len(sample["candidate_region_boxes"]) for sample in samples), default=0)
    return {
        "ok": True,
        "mode": "dry_run",
        **_manifest(
            args,
            samples,
            image_coverage,
            missing_images,
            text_shape=[len(samples), f"<= {args.max_text_length}", "clip_projection_dim"],
            region_shape=[len(samples), max_regions, "clip_projection_dim"],
        ),
    }


def _manifest(
    args: argparse.Namespace,
    samples,
    image_coverage: float,
    missing_images: list[str],
    *,
    text_shape: list[Any],
    region_shape: list[Any],
) -> dict[str, Any]:
    return {
        "dataset_name": args.dataset_name,
        "sample_count": len(samples),
        "splits": str(args.splits),
        "records": str(args.records),
        "refs": str(args.refs),
        "image_roots": [str(path) for path in args.image_root],
        "image_coverage": image_coverage,
        "missing_image_count": len(missing_images),
        "feature_extractor_versions": {
            "text": f"{args.model}@{args.revision}:text_projection",
            "region": f"{args.model}@{args.revision}:image_projection",
        },
        "feature_shapes": {
            "text": text_shape,
            "region": region_shape,
        },
        "normalization": "l2" if args.normalize else "none",
        "candidate_region_source": "coco_gt_box_crop",
    }


def _normalized_source_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{context} source_id must be a non-empty normalized string")
    return value


def _int_value(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
