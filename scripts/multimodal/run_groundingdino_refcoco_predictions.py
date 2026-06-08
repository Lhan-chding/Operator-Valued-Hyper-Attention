#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO over RefCOCO samples and emit prediction JSONL for same-candidate scoring."
    )
    parser.add_argument("--groundingdino-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--dataset-name", default="refcoco")
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--split", required=True)
    parser.add_argument("--expressions-jsonl", type=Path, required=True, help="JSONL rows with source_id and expression/text.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-template", default="COCO_train2014_{image_number:012d}.jpg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    payload = run_groundingdino_refcoco_predictions(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_groundingdino_refcoco_predictions(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(args.groundingdino_root))
    import torch

    from groundingdino.util.inference import load_image, load_model, predict

    layout = MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version)
    sample_records = _sample_records(layout.root, args.split)
    expressions = _load_expressions(args.expressions_jsonl)
    selected_records = sample_records[: args.limit] if args.limit is not None else sample_records
    missing_expressions = [record["source_id"] for record in selected_records if record["source_id"] not in expressions]
    if missing_expressions:
        raise ValueError("missing expressions for source_id values: " + ", ".join(missing_expressions[:20]))
    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    model = load_model(str(args.config), str(args.checkpoint), device=device)
    rows = []
    for record in selected_records:
        source_id = record["source_id"]
        image_path = _image_path_from_record(record, image_root=args.image_root, image_template=args.image_template)
        image_source, image = load_image(str(image_path))
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=_normalize_caption(expressions[source_id]),
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=device,
        )
        rows.append(
            {
                "source_id": source_id,
                "split": args.split,
                "image_path": str(image_path),
                "expression": expressions[source_id],
                "boxes": _boxes_cxcywh_to_xyxy_list(boxes),
                "scores": _tensor_to_float_list(logits),
                "phrases": [str(phrase) for phrase in phrases],
                "prediction_box_format": "xyxy_normalized",
                "image_shape": list(getattr(image_source, "shape", [])),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))
    return {
        "ok": True,
        "split": args.split,
        "sample_count": len(selected_records),
        "output": str(args.output),
        "device": device,
        "prediction_box_format": "xyxy_normalized",
        "next": (
            "python scripts/multimodal/score_groundingdino_same_candidates.py "
            f"--predictions {args.output} --split {args.split} --output-dir {args.output.parent / 'same_candidate_scores'}"
        ),
    }


def _sample_records(root: Path, split: str) -> list[dict[str, Any]]:
    path = root / "provenance" / f"sample_records_{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} contains no sample records")
    return rows


def _load_expressions(path: Path) -> dict[str, str]:
    expressions: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        source_id = row.get("source_id")
        expression = row.get("expression", row.get("text", row.get("sentence")))
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{path} line {line_number} missing source_id")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"{path} line {line_number} missing expression/text")
        if source_id in expressions:
            raise ValueError(f"duplicate expression source_id: {source_id}")
        expressions[source_id] = expression.strip()
    return expressions


def _image_path_from_record(record: dict[str, Any], *, image_root: Path, image_template: str) -> Path:
    if isinstance(record.get("image_path"), str) and record["image_path"].strip():
        return Path(record["image_path"])
    image_id = str(record.get("image_id", "")).strip()
    if not image_id:
        raise ValueError(f"sample record missing image_id for source_id: {record.get('source_id')}")
    digits = "".join(character for character in image_id if character.isdigit())
    if not digits:
        raise ValueError(f"image_id must include numeric COCO id: {image_id}")
    image_number = int(digits)
    return image_root / image_template.format(image_id=image_id, image_number=image_number)


def _normalize_caption(expression: str) -> str:
    caption = " ".join(expression.strip().lower().split())
    if not caption:
        raise ValueError("expression must be non-empty")
    return caption if caption.endswith(".") else f"{caption} ."


def _boxes_cxcywh_to_xyxy_list(boxes: Any) -> list[list[float]]:
    import torch

    tensor = boxes.detach().cpu() if hasattr(boxes, "detach") else torch.as_tensor(boxes)
    if tensor.numel() == 0:
        return []
    tensor = tensor.reshape(-1, 4).float()
    cx, cy, width, height = tensor[:, 0], tensor[:, 1], tensor[:, 2].clamp_min(0), tensor[:, 3].clamp_min(0)
    converted = torch.stack((cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height), dim=1)
    return converted.clamp(0, 1).tolist()


def _tensor_to_float_list(values: Any) -> list[float]:
    import torch

    tensor = values.detach().cpu() if hasattr(values, "detach") else torch.as_tensor(values)
    return [float(value) for value in tensor.reshape(-1).tolist()]


if __name__ == "__main__":
    raise SystemExit(main())
