#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build source_id to expression JSONL from UNC RefCOCO refs annotations."
    )
    parser.add_argument("--dataset-name", default="refcoco")
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    payload = build_refcoco_expression_map(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_refcoco_expression_map(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for ref in _read_refs(args.refs):
        image_id = _required_int(ref, "image_id")
        ann_id = _required_int(ref, "ann_id")
        for sentence in _sentences(ref):
            sent_id = _required_int(sentence, "sent_id")
            expression = _expression(sentence)
            rows.append(
                {
                    "source_id": f"{args.dataset_name}::image{image_id}::ann{ann_id}::sent{sent_id}",
                    "image_id": f"image{image_id}",
                    "ann_id": ann_id,
                    "sent_id": sent_id,
                    "expression": expression,
                }
            )
    if not rows:
        raise ValueError("refs produced no expressions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return {
        "ok": True,
        "dataset_name": args.dataset_name,
        "refs": str(args.refs),
        "output": str(args.output),
        "expression_count": len(rows),
    }


def _read_refs(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".p", ".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        payload = json.loads(path.read_text())
    refs = payload.get("refs") if isinstance(payload, dict) else payload
    if not isinstance(refs, list):
        raise ValueError("refs must be a list or an object with refs")
    return [ref for ref in refs if isinstance(ref, dict)]


def _sentences(ref: dict[str, Any]) -> list[dict[str, Any]]:
    sentences = ref.get("sentences")
    if isinstance(sentences, list):
        return [sentence for sentence in sentences if isinstance(sentence, dict)]
    return []


def _expression(sentence: dict[str, Any]) -> str:
    for key in ("raw", "sent", "sentence", "expression"):
        value = sentence.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    tokens = sentence.get("tokens")
    if isinstance(tokens, list) and tokens:
        expression = " ".join(str(token) for token in tokens if str(token).strip())
        if expression.strip():
            return expression.strip()
    raise ValueError(f"sentence {sentence.get('sent_id')} missing expression text")


def _required_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"ref/sentence missing integer {key}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
