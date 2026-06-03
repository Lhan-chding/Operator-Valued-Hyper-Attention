#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from moat_ovha_torch.eval.mosei_standard_metrics import mosei_standard_metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute CMU-MOSEI/MOSI sentiment metrics with the project standard "
            "MULT-compatible evaluator from external prediction/truth artifacts."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truths", type=Path, required=True)
    parser.add_argument("--prediction-key", default=None)
    parser.add_argument("--truth-key", default=None)
    parser.add_argument("--model", default="external_model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", default="cmu_mosei")
    parser.add_argument("--source", default="external_reproduction")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()

    try:
        payload = recompute_metrics(args)
        exit_code = 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        payload = {
            "ok": False,
            "mode": "mosei_standard_metric_recompute",
            "errors": [str(exc)],
            "warnings": [],
        }
        exit_code = 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def recompute_metrics(args: argparse.Namespace) -> dict[str, Any]:
    prediction = _load_tensor(args.predictions, key=args.prediction_key)
    truth = _load_tensor(args.truths, key=args.truth_key)
    mask = torch.ones_like(_mask_shape_reference(truth), dtype=torch.bool)
    metrics = mosei_standard_metrics(prediction, truth, mask)
    row = {
        "artifact_type": "external_mosei_standard_metric",
        "evidence_scope": "external_reproduction",
        "dataset": args.dataset,
        "task": "sentiment_emotion",
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "source": args.source,
        "metric_protocol": "mosei_standard_mult_compatible_exclude_zero",
        "metric_name": "mae",
        "score": metrics["mae"],
        "higher_is_better": False,
        "metrics": metrics,
        "prediction_artifact": str(args.predictions),
        "truth_artifact": str(args.truths),
    }
    payload = {
        "ok": True,
        "mode": "mosei_standard_metric_recompute",
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "dataset": args.dataset,
        "metrics": metrics,
        "row": row,
        "warnings": [],
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return payload


def _mask_shape_reference(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 1 and int(tensor.shape[-1]) == 1:
        return tensor.squeeze(-1)
    return tensor


def _load_tensor(path: Path, *, key: str | None) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return torch.as_tensor(_numpy().load(path))
    if suffix == ".npz":
        payload = _numpy().load(path)
        selected_key = key or _single_key(payload.files, path)
        return torch.as_tensor(payload[selected_key])
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        return _tensor_from_payload(payload, key=key, path=path)
    if suffix == ".json":
        return _tensor_from_payload(json.loads(path.read_text()), key=key, path=path)
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return _tensor_from_payload(rows, key=key, path=path)
    raise ValueError(f"unsupported tensor artifact extension for {path}; expected .npy/.npz/.pt/.pth/.json/.jsonl")


def _tensor_from_payload(payload: Any, *, key: str | None, path: Path) -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload.detach().cpu()
    if isinstance(payload, dict):
        selected_key = key or _first_present_key(payload, ("predictions", "prediction", "preds", "truths", "truth", "targets", "target", "labels"))
        if selected_key is None:
            raise ValueError(f"{path} is a mapping; pass --prediction-key/--truth-key")
        return _tensor_from_payload(payload[selected_key], key=None, path=path)
    return torch.as_tensor(payload)


def _single_key(keys: list[str], path: Path) -> str:
    if len(keys) != 1:
        raise ValueError(f"{path} contains multiple arrays; pass --prediction-key/--truth-key")
    return keys[0]


def _first_present_key(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in payload:
            return key
    return None


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to read .npy/.npz artifacts") from exc
    return np


if __name__ == "__main__":
    raise SystemExit(main())
