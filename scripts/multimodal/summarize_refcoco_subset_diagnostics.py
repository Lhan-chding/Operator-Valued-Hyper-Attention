#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
from scripts.multimodal.score_groundingdino_same_candidates import _load_source_ids, _safe_rate


SPATIAL_TERMS = frozenset(
    "above below left right top bottom front behind back near next beside between under over beside middle center".split()
)
ATTRIBUTE_TERMS = frozenset(
    "red blue green yellow black white brown gray grey pink orange purple small large big little tall short old young".split()
)
RELATIONAL_TERMS = frozenset(
    "near next beside between holding wearing riding sitting standing looking with on in under over by".split()
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize RefCOCO per-sample scores by diagnostic subsets.")
    parser.add_argument("--per-sample-scores", type=Path, required=True)
    parser.add_argument(
        "--input-format",
        choices=("auto", "score_rows", "public_main_predictions"),
        default="auto",
        help="score_rows use selected_iou/hit fields; public_main_predictions use run_public_main per-sample logits.",
    )
    parser.add_argument("--expressions-jsonl", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--dataset-name", default="refcoco")
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=20)
    payload = summarize_refcoco_subset_diagnostics(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def summarize_refcoco_subset_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    layout = MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version)
    root = layout.root
    source_ids = _load_source_ids(root, args.split)
    source_index = {source_id: index for index, source_id in enumerate(source_ids)}
    region_mask = np.load(root / "masks" / f"region_mask_{args.split}.npy").astype(bool, copy=False)
    candidate_boxes = np.load(root / "supervision" / f"candidate_region_boxes_{args.split}.npy").astype(np.float32, copy=False)
    bbox_targets = np.load(root / "supervision" / f"bbox_targets_{args.split}.npy").astype(np.float32, copy=False)
    region_targets = np.load(root / "supervision" / f"region_targets_{args.split}.npy").reshape(-1).astype(np.int64, copy=False)
    expressions = _load_expressions(args.expressions_jsonl)
    score_rows = _load_score_rows(
        args.per_sample_scores,
        input_format=getattr(args, "input_format", "auto"),
        source_index=source_index,
        candidate_boxes=candidate_boxes,
        bbox_targets=bbox_targets,
        region_targets=region_targets,
        region_mask=region_mask,
    )
    groups: dict[str, list[dict[str, Any]]] = {"all": []}
    for row in score_rows:
        source_id = row["source_id"]
        if source_id not in source_index:
            continue
        expression = expressions.get(source_id, "")
        tokens = _tokens(expression)
        valid_k = int(region_mask[source_index[source_id]].sum())
        _add(groups, "all", row)
        if int(row.get("prediction_count", 0)) == 0:
            _add(groups, "empty_prediction", row)
        if SPATIAL_TERMS & set(tokens):
            _add(groups, "spatial_expression", row)
        if ATTRIBUTE_TERMS & set(tokens):
            _add(groups, "attribute_expression", row)
        if RELATIONAL_TERMS & set(tokens):
            _add(groups, "relational_expression", row)
        _add(groups, _length_bucket(tokens), row)
        _add(groups, _candidate_bucket(valid_k), row)
    summaries = [
        summary
        for summary in (_summarize_group(name, rows) for name, rows in sorted(groups.items()))
        if summary["sample_count"] >= int(args.min_count) or summary["group"] == "all"
    ]
    payload = {
        "artifact_type": "refcoco_subset_diagnostics",
        "dataset": args.dataset_name,
        "split": args.split,
        "version": args.version,
        "per_sample_scores": str(args.per_sample_scores),
        "groups": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "subset_diagnostics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "subset_diagnostics.md").write_text(_markdown_table(summaries) + "\n")
    payload["output_dir"] = str(args.output_dir)
    payload["artifacts"] = {
        "json": str(args.output_dir / "subset_diagnostics.json"),
        "markdown": str(args.output_dir / "subset_diagnostics.md"),
    }
    return payload


def _load_score_rows(
    path: Path,
    *,
    input_format: str = "auto",
    source_index: dict[str, int] | None = None,
    candidate_boxes: np.ndarray | None = None,
    bbox_targets: np.ndarray | None = None,
    region_targets: np.ndarray | None = None,
    region_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} contains no score rows")
    detected = input_format
    if detected == "auto":
        detected = "public_main_predictions" if "prediction_full" in rows[0] else "score_rows"
    if detected == "score_rows":
        for index, row in enumerate(rows, start=1):
            if "source_id" not in row:
                raise ValueError(f"{path} line {index} missing source_id")
        return rows
    if detected != "public_main_predictions":
        raise ValueError(f"unknown input format: {input_format}")
    if source_index is None or candidate_boxes is None or bbox_targets is None or region_targets is None or region_mask is None:
        raise ValueError("public_main_predictions format requires cache supervision arrays")
    converted = []
    for line_number, row in enumerate(rows, start=1):
        source_id = row.get("source_id", row.get("sample_id"))
        if not isinstance(source_id, str) or source_id not in source_index:
            raise ValueError(f"{path} line {line_number} missing known source_id/sample_id")
        logits = _flatten_logits(row.get("prediction_full"))
        row_index = source_index[source_id]
        valid_mask = region_mask[row_index].reshape(-1).astype(bool)
        if logits.shape[0] < valid_mask.shape[0]:
            logits = np.pad(logits, (0, valid_mask.shape[0] - logits.shape[0]), constant_values=-1.0e9)
        logits = logits[: valid_mask.shape[0]]
        logits = np.where(valid_mask, logits, -1.0e9)
        selected_index = int(np.argmax(logits)) if bool(valid_mask.any()) else -1
        target_index = int(region_targets[row_index])
        selected_iou = 0.0 if selected_index < 0 else float(_box_iou(candidate_boxes[row_index, selected_index], bbox_targets[row_index]))
        ranked = [int(index) for index in np.argsort(-logits, kind="stable").tolist() if bool(valid_mask[index])]
        target_rank = ranked.index(target_index) + 1 if target_index in ranked else None
        converted.append(
            {
                "source_id": source_id,
                "split": row.get("split"),
                "seed": row.get("seed"),
                "model": row.get("model"),
                "selected_index": selected_index,
                "target_index": target_index,
                "selected_iou": selected_iou,
                "hit_at_1": selected_index == target_index,
                "hit_at_5": target_rank is not None and target_rank <= 5,
                "target_rank": target_rank,
                "prediction_count": int(valid_mask.sum()),
            }
        )
    return converted


def _flatten_logits(payload: Any) -> np.ndarray:
    array = np.asarray(payload, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("prediction_full must contain candidate logits")
    return array.reshape(-1)


def _box_iou(box: np.ndarray, target: np.ndarray) -> float:
    x1 = max(float(box[0]), float(target[0]))
    y1 = max(float(box[1]), float(target[1]))
    x2 = min(float(box[2]), float(target[2]))
    y2 = min(float(box[3]), float(target[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box_area = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
    target_area = max(0.0, float(target[2]) - float(target[0])) * max(0.0, float(target[3]) - float(target[1]))
    union = box_area + target_area - inter
    return 0.0 if union <= 0.0 else inter / union


def _load_expressions(path: Path) -> dict[str, str]:
    expressions: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        source_id = row.get("source_id")
        expression = row.get("expression", row.get("text", row.get("sentence", "")))
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{path} line {line_number} missing source_id")
        if isinstance(expression, str):
            expressions[source_id] = expression
    return expressions


def _add(groups: dict[str, list[dict[str, Any]]], name: str, row: dict[str, Any]) -> None:
    groups.setdefault(name, []).append(row)


def _tokens(expression: str) -> list[str]:
    return [token.strip(".,;:!?()[]{}\"'").lower() for token in expression.split() if token.strip(".,;:!?()[]{}\"'")]


def _length_bucket(tokens: list[str]) -> str:
    length = len(tokens)
    if length <= 3:
        return "len_1_3"
    if length <= 6:
        return "len_4_6"
    return "len_7_plus"


def _candidate_bucket(valid_k: int) -> str:
    if valid_k <= 4:
        return "candidate_k_1_4"
    if valid_k <= 8:
        return "candidate_k_5_8"
    if valid_k <= 16:
        return "candidate_k_9_16"
    return "candidate_k_17_plus"


def _summarize_group(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(rows)
    acc_at_0_5 = sum(float(row.get("selected_iou", 0.0)) >= 0.5 for row in rows)
    acc_at_0_7 = sum(float(row.get("selected_iou", 0.0)) >= 0.7 for row in rows)
    hit_at_1 = sum(bool(row.get("hit_at_1", row.get("acc_at_0_5", False))) for row in rows)
    hit_at_5 = sum(bool(row.get("hit_at_5", False)) for row in rows)
    mean_iou = sum(float(row.get("selected_iou", 0.0)) for row in rows)
    empty_predictions = sum(int(row.get("prediction_count", 0)) == 0 for row in rows)
    return {
        "group": name,
        "sample_count": sample_count,
        "acc_at_0_5": _safe_rate(acc_at_0_5, sample_count),
        "acc_at_0_7": _safe_rate(acc_at_0_7, sample_count),
        "recall_at_1": _safe_rate(hit_at_1, sample_count),
        "recall_at_5": _safe_rate(hit_at_5, sample_count),
        "mean_iou": _safe_rate(mean_iou, sample_count),
        "empty_prediction_rate": _safe_rate(empty_predictions, sample_count),
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| group | n | acc@0.5 | R@1 | R@5 | mIoU | empty |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {group} | {sample_count} | {acc_at_0_5:.4f} | {recall_at_1:.4f} | {recall_at_5:.4f} | {mean_iou:.4f} | {empty_prediction_rate:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
