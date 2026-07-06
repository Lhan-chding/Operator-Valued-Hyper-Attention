#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from moat_ovha_torch.data.multimodal.cache_schema import file_sha256
from scripts.multimodal.score_groundingdino_same_candidates import _pairwise_iou


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a K-truncated RefCOCO GroundingDINO proposal cache for proposal-pool "
            "sensitivity studies. The source cache must already contain at least K proposals."
        )
    )
    parser.add_argument("--source-cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--source-dataset-name", default="refcoco_gdino_proposals")
    parser.add_argument("--source-version", default="v0.1")
    parser.add_argument("--output-cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--output-dataset-name", required=True)
    parser.add_argument("--output-version", default="v0.1")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "testA", "testB"])
    parser.add_argument("--max-proposals-per-sample", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    payload = build_groundingdino_proposal_k_cache(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_groundingdino_proposal_k_cache(args: argparse.Namespace) -> dict[str, Any]:
    max_k = int(args.max_proposals_per_sample)
    if max_k <= 0:
        raise ValueError("--max-proposals-per-sample must be positive")
    source_root = Path(args.source_cache_root) / args.source_dataset_name / args.source_version
    output_root = Path(args.output_cache_root) / args.output_dataset_name / args.output_version
    if not source_root.exists():
        raise ValueError(f"source cache does not exist: {source_root}")
    if output_root.exists():
        if not bool(args.overwrite):
            raise ValueError(f"output cache already exists; pass --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("source and output cache roots must differ")

    shutil.copytree(source_root, output_root)
    summaries = [_truncate_split(output_root, split, max_k=max_k) for split in args.splits]
    _patch_common_metadata(
        output_root,
        source_dataset=str(args.source_dataset_name),
        source_version=str(args.source_version),
        max_k=max_k,
    )
    _write_checksums(output_root)
    return {
        "ok": True,
        "artifact_type": "groundingdino_proposal_k_sensitivity_cache",
        "source_dataset": str(args.source_dataset_name),
        "source_version": str(args.source_version),
        "dataset": str(args.output_dataset_name),
        "version": str(args.output_version),
        "output_root": str(output_root),
        "splits": summaries,
    }


def _truncate_split(root: Path, split: str, *, max_k: int) -> dict[str, Any]:
    region_features_path = root / "token_fields" / f"region_{split}.npy"
    region_pos_path = root / "positions" / f"region_pos_{split}.npy"
    region_mask_path = root / "masks" / f"region_mask_{split}.npy"
    boxes_path = root / "supervision" / f"candidate_region_boxes_{split}.npy"
    detector_scores_path = root / "supervision" / f"candidate_region_detector_scores_{split}.npy"
    bbox_targets_path = root / "supervision" / f"bbox_targets_{split}.npy"
    if not region_features_path.exists() or not boxes_path.exists() or not bbox_targets_path.exists():
        raise ValueError(f"split {split} is missing required region proposal cache artifacts")

    region_features = _truncate_candidate_axis(np.load(region_features_path), max_k, name=str(region_features_path))
    region_pos = _truncate_candidate_axis(np.load(region_pos_path), max_k, name=str(region_pos_path))
    region_mask = _truncate_mask_axis(np.load(region_mask_path).astype(bool, copy=False), max_k, name=str(region_mask_path))
    boxes = _truncate_candidate_axis(np.load(boxes_path).astype(np.float32, copy=False), max_k, name=str(boxes_path))
    bbox_targets = np.load(bbox_targets_path).astype(np.float32, copy=False)
    detector_scores = None
    if detector_scores_path.exists():
        detector_scores = _truncate_mask_axis(
            np.load(detector_scores_path).astype(np.float32, copy=False),
            max_k,
            name=str(detector_scores_path),
        )

    target_indices, labels, oracle_best_iou = _target_labels_for_truncated_pool(boxes, region_mask, bbox_targets)
    np.save(region_features_path, region_features)
    np.save(region_pos_path, region_pos)
    np.save(region_mask_path, region_mask)
    np.save(boxes_path, boxes)
    if detector_scores is not None:
        np.save(detector_scores_path, detector_scores)
    np.save(root / "supervision" / f"region_targets_{split}.npy", target_indices.reshape(-1, 1))
    np.save(root / "supervision" / f"task_labels_{split}.npy", labels)
    _write_target_histogram(
        root / "supervision" / f"target_slot_histogram_by_valid_count_{split}.json",
        target_indices,
        region_mask,
    )
    _patch_sample_records(
        root / "provenance" / f"sample_records_{split}.jsonl",
        boxes=boxes,
        detector_scores=detector_scores,
        target_indices=target_indices,
    )
    valid_counts = region_mask.sum(axis=1)
    return {
        "split": split,
        "sample_count": int(boxes.shape[0]),
        "max_proposals_per_sample": int(max_k),
        "mean_retained_proposals": float(np.mean(valid_counts)) if valid_counts.size else 0.0,
        "empty_proposal_count": int(np.count_nonzero(valid_counts == 0)),
        "proposal_oracle_acc_at_0_5": float(np.mean(oracle_best_iou >= 0.5)) if oracle_best_iou.size else 0.0,
        "proposal_oracle_mean_iou": float(np.mean(oracle_best_iou)) if oracle_best_iou.size else 0.0,
    }


def _truncate_candidate_axis(array: np.ndarray, max_k: int, *, name: str) -> np.ndarray:
    if array.ndim < 3:
        raise ValueError(f"{name} must have at least [sample, candidate, ...] axes")
    if int(array.shape[1]) < max_k:
        raise ValueError(f"{name} has only {array.shape[1]} candidates, cannot build K={max_k}")
    return np.asarray(array[:, :max_k, ...])


def _truncate_mask_axis(array: np.ndarray, max_k: int, *, name: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [sample, candidate]")
    if int(array.shape[1]) < max_k:
        raise ValueError(f"{name} has only {array.shape[1]} candidates, cannot build K={max_k}")
    return np.asarray(array[:, :max_k])


def _target_labels_for_truncated_pool(
    boxes: np.ndarray,
    region_mask: np.ndarray,
    bbox_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if boxes.shape[:2] != region_mask.shape:
        raise ValueError("region_mask must match candidate_region_boxes sample/candidate axes")
    if boxes.shape[0] != bbox_targets.shape[0]:
        raise ValueError("bbox_targets sample count must match candidate_region_boxes")
    ious = np.zeros(boxes.shape[:2], dtype=np.float32)
    for row_index in range(boxes.shape[0]):
        row_ious = _pairwise_iou(boxes[row_index], bbox_targets[row_index].reshape(1, 4)).reshape(-1)
        ious[row_index] = np.where(region_mask[row_index], row_ious, -1.0)
    target_indices = np.argmax(ious, axis=1).astype(np.int64)
    labels = np.zeros((boxes.shape[0], boxes.shape[1]), dtype=np.float32)
    labels[np.arange(boxes.shape[0]), target_indices] = 1.0
    oracle_best_iou = ious[np.arange(boxes.shape[0]), target_indices]
    oracle_best_iou = np.maximum(oracle_best_iou, 0.0)
    return target_indices, labels, oracle_best_iou


def _patch_sample_records(
    path: Path,
    *,
    boxes: np.ndarray,
    detector_scores: np.ndarray | None,
    target_indices: np.ndarray,
) -> None:
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != boxes.shape[0]:
        raise ValueError(f"{path} row count must match candidate_region_boxes")
    patched = []
    for row_index, row in enumerate(rows):
        updated = dict(row)
        updated["candidate_region_boxes"] = [
            [float(value) for value in box]
            for box in boxes[row_index].tolist()
        ]
        if detector_scores is not None:
            updated["candidate_region_detector_scores"] = [
                float(value)
                for value in detector_scores[row_index].tolist()
            ]
        updated["target_region_index"] = int(target_indices[row_index])
        patched.append(updated)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in patched) + "\n")


def _write_target_histogram(path: Path, target_indices: np.ndarray, region_mask: np.ndarray) -> None:
    histogram: dict[str, dict[str, int]] = {}
    for target_index, mask_row in zip(target_indices.tolist(), region_mask):
        valid_k = str(int(np.count_nonzero(mask_row)))
        histogram.setdefault(valid_k, {})
        histogram[valid_k][str(int(target_index))] = histogram[valid_k].get(str(int(target_index)), 0) + 1
    path.write_text(json.dumps(histogram, indent=2, sort_keys=True) + "\n")


def _patch_common_metadata(root: Path, *, source_dataset: str, source_version: str, max_k: int) -> None:
    data_card_path = root / "data_card.json"
    if data_card_path.exists():
        data_card = json.loads(data_card_path.read_text())
    else:
        data_card = {}
    protocol = data_card.get("candidate_protocol")
    if not isinstance(protocol, dict):
        protocol = {}
    protocol.update(
        {
            "fixed_k": int(max_k),
            "sensitivity_axis": "retained_proposal_budget_k",
            "source_dataset": source_dataset,
            "source_version": source_version,
        }
    )
    data_card["candidate_protocol"] = protocol
    data_card_path.write_text(json.dumps(data_card, sort_keys=True) + "\n")


def _write_checksums(root: Path) -> None:
    checksums = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
