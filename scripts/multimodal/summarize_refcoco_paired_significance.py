#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
from scripts.multimodal.score_groundingdino_same_candidates import _load_source_ids
from scripts.multimodal.summarize_refcoco_subset_diagnostics import _load_score_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired RefCOCO significance from per-sample reranker outputs. "
            "Inputs can be score rows or run_public_main per_sample_predictions rows."
        )
    )
    parser.add_argument("--full", type=Path, required=True, help="Per-sample file for the OVHA/full model.")
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        help="Baseline per-sample file as name=path. Repeat for multiple baselines.",
    )
    parser.add_argument("--full-name", default="ovha_admitted_bank")
    parser.add_argument("--input-format", choices=("auto", "score_rows", "public_main_predictions"), default="auto")
    parser.add_argument("--cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--dataset-name", default="refcoco")
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--metric",
        choices=("acc_at_0_5", "acc_at_0_7", "recall_at_1", "mean_iou", "mrr"),
        default="acc_at_0_5",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, required=True)
    payload = summarize_refcoco_paired_significance(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def summarize_refcoco_paired_significance(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.permutation_samples <= 0:
        raise ValueError("--permutation-samples must be positive")
    layout = MultimodalCacheLayout(args.cache_root, args.dataset_name, args.version)
    arrays = _cache_arrays(layout.root, args.split)
    full_rows = _load_score_rows(args.full, input_format=args.input_format, **arrays)
    full_scores = _indexed_scores(full_rows, metric=args.metric)
    rng = np.random.default_rng(int(args.seed))
    comparisons = []
    for baseline_name, baseline_path in _parse_baselines(args.baseline):
        baseline_rows = _load_score_rows(baseline_path, input_format=args.input_format, **arrays)
        baseline_scores = _indexed_scores(baseline_rows, metric=args.metric)
        comparisons.append(
            _compare_scores(
                full_scores,
                baseline_scores,
                full_name=str(args.full_name),
                baseline_name=baseline_name,
                baseline_path=baseline_path,
                metric=str(args.metric),
                rng=rng,
                bootstrap_samples=int(args.bootstrap_samples),
                permutation_samples=int(args.permutation_samples),
            )
        )
    payload = {
        "artifact_type": "refcoco_paired_significance",
        "dataset": args.dataset_name,
        "split": args.split,
        "version": args.version,
        "metric": args.metric,
        "full_name": args.full_name,
        "full_path": str(args.full),
        "bootstrap_samples": int(args.bootstrap_samples),
        "permutation_samples": int(args.permutation_samples),
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_significance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "paired_significance.md").write_text(_markdown_table(comparisons, str(args.metric)) + "\n")
    payload["output_dir"] = str(args.output_dir)
    payload["artifacts"] = {
        "json": str(args.output_dir / "paired_significance.json"),
        "markdown": str(args.output_dir / "paired_significance.md"),
    }
    return payload


def _cache_arrays(root: Path, split: str) -> dict[str, Any]:
    source_ids = _load_source_ids(root, split)
    return {
        "source_index": {source_id: index for index, source_id in enumerate(source_ids)},
        "candidate_boxes": np.load(root / "supervision" / f"candidate_region_boxes_{split}.npy").astype(np.float32, copy=False),
        "bbox_targets": np.load(root / "supervision" / f"bbox_targets_{split}.npy").astype(np.float32, copy=False),
        "region_targets": np.load(root / "supervision" / f"region_targets_{split}.npy").reshape(-1).astype(np.int64, copy=False),
        "region_mask": np.load(root / "masks" / f"region_mask_{split}.npy").astype(bool, copy=False),
    }


def _parse_baselines(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError("--baseline must use name=path")
        name, path = value.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError("--baseline must use non-empty name=path")
        parsed.append((name.strip(), Path(path)))
    return parsed


def _indexed_scores(rows: list[dict[str, Any]], *, metric: str) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    scores: dict[tuple[str, str], float] = {}
    for row in rows:
        source_id = str(row["source_id"])
        seed = str(row.get("seed", "none"))
        key = (seed, source_id)
        if key in scores:
            counts[key] += 1
            key = (f"{seed}#{counts[key]}", source_id)
        scores[key] = _metric_value(row, metric)
    return scores


def _metric_value(row: dict[str, Any], metric: str) -> float:
    selected_iou = float(row.get("selected_iou", 0.0))
    if metric == "acc_at_0_5":
        return float(selected_iou >= 0.5)
    if metric == "acc_at_0_7":
        return float(selected_iou >= 0.7)
    if metric == "recall_at_1":
        return float(bool(row.get("hit_at_1", False)))
    if metric == "mean_iou":
        return selected_iou
    if metric == "mrr":
        rank = row.get("target_rank", row.get("first_positive_rank"))
        return 0.0 if rank in (None, 0) else 1.0 / float(rank)
    raise ValueError(f"unknown metric: {metric}")


def _compare_scores(
    full_scores: dict[tuple[str, str], float],
    baseline_scores: dict[tuple[str, str], float],
    *,
    full_name: str,
    baseline_name: str,
    baseline_path: Path,
    metric: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict[str, Any]:
    shared_keys = sorted(set(full_scores) & set(baseline_scores))
    if not shared_keys:
        raise ValueError(f"no paired samples shared by full and baseline {baseline_name}")
    full = np.asarray([full_scores[key] for key in shared_keys], dtype=np.float64)
    baseline = np.asarray([baseline_scores[key] for key in shared_keys], dtype=np.float64)
    delta = full - baseline
    observed_delta = float(delta.mean())
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sample_indices = rng.integers(0, delta.shape[0], size=delta.shape[0])
        bootstrap_means[index] = float(delta[sample_indices].mean())
    extreme_count = 0
    remaining = permutation_samples
    while remaining > 0:
        chunk = min(512, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=(chunk, delta.shape[0]))
        permuted = (signs * delta.reshape(1, -1)).mean(axis=1)
        extreme_count += int(np.count_nonzero(np.abs(permuted) >= abs(observed_delta)))
        remaining -= chunk
    p_value = float((extreme_count + 1) / (permutation_samples + 1))
    return {
        "full_model": full_name,
        "baseline_model": baseline_name,
        "baseline_path": str(baseline_path),
        "metric": metric,
        "paired_observation_count": int(delta.shape[0]),
        "full_mean": float(full.mean()),
        "baseline_mean": float(baseline.mean()),
        "mean_delta": observed_delta,
        "bootstrap_ci_95": [float(np.quantile(bootstrap_means, 0.025)), float(np.quantile(bootstrap_means, 0.975))],
        "paired_sign_permutation_p": p_value,
    }


def _markdown_table(rows: list[dict[str, Any]], metric: str) -> str:
    lines = [
        f"| baseline | n | full {metric} | baseline {metric} | delta | 95% CI | paired p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci_low, ci_high = row["bootstrap_ci_95"]
        lines.append(
            "| {baseline_model} | {paired_observation_count} | {full_mean:.4f} | {baseline_mean:.4f} | {mean_delta:.4f} | [{ci_low:.4f}, {ci_high:.4f}] | {paired_sign_permutation_p:.4g} |".format(
                ci_low=ci_low,
                ci_high=ci_high,
                **row,
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
