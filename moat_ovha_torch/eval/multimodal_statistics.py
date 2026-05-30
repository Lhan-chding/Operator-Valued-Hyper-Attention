from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PublicSummaryValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


def summarize_public_results(
    rows: list[dict[str, Any]],
    *,
    full_model: str,
    baseline_model: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["split"]), str(row["model"]))
        grouped.setdefault(key, []).append(row)

    main_table: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for (task, split, model), group_rows in grouped.items():
        scores = [float(row["score"]) for row in sorted(group_rows, key=lambda row: int(row["seed"]))]
        main_table.setdefault(task, {}).setdefault(split, {})[model] = {
            "mean": _mean(scores),
            "std": _std(scores),
            "ci95": _ci95(scores),
            "seed_count": len({int(row["seed"]) for row in group_rows}),
            "higher_is_better": bool(group_rows[0].get("higher_is_better", True)),
        }

    paired_tests: dict[str, dict[str, dict[str, Any]]] = {}
    for task, split in sorted({(str(row["task"]), str(row["split"])) for row in rows}):
        full = _scores_by_seed(rows, task, split, full_model)
        baseline = _scores_by_seed(rows, task, split, baseline_model)
        common = sorted(set(full) & set(baseline))
        if common:
            deltas = [full[seed] - baseline[seed] for seed in common]
            paired_tests.setdefault(task, {})[split] = {
                "model_delta": full_model + "_minus_" + baseline_model,
                "common_seed_count": len(common),
                "mean_delta": _mean(deltas),
                "paired_permutation_p": _paired_sign_permutation_p(deltas),
                "paired_bootstrap_ci95": _bootstrap_ci95(deltas),
            }

    return {
        "main_table": main_table,
        "paired_tests": paired_tests,
        "per_seed_appendix": sorted(rows, key=lambda row: (str(row["task"]), str(row["split"]), str(row["model"]), int(row["seed"]))),
        "metadata": _metadata(rows),
    }


def validate_public_summary(summary: dict[str, Any]) -> PublicSummaryValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    main_table = summary.get("main_table", {})
    for task, splits in main_table.items():
        for split, models in splits.items():
            for model, values in models.items():
                if values.get("seed_count", 0) < 3:
                    errors.append(f"{task}/{split}/{model} must report at least 3 seeds, not best seed only")
                for key in ("mean", "std", "ci95"):
                    if key not in values:
                        errors.append(f"{task}/{split}/{model} missing {key}")
    if not summary.get("paired_tests"):
        errors.append("paired_tests missing; main deltas require paired permutation/bootstrap evidence")
    else:
        for task, splits in summary["paired_tests"].items():
            for split, values in splits.items():
                if values.get("common_seed_count", 0) < 3:
                    errors.append(f"{task}/{split} paired_tests require at least 3 common seeds")
                for key in ("paired_permutation_p", "paired_bootstrap_ci95"):
                    if key not in values:
                        errors.append(f"{task}/{split} paired_tests missing {key}")
    metadata = summary.get("metadata", {})
    for key in ("parameter_count", "training_steps", "frozen_feature_extractor_version", "hardware"):
        if key not in metadata or metadata[key] in ({}, None):
            errors.append(f"metadata missing {key}")
    if not summary.get("per_seed_appendix"):
        errors.append("per_seed_appendix missing")
    return PublicSummaryValidationReport(ok=not errors, errors=errors, warnings=warnings)


def _metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parameter_count = {}
    feature_versions = {}
    hardware = {}
    training_steps = None
    raw_metric_paths = []
    for row in rows:
        model = str(row["model"])
        if "parameter_count" in row:
            parameter_count[model] = int(row["parameter_count"])
        if "frozen_feature_extractor_version" in row:
            feature_versions.update(dict(row["frozen_feature_extractor_version"]))
        if "hardware" in row:
            hardware.update(dict(row["hardware"]))
        if "training_steps" in row and training_steps is None:
            training_steps = int(row["training_steps"])
        if "raw_metric_path" in row:
            raw_metric_paths.append(str(row["raw_metric_path"]))
    return {
        "parameter_count": parameter_count,
        "training_steps": training_steps,
        "frozen_feature_extractor_version": feature_versions,
        "hardware": hardware,
        "raw_metric_paths": sorted(set(raw_metric_paths)),
    }


def _scores_by_seed(rows: list[dict[str, Any]], task: str, split: str, model: str) -> dict[int, float]:
    return {
        int(row["seed"]): float(row["score"])
        for row in rows
        if str(row["task"]) == task and str(row["split"]) == split and str(row["model"]) == model
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    radius = 1.96 * _std(values) / math.sqrt(max(len(values), 1))
    mean = _mean(values)
    return (mean - radius, mean + radius)


def _paired_sign_permutation_p(deltas: list[float]) -> float:
    if not deltas:
        return 1.0
    observed = abs(sum(deltas))
    total = 2 ** len(deltas)
    extreme = 0
    for mask in range(total):
        signed_sum = 0.0
        for index, delta in enumerate(deltas):
            signed_sum += delta if mask & (1 << index) else -delta
        if abs(signed_sum) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def _bootstrap_ci95(deltas: list[float]) -> tuple[float, float]:
    if not deltas:
        return (0.0, 0.0)
    if len(deltas) == 1:
        return (deltas[0], deltas[0])
    means = []
    for start in range(len(deltas)):
        sample = [deltas[(start + offset) % len(deltas)] for offset in range(len(deltas))]
        means.append(_mean(sample))
    means = sorted(means)
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return (means[low_index], means[high_index])
