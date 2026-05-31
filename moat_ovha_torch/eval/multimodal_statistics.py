from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task


REGION_TEXT_REQUIRED_PUBLIC_METRICS = (
    "acc_at_0_5",
    "recall_at_1",
    "recall_at_5",
    "mean_iou",
    "phrase_region_topk_accuracy",
    "alignment_entropy",
    "cato_router_load",
    "cato_candidate_loss",
    "cato_top_alignment_accuracy",
    "null_unmatched_rate",
)
SENTIMENT_REQUIRED_PUBLIC_METRICS = (
    "mae",
    "pearson_correlation",
    "accuracy",
    "f1",
    "missing_modality_performance_drop",
    "corruption_robustness_auc",
    "router_load_by_corruption_type",
    "lrio_rank_entropy",
    "spo_prototype_entropy",
    "rceo_reliability_calibration",
)
REQUIRED_PUBLIC_METRICS_BY_TASK = {
    "phrase_region_grounding": REGION_TEXT_REQUIRED_PUBLIC_METRICS,
    "region_text_grounding": REGION_TEXT_REQUIRED_PUBLIC_METRICS,
    "refcoco": REGION_TEXT_REQUIRED_PUBLIC_METRICS,
    "flickr30k_entities": REGION_TEXT_REQUIRED_PUBLIC_METRICS,
    "sentiment_emotion": SENTIMENT_REQUIRED_PUBLIC_METRICS,
    "sentiment_regression": SENTIMENT_REQUIRED_PUBLIC_METRICS,
    "emotion_classification": SENTIMENT_REQUIRED_PUBLIC_METRICS,
    "cmu_mosei": SENTIMENT_REQUIRED_PUBLIC_METRICS,
    "meld": SENTIMENT_REQUIRED_PUBLIC_METRICS,
}
PUBLIC_PROBABILITY_METRICS = frozenset(
    {
        "acc_at_0_5",
        "recall_at_1",
        "recall_at_5",
        "mean_iou",
        "phrase_region_topk_accuracy",
        "cato_router_load",
        "cato_top_alignment_accuracy",
        "null_unmatched_rate",
        "accuracy",
        "f1",
        "missing_modality_performance_drop",
        "corruption_robustness_auc",
    }
)
PUBLIC_NON_NEGATIVE_METRICS = frozenset(
    {
        "alignment_entropy",
        "cato_candidate_loss",
        "mae",
        "lrio_rank_entropy",
        "spo_prototype_entropy",
    }
)
PUBLIC_CORRELATION_METRICS = frozenset({"pearson_correlation"})
PUBLIC_NESTED_PROBABILITY_METRICS = frozenset({"router_load_by_corruption_type"})


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
        higher_is_better = _raw_metric_direction(group_rows, task, split, model)
        main_table.setdefault(task, {}).setdefault(split, {})[model] = {
            "mean": _mean(scores),
            "std": _std(scores),
            "ci95": _ci95(scores),
            "seed_count": len({int(row["seed"]) for row in group_rows}),
            "per_seed_scores": scores,
            "higher_is_better": higher_is_better,
        }

    paired_tests: dict[str, dict[str, dict[str, Any]]] = {}
    for task, split in sorted({(str(row["task"]), str(row["split"])) for row in rows}):
        full = _scores_by_seed(rows, task, split, full_model)
        baseline = _scores_by_seed(rows, task, split, baseline_model)
        common = sorted(set(full) & set(baseline))
        if common:
            higher_is_better = _higher_is_better(main_table, task, split, full_model)
            deltas = [_directional_delta(full[seed], baseline[seed], higher_is_better) for seed in common]
            paired_tests.setdefault(task, {})[split] = {
                "model_delta": full_model + "_minus_" + baseline_model,
                "metric_direction": "higher_is_better" if higher_is_better else "lower_is_better",
                "delta_interpretation": f"positive means {full_model} improves over {baseline_model}",
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
        "reporting_metadata": _reporting_metadata(rows),
    }


def validate_public_summary(summary: dict[str, Any]) -> PublicSummaryValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    main_table = summary.get("main_table", {})
    for task, splits in main_table.items():
        for split, models in splits.items():
            _validate_required_same_feature_baselines(str(task), str(split), models, errors)
            _validate_metric_direction_consistency(str(task), str(split), models, errors)
            if not isinstance(models, dict):
                continue
            for model, values in models.items():
                if isinstance(values, dict):
                    if values.get("seed_count", 0) < 3:
                        errors.append(f"{task}/{split}/{model} must report at least 3 seeds, not best seed only")
                    for key in ("mean", "std", "ci95"):
                        if key not in values:
                            errors.append(f"{task}/{split}/{model} missing {key}")
                _validate_main_table_model_row(summary, str(task), str(split), str(model), values, errors)
    if not summary.get("paired_tests"):
        errors.append("paired_tests missing; main deltas require paired permutation/bootstrap evidence")
    else:
        for task, splits in summary["paired_tests"].items():
            for split, values in splits.items():
                if values.get("common_seed_count", 0) < 3:
                    errors.append(f"{task}/{split} paired_tests require at least 3 common seeds")
                for key in ("metric_direction", "mean_delta", "paired_permutation_p", "paired_bootstrap_ci95"):
                    if key not in values:
                        errors.append(f"{task}/{split} paired_tests missing {key}")
                _validate_paired_delta_consistency(summary, str(task), str(split), values, errors)
    metadata = summary.get("metadata", {})
    for key in ("parameter_count", "training_steps", "frozen_feature_extractor_version", "hardware"):
        if key not in metadata or metadata[key] in ({}, None):
            errors.append(f"metadata missing {key}")
    _validate_reporting_metadata(summary, metadata, errors)
    _validate_report_facing_metadata(summary, errors)
    _validate_baseline_strength(summary.get("paired_tests", {}), metadata, errors)
    _validate_label_provenance(summary.get("per_seed_appendix", []), metadata, errors)
    _validate_public_metric_inventory(summary.get("per_seed_appendix", []), errors)
    if not summary.get("per_seed_appendix"):
        errors.append("per_seed_appendix missing")
    return PublicSummaryValidationReport(ok=not errors, errors=errors, warnings=warnings)


def _validate_required_same_feature_baselines(
    task: str,
    split: str,
    models: Any,
    errors: list[str],
) -> None:
    if not isinstance(models, dict):
        errors.append(f"{task}/{split} main_table must be keyed by model")
        return
    try:
        required_baselines = baseline_names_for_task(task)
    except ValueError:
        return
    for baseline in required_baselines:
        if baseline not in models:
            errors.append(f"statistics summary missing required same-feature baseline: {baseline}")


def _validate_metric_direction_consistency(
    task: str,
    split: str,
    models: Any,
    errors: list[str],
) -> None:
    if not isinstance(models, dict):
        return
    directions: set[bool] = set()
    for model, values in models.items():
        if not isinstance(values, dict):
            continue
        if "higher_is_better" not in values:
            errors.append(f"{task}/{split}/{model} missing higher_is_better")
            continue
        higher_is_better = values["higher_is_better"]
        if not isinstance(higher_is_better, bool):
            errors.append(f"{task}/{split}/{model} higher_is_better must be boolean")
            continue
        directions.add(higher_is_better)
    if len(directions) > 1:
        errors.append(f"{task}/{split} higher_is_better must be consistent across models")


def _raw_metric_direction(
    rows: list[dict[str, Any]],
    task: str,
    split: str,
    model: str,
) -> bool:
    directions: set[bool] = set()
    for row in rows:
        row_name = f"{task}/{split}/{model}/seed={row.get('seed', '?')}"
        if "higher_is_better" not in row:
            raise ValueError(f"{row_name} higher_is_better missing")
        higher_is_better = row["higher_is_better"]
        if not isinstance(higher_is_better, bool):
            raise ValueError(f"{row_name} higher_is_better must be boolean")
        directions.add(higher_is_better)
    if len(directions) != 1:
        raise ValueError(f"{task}/{split}/{model} higher_is_better must be consistent across seeds")
    return next(iter(directions))


def _metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parameter_count = {}
    baseline_strength = {}
    feature_versions = {}
    hardware = {}
    training_steps = None
    raw_metric_paths = []
    for row in rows:
        model = str(row["model"])
        if "parameter_count" in row:
            parameter_count[model] = int(row["parameter_count"])
        if "baseline_strength" in row:
            baseline_strength[model] = str(row["baseline_strength"])
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
        "baseline_strength": baseline_strength,
        "training_steps": training_steps,
        "frozen_feature_extractor_version": feature_versions,
        "hardware": hardware,
        "label_provenance": _label_provenance(rows),
        "raw_metric_paths": sorted(set(raw_metric_paths)),
    }


def _reporting_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parameter_count: dict[str, int] = {}
    training_steps: dict[str, int] = {}
    feature_versions: dict[str, Any] = {}
    hardware: dict[str, Any] = {}
    wall_clock: dict[str, Any] = {}
    per_seed_table: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model"])
        if "parameter_count" in row:
            parameter_count[model] = int(row["parameter_count"])
        if "training_steps" in row:
            training_steps[model] = int(row["training_steps"])
        if "frozen_feature_extractor_version" in row:
            feature_versions.update(dict(row["frozen_feature_extractor_version"]))
        if "hardware" in row:
            hardware.update(dict(row["hardware"]))
            if row["hardware"].get("wall_clock_hours") is not None:
                wall_clock["wall_clock_hours"] = row["hardware"]["wall_clock_hours"]
        per_seed_table.append(
            {
                "task": str(row["task"]),
                "split": str(row["split"]),
                "model": model,
                "seed": int(row["seed"]),
                "score": float(row["score"]),
                "raw_metric_path": str(row["raw_metric_path"]) if row.get("raw_metric_path") else "",
            }
        )
    return {
        "parameter_count": parameter_count,
        "training_steps": training_steps,
        "frozen_feature_versions": feature_versions,
        "hardware": hardware,
        "wall_clock_summary": wall_clock,
        "per_seed_table": sorted(
            per_seed_table,
            key=lambda row: (row["task"], row["split"], row["model"], row["seed"]),
        ),
    }


def _label_provenance(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    for row in rows:
        supervision_type, source = _row_label_provenance(row)
        if supervision_type is None:
            continue
        entry = provenance.setdefault(supervision_type, {"count": 0, "sources": []})
        entry["count"] += 1
        if source:
            entry["sources"].append(source)
    for entry in provenance.values():
        entry["sources"] = sorted(set(entry["sources"]))
    return provenance


def _validate_label_provenance(
    appendix_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    errors: list[str],
) -> None:
    metadata_provenance = metadata.get("label_provenance", {}) if isinstance(metadata, dict) else {}
    for row in appendix_rows:
        explicit = row.get("label_provenance", {})
        explicit_type = _normalized_supervision_type(explicit.get("supervision_type")) if isinstance(explicit, dict) else None
        must_report_as = _normalized_supervision_type(explicit.get("must_report_as")) if isinstance(explicit, dict) else None
        pseudo_source = row.get("pseudo_label_source")
        weak_source = row.get("weak_label_source") or row.get("weak_labels_source")
        row_name = f"{row.get('task', '?')}/{row.get('split', '?')}/{row.get('model', '?')}/seed={row.get('seed', '?')}"
        if pseudo_source and explicit_type not in (None, "pseudo"):
            errors.append(f"{row_name}: pseudo labels must be reported as pseudo, not {explicit_type}")
        if weak_source and explicit_type not in (None, "weak"):
            errors.append(f"{row_name}: weak labels must be reported as weak, not {explicit_type}")
        if must_report_as and explicit_type and must_report_as != explicit_type:
            errors.append(f"{row_name}: label_provenance.must_report_as={must_report_as} conflicts with {explicit_type}")
        supervision_type, _ = _row_label_provenance(row)
        if supervision_type in {"weak", "pseudo"} and supervision_type not in metadata_provenance:
            errors.append(f"metadata.label_provenance missing {supervision_type} supervision summary")


def _validate_public_metric_inventory(appendix_rows: Any, errors: list[str]) -> None:
    if not isinstance(appendix_rows, list):
        return
    for row in appendix_rows:
        if not isinstance(row, dict):
            continue
        task = str(row.get("task", ""))
        required = REQUIRED_PUBLIC_METRICS_BY_TASK.get(task)
        if not required:
            continue
        row_name = f"{row.get('task', '?')}/{row.get('split', '?')}/{row.get('model', '?')}/seed={row.get('seed', '?')}"
        metrics = row.get("public_metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{row_name} missing public_metrics object")
            continue
        for metric in required:
            if metric not in metrics:
                errors.append(f"{row_name} missing required public metric: {metric}")
                continue
            _validate_public_metric_value(row_name, metric, metrics.get(metric), errors)


def _validate_public_metric_value(row_name: str, metric: str, value: Any, errors: list[str]) -> None:
    if isinstance(value, dict):
        if not value:
            errors.append(f"{row_name} public metric {metric} must be non-empty")
            return
        if metric in PUBLIC_NESTED_PROBABILITY_METRICS:
            _validate_nested_numeric_metric(row_name, metric, value, errors, value_range=(0.0, 1.0))
            return
        if metric == "rceo_reliability_calibration":
            _validate_rceo_calibration_metric(row_name, value, errors)
            return
        _validate_nested_numeric_metric(row_name, metric, value, errors)
        return
    numeric = _finite_float(value)
    if numeric is None:
        errors.append(f"{row_name} public metric {metric} must be finite")
        return
    if metric in PUBLIC_PROBABILITY_METRICS and not 0.0 <= numeric <= 1.0:
        errors.append(f"{row_name} public metric {metric} must be in [0, 1]")
    elif metric in PUBLIC_CORRELATION_METRICS and not -1.0 <= numeric <= 1.0:
        errors.append(f"{row_name} public metric {metric} must be in [-1, 1]")
    elif metric in PUBLIC_NON_NEGATIVE_METRICS and numeric < 0.0:
        errors.append(f"{row_name} public metric {metric} must be finite non-negative")


def _validate_rceo_calibration_metric(row_name: str, value: dict[str, Any], errors: list[str]) -> None:
    ece = _finite_float(value.get("ece", value.get("expected_calibration_error")))
    if ece is None or not 0.0 <= ece <= 1.0:
        errors.append(f"{row_name} public metric rceo_reliability_calibration.ece must be in [0, 1]")
    bin_count = _safe_int(value.get("bin_count", value.get("bins")))
    if bin_count is None or bin_count <= 0:
        errors.append(f"{row_name} public metric rceo_reliability_calibration.bin_count must be positive")
    curve = value.get("calibration_curve", value.get("curve"))
    if curve is not None and not isinstance(curve, list):
        errors.append(f"{row_name} public metric rceo_reliability_calibration.calibration_curve must be a list")


def _validate_nested_numeric_metric(
    row_name: str,
    metric: str,
    value: dict[str, Any],
    errors: list[str],
    *,
    value_range: tuple[float, float] | None = None,
) -> None:
    observed_numeric = False
    for key, nested in value.items():
        if not str(key).strip():
            errors.append(f"{row_name} public metric {metric} contains empty nested key")
            return
        if isinstance(nested, dict):
            _validate_nested_numeric_metric(row_name, f"{metric}.{key}", nested, errors, value_range=value_range)
            observed_numeric = True
        else:
            numeric = _finite_float(nested)
            if numeric is None:
                errors.append(f"{row_name} public metric {metric}.{key} must be finite")
                return
            if value_range is not None and not value_range[0] <= numeric <= value_range[1]:
                errors.append(
                    f"{row_name} public metric {metric}.{key} must be in "
                    f"[{value_range[0]:g}, {value_range[1]:g}]"
                )
                return
            observed_numeric = True
    if not observed_numeric:
        errors.append(f"{row_name} public metric {metric} must contain finite values")


def _validate_reporting_metadata(summary: dict[str, Any], metadata: dict[str, Any], errors: list[str]) -> None:
    parameter_count = metadata.get("parameter_count", {}) if isinstance(metadata, dict) else {}
    for model in _models_in_main_table(summary.get("main_table", {})):
        if not isinstance(parameter_count, dict) or model not in parameter_count:
            errors.append(f"parameter_count missing for model: {model}")

    hardware = metadata.get("hardware", {}) if isinstance(metadata, dict) else {}
    if not isinstance(hardware, dict) or hardware.get("wall_clock_hours") is None:
        errors.append("metadata.hardware missing wall_clock_hours")

    raw_metric_paths = metadata.get("raw_metric_paths", []) if isinstance(metadata, dict) else []
    if not raw_metric_paths:
        errors.append("metadata.raw_metric_paths missing; cannot report only final summary without raw metrics")
    for row in summary.get("per_seed_appendix", []):
        row_name = f"{row.get('task', '?')}/{row.get('split', '?')}/{row.get('model', '?')}/seed={row.get('seed', '?')}"
        if not row.get("raw_metric_path"):
            errors.append(f"{row_name}: raw_metric_path missing")


def _validate_report_facing_metadata(summary: dict[str, Any], errors: list[str]) -> None:
    models = _models_in_main_table(summary.get("main_table", {}))
    metadata = summary.get("reporting_metadata")
    if not isinstance(metadata, dict):
        for key in (
            "parameter_count",
            "training_steps",
            "frozen_feature_versions",
            "hardware",
            "wall_clock_summary",
            "per_seed_table",
        ):
            errors.append(f"reporting_metadata missing {key}")
        return

    _validate_report_model_map(metadata, "parameter_count", models, errors)
    _validate_report_model_map(metadata, "training_steps", models, errors)
    for key in ("frozen_feature_versions", "hardware", "wall_clock_summary"):
        if _is_empty_report_value(metadata.get(key)):
            errors.append(f"reporting_metadata missing {key}")
    per_seed_table = metadata.get("per_seed_table")
    if not isinstance(per_seed_table, list) or not per_seed_table:
        errors.append("reporting_metadata missing per_seed_table")
        return
    _validate_reporting_per_seed_table(summary, per_seed_table, errors)
    covered_models = {str(row.get("model")) for row in per_seed_table if isinstance(row, dict) and row.get("model")}
    for model in sorted(models):
        if model not in covered_models:
            errors.append(f"reporting_metadata per_seed_table missing model: {model}")


def _validate_report_model_map(
    metadata: dict[str, Any],
    key: str,
    models: set[str],
    errors: list[str],
) -> None:
    values = metadata.get(key)
    if not isinstance(values, dict) or not values:
        errors.append(f"reporting_metadata missing {key}")
        return
    for model in sorted(models):
        if _is_empty_report_value(values.get(model)):
            errors.append(f"reporting_metadata {key} missing model: {model}")


def _validate_reporting_per_seed_table(
    summary: dict[str, Any],
    per_seed_table: list[Any],
    errors: list[str],
) -> None:
    expected = _per_seed_table_signature(summary.get("per_seed_appendix", []))
    observed = _per_seed_table_signature(per_seed_table)
    if expected != observed:
        errors.append("reporting_metadata per_seed_table must match per_seed_appendix")


def _per_seed_table_signature(rows: Any) -> set[tuple[str, str, str, int, float, str]]:
    if not isinstance(rows, list):
        return set()
    signature: set[tuple[str, str, str, int, float, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        seed = _safe_int(row.get("seed"))
        score = _finite_float(row.get("score"))
        if seed is None or score is None:
            continue
        signature.add(
            (
                str(row.get("task")),
                str(row.get("split")),
                str(row.get("model")),
                seed,
                score,
                str(row.get("raw_metric_path", "")),
            )
        )
    return signature


def _is_empty_report_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, list, dict, tuple, set)):
        return len(value) == 0
    return False


def _models_in_main_table(main_table: dict[str, Any]) -> set[str]:
    models: set[str] = set()
    if not isinstance(main_table, dict):
        return models
    for splits in main_table.values():
        if not isinstance(splits, dict):
            continue
        for split_models in splits.values():
            if isinstance(split_models, dict):
                models.update(str(model) for model in split_models)
    return models


def _validate_baseline_strength(paired_tests: dict[str, Any], metadata: dict[str, Any], errors: list[str]) -> None:
    baseline_strength = metadata.get("baseline_strength", {}) if isinstance(metadata, dict) else {}
    if not isinstance(baseline_strength, dict):
        return
    for splits in paired_tests.values() if isinstance(paired_tests, dict) else ():
        if not isinstance(splits, dict):
            continue
        for values in splits.values():
            if not isinstance(values, dict):
                continue
            baseline_model = _baseline_model_from_delta(values.get("model_delta"))
            if baseline_model is None:
                continue
            strength = str(baseline_strength.get(baseline_model, "")).strip().lower().replace("-", "_")
            if strength in {"weak", "weak_baseline", "toy", "sanity", "ablation_only"}:
                errors.append(
                    "cannot report only improvement over weak baseline: "
                    f"{baseline_model} is marked {strength}"
                )


def _validate_paired_delta_consistency(
    summary: dict[str, Any],
    task: str,
    split: str,
    values: Any,
    errors: list[str],
) -> None:
    if not isinstance(values, dict):
        errors.append(f"{task}/{split} paired_tests entry must be an object")
        return
    models = _models_from_delta(values.get("model_delta"))
    if models is None:
        errors.append(f"{task}/{split} paired_tests model_delta must be '<full>_minus_<baseline>'")
        return
    full_model, baseline_model = models
    direction = values.get("metric_direction")
    if direction not in {"higher_is_better", "lower_is_better"}:
        errors.append(f"{task}/{split} paired_tests metric_direction must be higher_is_better or lower_is_better")
        return
    higher_is_better = _higher_is_better(summary.get("main_table", {}), task, split, full_model)
    expected_direction = "higher_is_better" if higher_is_better else "lower_is_better"
    if direction != expected_direction:
        errors.append(f"{task}/{split} paired_tests metric_direction disagrees with main_table higher_is_better")
        return
    expected_deltas = _expected_paired_deltas(summary, task, split, full_model, baseline_model, higher_is_better)
    observed_common_seed_count = _safe_int(values.get("common_seed_count"))
    if observed_common_seed_count != len(expected_deltas):
        errors.append(f"{task}/{split} paired_tests common_seed_count disagrees with per_seed_appendix")
    if not expected_deltas:
        return
    expected_mean = _mean(expected_deltas)
    observed_mean = _finite_float(values.get("mean_delta"))
    if observed_mean is None or not math.isclose(observed_mean, expected_mean, rel_tol=1e-9, abs_tol=1e-9):
        errors.append(f"{task}/{split} paired_tests mean_delta disagrees with metric_direction")
    expected_ci = _bootstrap_ci95(expected_deltas)
    observed_ci = _finite_ci95(values.get("paired_bootstrap_ci95"))
    if observed_ci is None or any(
        not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9)
        for observed, expected in zip(observed_ci, expected_ci)
    ):
        errors.append(f"{task}/{split} paired_tests bootstrap CI disagrees with metric_direction")


def _validate_main_table_model_row(
    summary: dict[str, Any],
    task: str,
    split: str,
    model: str,
    values: Any,
    errors: list[str],
) -> None:
    if not isinstance(values, dict):
        errors.append(f"{task}/{split}/{model} main table row must be an object")
        return
    appendix_scores = _appendix_scores(summary, task, split, model)
    observed_seed_count = _safe_int(values.get("seed_count"))
    if observed_seed_count != len(appendix_scores):
        errors.append(f"{task}/{split}/{model} seed_count disagrees with per_seed_appendix")
    if not appendix_scores:
        return
    expected_scores = [score for _, score in appendix_scores]
    observed_scores = _finite_float_list(values.get("per_seed_scores"))
    if observed_scores is None or len(observed_scores) != len(expected_scores) or any(
        not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9)
        for observed, expected in zip(observed_scores, expected_scores)
    ):
        errors.append(f"{task}/{split}/{model} per_seed_scores disagree with per_seed_appendix")
    observed_mean = _finite_float(values.get("mean"))
    if observed_mean is None or not math.isclose(observed_mean, _mean(expected_scores), rel_tol=1e-9, abs_tol=1e-9):
        errors.append(f"{task}/{split}/{model} mean disagrees with per_seed_appendix")
    observed_std = _finite_float(values.get("std"))
    if observed_std is None or not math.isclose(observed_std, _std(expected_scores), rel_tol=1e-9, abs_tol=1e-9):
        errors.append(f"{task}/{split}/{model} std disagrees with per_seed_appendix")
    observed_ci = _finite_ci95(values.get("ci95"))
    expected_ci = _ci95(expected_scores)
    if observed_ci is None or any(
        not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9)
        for observed, expected in zip(observed_ci, expected_ci)
    ):
        errors.append(f"{task}/{split}/{model} ci95 disagrees with per_seed_appendix")


def _baseline_model_from_delta(model_delta: Any) -> str | None:
    if not model_delta:
        return None
    text = str(model_delta)
    if "_minus_" not in text:
        return None
    return text.rsplit("_minus_", 1)[1]


def _models_from_delta(model_delta: Any) -> tuple[str, str] | None:
    if not model_delta:
        return None
    text = str(model_delta)
    if "_minus_" not in text:
        return None
    full_model, baseline_model = text.rsplit("_minus_", 1)
    if not full_model or not baseline_model:
        return None
    return full_model, baseline_model


def _row_label_provenance(row: dict[str, Any]) -> tuple[str | None, str | None]:
    explicit = row.get("label_provenance", {})
    if isinstance(explicit, dict):
        supervision_type = _normalized_supervision_type(explicit.get("supervision_type"))
        source = explicit.get("source")
        if supervision_type:
            return supervision_type, str(source) if source else None
    if row.get("pseudo_label_source"):
        return "pseudo", str(row["pseudo_label_source"])
    weak_source = row.get("weak_label_source") or row.get("weak_labels_source")
    if weak_source:
        return "weak", str(weak_source)
    return None, None


def _normalized_supervision_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "gt": "ground_truth",
        "gold": "ground_truth",
        "public_ground_truth": "ground_truth",
        "weak_label": "weak",
        "weak_labels": "weak",
        "pseudo_label": "pseudo",
        "pseudo_labels": "pseudo",
    }
    return aliases.get(text, text)


def _scores_by_seed(rows: list[dict[str, Any]], task: str, split: str, model: str) -> dict[int, float]:
    return {
        int(row["seed"]): float(row["score"])
        for row in rows
        if str(row["task"]) == task and str(row["split"]) == split and str(row["model"]) == model
    }


def _expected_paired_deltas(
    summary: dict[str, Any],
    task: str,
    split: str,
    full_model: str,
    baseline_model: str,
    higher_is_better: bool,
) -> list[float]:
    appendix = summary.get("per_seed_appendix", [])
    if not isinstance(appendix, list):
        return []
    full_scores: dict[int, float] = {}
    baseline_scores: dict[int, float] = {}
    for row in appendix:
        if not isinstance(row, dict):
            continue
        if str(row.get("task")) != task or str(row.get("split")) != split:
            continue
        seed = _safe_int(row.get("seed"))
        score = _finite_float(row.get("score"))
        if seed is None or score is None:
            continue
        model = str(row.get("model"))
        if model == full_model:
            full_scores[seed] = score
        elif model == baseline_model:
            baseline_scores[seed] = score
    return [
        _directional_delta(full_scores[seed], baseline_scores[seed], higher_is_better)
        for seed in sorted(set(full_scores) & set(baseline_scores))
    ]


def _appendix_scores(
    summary: dict[str, Any],
    task: str,
    split: str,
    model: str,
) -> list[tuple[int, float]]:
    appendix = summary.get("per_seed_appendix", [])
    if not isinstance(appendix, list):
        return []
    scores: dict[int, float] = {}
    for row in appendix:
        if not isinstance(row, dict):
            continue
        if str(row.get("task")) != task or str(row.get("split")) != split or str(row.get("model")) != model:
            continue
        seed = _safe_int(row.get("seed"))
        score = _finite_float(row.get("score"))
        if seed is None or score is None:
            continue
        scores[seed] = score
    return sorted(scores.items())


def _higher_is_better(
    main_table: dict[str, dict[str, dict[str, dict[str, Any]]]],
    task: str,
    split: str,
    model: str,
) -> bool:
    return bool(main_table.get(task, {}).get(split, {}).get(model, {}).get("higher_is_better", True))


def _directional_delta(full_score: float, baseline_score: float, higher_is_better: bool) -> float:
    if higher_is_better:
        return full_score - baseline_score
    return baseline_score - full_score


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _finite_ci95(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = _finite_float(value[0])
    high = _finite_float(value[1])
    if low is None or high is None:
        return None
    return (low, high)


def _finite_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    values: list[float] = []
    for item in value:
        numeric = _finite_float(item)
        if numeric is None:
            return None
        values.append(numeric)
    return values


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
