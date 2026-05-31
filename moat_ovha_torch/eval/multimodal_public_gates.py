from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task


_SENTIMENT_ANCHOR_BASELINES = ("tfn_lmf", "mult_style_crossmodal_transformer")


def evaluate_region_text_gate(
    *,
    statistics_summary: dict[str, Any],
    diagnostics_rows: list[dict[str, Any]],
    no_cato_score: float | None,
    task: str,
    split: str,
    full_model: str = "ovha_full",
    baseline_model: str = "cross_attention_transformer",
) -> dict[str, Any]:
    checks = {
        "full_beats_same_feature_baseline": _full_beats_baseline(statistics_summary, task, split, full_model, baseline_model),
        "full_beats_required_strong_baselines": _full_beats_required_strong_baselines(
            statistics_summary,
            task,
            split,
            full_model,
        ),
        "no_cato_drops": _ablation_drop(statistics_summary, task, split, full_model, no_cato_score, "no-CATO"),
        "cato_router_load_high": _router_load_high(diagnostics_rows, "CATO", minimum=0.35),
        "alignment_entropy_improves": _entropy_improves(diagnostics_rows, "CATO", "alignment_entropy"),
        "cato_top_alignment_accuracy_high": _candidate_diag_at_least(
            diagnostics_rows,
            "clean",
            "CATO",
            "top_alignment_accuracy",
            minimum=0.50,
        ),
        "grounding_accuracy_improves_with_entropy": _grounding_accuracy_improves_with_entropy(diagnostics_rows),
        "rceo_visual_stress_router_shift": _rceo_visual_stress_router_shift(diagnostics_rows),
    }
    return _gate_report("region_text_public", checks)


def evaluate_sentiment_gate(
    *,
    statistics_summary: dict[str, Any],
    diagnostics_rows: list[dict[str, Any]],
    ablation_scores: dict[str, float | None],
    robustness_summary: dict[str, Any],
    task: str,
    split: str,
    full_model: str = "ovha_full",
    baseline_model: str = "cross_attention_transformer",
) -> dict[str, Any]:
    checks = {
        "full_beats_same_feature_baseline": _full_beats_baseline(statistics_summary, task, split, full_model, baseline_model),
        "full_beats_lmf_or_mult_baseline": _full_beats_any_required_baseline(
            statistics_summary,
            task,
            split,
            full_model,
            _SENTIMENT_ANCHOR_BASELINES,
            "full model must beat at least one required sentiment baseline: tfn_lmf or mult_style_crossmodal_transformer",
        ),
        "no_lrio_drops": _ablation_drop(statistics_summary, task, split, full_model, ablation_scores.get("ovha_no_lrio"), "no-LRIO"),
        "no_spo_drops": _ablation_drop(statistics_summary, task, split, full_model, ablation_scores.get("ovha_no_spo"), "no-SPO"),
        "no_rceo_drops": _ablation_drop(statistics_summary, task, split, full_model, ablation_scores.get("ovha_no_rceo"), "no-RCEO"),
        "lrio_router_load_high": _router_load_high(diagnostics_rows, "LRIO", minimum=0.25),
        "spo_router_load_high": _router_load_high(diagnostics_rows, "SPO", minimum=0.20),
        "lrio_rank_entropy_present": _candidate_diag_positive(diagnostics_rows, "LRIO", "rank_entropy"),
        "spo_prototype_entropy_present": _candidate_diag_positive(diagnostics_rows, "SPO", "prototype_entropy"),
        "spo_top_prototype_differentiates": _spo_top_prototype_differentiates(diagnostics_rows),
        "rceo_reliability_calibrated": _rceo_reliability_calibrated(robustness_summary),
        "robustness_passes": _robustness_passes(robustness_summary),
    }
    return _gate_report("sentiment_emotion_public", checks)


def _gate_report(name: str, checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reasons = [check["reason"] for check in checks.values() if not check["passed"]]
    return {"name": name, "passed": not reasons, "checks": checks, "reasons": reasons}


def _full_beats_baseline(
    summary: dict[str, Any],
    task: str,
    split: str,
    full_model: str,
    baseline_model: str,
) -> dict[str, Any]:
    full = _model_mean(summary, task, split, full_model)
    baseline = _model_mean(summary, task, split, baseline_model)
    reasons = _statistical_evidence_reasons(summary, task, split, full_model, baseline_model)
    if full is None or baseline is None:
        reasons.append("full or same-feature baseline score missing")
        return {"passed": False, "reason": "; ".join(reasons)}
    higher_is_better = _higher_is_better(summary, task, split, full_model)
    improvement = _directional_improvement(full, baseline, higher_is_better)
    if improvement <= 0.0:
        reasons.append("full model does not beat same-feature baseline")
    passed = not reasons
    return {
        "passed": passed,
        "value": improvement,
        "reason": "; ".join(reasons),
    }


def _ablation_drop(
    summary: dict[str, Any],
    task: str,
    split: str,
    full_model: str,
    ablation_score: float | None,
    ablation_name: str,
) -> dict[str, Any]:
    full = _model_mean(summary, task, split, full_model)
    if ablation_score is None:
        return {"passed": False, "reason": f"{ablation_name} ablation score missing"}
    if full is None:
        return {"passed": False, "reason": "full model score missing"}
    higher_is_better = _higher_is_better(summary, task, split, full_model)
    improvement = _directional_improvement(full, float(ablation_score), higher_is_better)
    passed = improvement > 0.0
    return {
        "passed": passed,
        "value": improvement,
        "reason": f"{ablation_name} ablation does not drop" if not passed else "",
    }


def _full_beats_required_strong_baselines(
    summary: dict[str, Any],
    task: str,
    split: str,
    full_model: str,
) -> dict[str, Any]:
    full = _model_mean(summary, task, split, full_model)
    if full is None:
        return {"passed": False, "reason": "full model score missing"}
    higher_is_better = _higher_is_better(summary, task, split, full_model)
    reasons: list[str] = []
    values: dict[str, float] = {}
    for baseline_model in _required_strong_baselines_for_gate(task):
        baseline = _model_mean(summary, task, split, baseline_model)
        if baseline is None:
            reasons.append(f"required same-feature baseline missing: {baseline_model}")
            continue
        improvement = _directional_improvement(full, baseline, higher_is_better)
        values[baseline_model] = improvement
        if improvement <= 0.0:
            reasons.append(f"full model does not beat required same-feature baseline: {baseline_model}")
    return {
        "passed": not reasons,
        "value": values,
        "reason": "; ".join(reasons),
    }


def _full_beats_any_required_baseline(
    summary: dict[str, Any],
    task: str,
    split: str,
    full_model: str,
    baseline_models: tuple[str, ...],
    failure_reason: str,
) -> dict[str, Any]:
    full = _model_mean(summary, task, split, full_model)
    if full is None:
        return {"passed": False, "reason": "full model score missing"}
    higher_is_better = _higher_is_better(summary, task, split, full_model)
    missing: list[str] = []
    values: dict[str, float] = {}
    for baseline_model in baseline_models:
        baseline = _model_mean(summary, task, split, baseline_model)
        if baseline is None:
            missing.append(baseline_model)
            continue
        values[baseline_model] = _directional_improvement(full, baseline, higher_is_better)
    best_improvement = max(values.values()) if values else None
    passed = best_improvement is not None and best_improvement > 0.0
    reasons = [] if passed else [failure_reason]
    if missing and not passed:
        reasons.append(f"required sentiment baseline missing: {', '.join(missing)}")
    return {
        "passed": passed,
        "value": values,
        "reason": "; ".join(reasons),
    }


def _required_strong_baselines_for_gate(task: str) -> tuple[str, ...]:
    if task == "phrase_region_grounding":
        return ("modality_expert_moe",)
    return ()


def _router_load_high(rows: list[dict[str, Any]], candidate: str, minimum: float) -> dict[str, Any]:
    loads = [
        float((row.get("router_load_by_candidate", {}) or {}).get(candidate, 0.0))
        for row in rows
        if row.get("setting", "clean") == "clean"
    ]
    value = max(loads) if loads else 0.0
    passed = value >= minimum
    return {
        "passed": passed,
        "value": value,
        "threshold": minimum,
        "reason": f"{candidate} router load is not elevated on relevant public samples" if not passed else "",
    }


def _candidate_diag_positive(rows: list[dict[str, Any]], candidate: str, key: str) -> dict[str, Any]:
    values = [
        value
        for row in rows
        if row.get("setting", "clean") == "clean"
        for value in [_candidate_diag_value_from_row(row, candidate, key)]
        if value is not None
    ]
    best_value = max(values) if values else None
    passed = best_value is not None and best_value > 0.0
    return {
        "passed": passed,
        "value": best_value,
        "reason": f"{candidate} {_display_key(key)} diagnostic missing or non-positive" if not passed else "",
    }


def _candidate_diag_at_least(
    rows: list[dict[str, Any]],
    setting: str,
    candidate: str,
    key: str,
    *,
    minimum: float,
) -> dict[str, Any]:
    values = [
        value
        for row in rows
        if row.get("setting", "clean") == setting
        for value in [_candidate_diag_value_from_row(row, candidate, key)]
        if value is not None
    ]
    best_value = max(values) if values else None
    passed = best_value is not None and best_value >= minimum
    return {
        "passed": passed,
        "value": best_value,
        "threshold": minimum,
        "reason": f"{candidate} {_display_key(key)} diagnostic missing or below threshold" if not passed else "",
    }


def _grounding_accuracy_improves_with_entropy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean_entropy = _candidate_diag_value(rows, "clean", "CATO", "alignment_entropy")
    ablated_entropy = _candidate_diag_value(rows, "no_cato", "CATO", "alignment_entropy")
    clean_accuracy = _candidate_diag_value(rows, "clean", "CATO", "grounding_accuracy")
    ablated_accuracy = _candidate_diag_value(rows, "no_cato", "CATO", "grounding_accuracy")
    if None in (clean_entropy, ablated_entropy, clean_accuracy, ablated_accuracy):
        return {
            "passed": False,
            "reason": "grounding accuracy must improve as CATO alignment entropy decreases",
        }
    entropy_delta = float(ablated_entropy) - float(clean_entropy)
    accuracy_delta = float(clean_accuracy) - float(ablated_accuracy)
    passed = entropy_delta > 0.0 and accuracy_delta > 0.0
    return {
        "passed": passed,
        "value": {"alignment_entropy_drop": entropy_delta, "grounding_accuracy_gain": accuracy_delta},
        "reason": "grounding accuracy must improve as CATO alignment entropy decreases" if not passed else "",
    }


def _rceo_visual_stress_router_shift(
    rows: list[dict[str, Any]],
    *,
    min_l1_shift: float = 0.10,
) -> dict[str, Any]:
    clean_rows = [row for row in rows if row.get("setting", "clean") == "clean"]
    stress_rows = [row for row in rows if str(row.get("setting", "")) in _VISUAL_STRESS_SETTINGS]
    for clean in clean_rows:
        for stress in stress_rows:
            shift = _router_l1_shift(clean, stress)
            if shift < min_l1_shift:
                continue
            reliability_shift = _reliability_decreases(clean, stress)
            corruption_response = _candidate_diag_value_from_row(stress, "RCEO", "corruption_response")
            if reliability_shift or (corruption_response is not None and corruption_response > 0.0):
                return {
                    "passed": True,
                    "value": {
                        "l1_router_shift": shift,
                        "reliability_decreases": reliability_shift,
                        "rceo_corruption_response": corruption_response,
                    },
                    "threshold": min_l1_shift,
                    "reason": "",
                }
    return {
        "passed": False,
        "threshold": min_l1_shift,
        "reason": "RCEO visual stress router shift missing or below threshold",
    }


def _spo_top_prototype_differentiates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("setting", "clean") != "clean":
            continue
        diagnostics = row.get("candidate_diagnostics", {}) or {}
        spo = diagnostics.get("SPO", {}) if isinstance(diagnostics, dict) else {}
        if not isinstance(spo, dict):
            continue
        if _top_prototype_map_differentiates(spo.get("top_prototype_by_class")):
            return {"passed": True, "value": "top_prototype_by_class", "reason": ""}
        if _prototype_load_map_differentiates(spo.get("prototype_load_by_class")):
            return {"passed": True, "value": "prototype_load_by_class", "reason": ""}
    return {
        "passed": False,
        "reason": "SPO top prototype differentiation missing or collapsed across emotion classes",
    }


def _rceo_reliability_calibrated(
    summary: dict[str, Any],
    *,
    max_ece: float = 0.10,
    min_bin_count: int = 3,
) -> dict[str, Any]:
    calibration = summary.get("rceo_reliability_calibration")
    if not isinstance(calibration, dict):
        return {"passed": False, "reason": "RCEO reliability calibration missing"}
    ece = _finite_float(calibration.get("ece", calibration.get("expected_calibration_error")))
    if ece is None:
        return {"passed": False, "reason": "RCEO reliability calibration missing ECE"}
    bin_count = _safe_int(calibration.get("bin_count", calibration.get("bins")))
    if bin_count is None or bin_count < min_bin_count:
        return {
            "passed": False,
            "value": ece,
            "threshold": max_ece,
            "reason": f"RCEO reliability calibration requires at least {min_bin_count} bins",
        }
    passed = ece <= max_ece
    return {
        "passed": passed,
        "value": ece,
        "threshold": max_ece,
        "bin_count": bin_count,
        "reason": "RCEO reliability calibration ECE exceeds threshold" if not passed else "",
    }


def _entropy_improves(rows: list[dict[str, Any]], candidate: str, key: str) -> dict[str, Any]:
    clean = _candidate_diag_value(rows, "clean", candidate, key)
    ablated = _candidate_diag_value(rows, "no_cato", candidate, key)
    if clean is None or ablated is None:
        return {"passed": False, "reason": f"{candidate} {key} diagnostic missing"}
    passed = clean < ablated
    return {
        "passed": passed,
        "value": ablated - clean,
        "reason": f"{candidate} {key} does not improve in full model" if not passed else "",
    }


def _robustness_passes(summary: dict[str, Any]) -> dict[str, Any]:
    ablations = summary.get("required_ablation_degradation", {})
    ablations_pass = bool(ablations.get("passed"))
    coverage = summary.get("required_stress_coverage", {})
    coverage_pass = bool(coverage.get("passed"))
    passed = (
        bool(summary.get("full_drop_less_than_baseline"))
        and bool(summary.get("rceo_reliability_monotonic"))
        and ablations_pass
        and coverage_pass
    )
    ablation_reasons = "; ".join(str(reason) for reason in ablations.get("reasons", ()) if reason)
    coverage_reasons = "; ".join(str(reason) for reason in coverage.get("reasons", ()) if reason)
    reason_parts = [
        "robustness summary does not show lower drop, monotonic RCEO reliability, and required ablation degradation",
    ]
    if not coverage_pass:
        reason_parts.append("robustness stress family coverage missing")
    if ablation_reasons:
        reason_parts.append(ablation_reasons)
    if coverage_reasons:
        reason_parts.append(coverage_reasons)
    return {
        "passed": passed,
        "reason": "; ".join(reason_parts) if not passed else "",
    }


def _model_mean(summary: dict[str, Any], task: str, split: str, model: str) -> float | None:
    try:
        return _finite_float(summary["main_table"][task][split][model]["mean"])
    except KeyError:
        return None


def _higher_is_better(summary: dict[str, Any], task: str, split: str, model: str) -> bool:
    try:
        value = summary["main_table"][task][split][model].get("higher_is_better", True)
    except KeyError:
        return True
    return value if isinstance(value, bool) else True


def _directional_improvement(full: float, comparison: float, higher_is_better: bool) -> float:
    if higher_is_better:
        return full - comparison
    return comparison - full


def _statistical_evidence_reasons(
    summary: dict[str, Any],
    task: str,
    split: str,
    full_model: str,
    baseline_model: str,
) -> list[str]:
    reasons: list[str] = []
    main_models = (((summary.get("main_table", {}) or {}).get(task, {}) or {}).get(split, {}) or {})
    direction_reasons, main_metric_direction = _metric_direction_reasons(main_models)
    reasons.extend(direction_reasons)
    full_seed_count = _seed_count(main_models.get(full_model, {}))
    baseline_seed_count = _seed_count(main_models.get(baseline_model, {}))
    if full_seed_count < 3 or baseline_seed_count < 3:
        reasons.append("full and baseline comparison requires at least 3 seeds")
    reasons.extend(_main_table_reporting_reasons(main_models, full_model, full_seed_count))
    reasons.extend(_main_table_reporting_reasons(main_models, baseline_model, baseline_seed_count))
    required_baselines = _required_baselines_for_summary(task)
    for required_baseline in required_baselines:
        if required_baseline not in main_models:
            reasons.append(f"statistics summary missing required same-feature baseline: {required_baseline}")
            continue
        reasons.extend(_main_table_reporting_reasons(main_models, required_baseline, _seed_count(main_models.get(required_baseline, {}))))
    reporting_models = tuple(dict.fromkeys((full_model, baseline_model, *required_baselines)))
    reasons.extend(_summary_reporting_metadata_reasons(summary, reporting_models, main_models, task))
    paired = (((summary.get("paired_tests", {}) or {}).get(task, {}) or {}).get(split, {}) or {})
    if not isinstance(paired, dict) or not paired:
        reasons.append("paired comparison missing")
        return reasons
    common_seed_count = _safe_int(paired.get("common_seed_count")) or 0
    if common_seed_count < 3:
        reasons.append("paired comparison requires at least 3 common seeds")
    if common_seed_count != min(full_seed_count, baseline_seed_count):
        reasons.append("paired comparison common_seed_count must cover full and baseline main_table seeds")
    for key in ("metric_direction", "mean_delta", "paired_permutation_p", "paired_bootstrap_ci95"):
        if key not in paired:
            reasons.append(f"paired comparison missing {key}")
    if "metric_direction" in paired:
        paired_direction = paired.get("metric_direction")
        if paired_direction not in {"higher_is_better", "lower_is_better"}:
            reasons.append("paired comparison metric_direction must be higher_is_better or lower_is_better")
        elif main_metric_direction is not None and paired_direction != main_metric_direction:
            reasons.append("paired comparison metric_direction disagrees with main_table higher_is_better")
    if "mean_delta" in paired:
        mean_delta = _finite_float(paired.get("mean_delta"))
        if mean_delta is None:
            reasons.append("paired comparison mean_delta must be a finite number")
        elif mean_delta <= 0.0:
            reasons.append("paired comparison mean_delta must be positive for claimed improvement")
        else:
            expected_delta = _expected_main_table_delta(main_models, full_model, baseline_model)
            if expected_delta is not None and not math.isclose(mean_delta, expected_delta, rel_tol=1e-9, abs_tol=1e-9):
                reasons.append("paired comparison mean_delta disagrees with main_table mean delta")
    if "paired_permutation_p" in paired:
        permutation_p = _finite_float(paired.get("paired_permutation_p"))
        if permutation_p is None or permutation_p < 0.0 or permutation_p > 1.0:
            reasons.append("paired comparison paired_permutation_p must be a finite probability")
    if "paired_bootstrap_ci95" in paired:
        bootstrap_ci = _finite_interval(paired.get("paired_bootstrap_ci95"))
        if bootstrap_ci is None:
            reasons.append("paired comparison paired_bootstrap_ci95 must be a finite length-2 interval")
        elif bootstrap_ci[0] > bootstrap_ci[1]:
            reasons.append("paired comparison paired_bootstrap_ci95 lower bound must not exceed upper bound")
        elif bootstrap_ci[0] <= 0.0:
            reasons.append("paired comparison bootstrap CI must be strictly positive for claimed improvement")
    return reasons


def _expected_main_table_delta(
    main_models: Any,
    full_model: str,
    baseline_model: str,
) -> float | None:
    if not isinstance(main_models, dict):
        return None
    full_row = main_models.get(full_model)
    baseline_row = main_models.get(baseline_model)
    if not isinstance(full_row, dict) or not isinstance(baseline_row, dict):
        return None
    full_mean = _finite_float(full_row.get("mean"))
    baseline_mean = _finite_float(baseline_row.get("mean"))
    higher_is_better = full_row.get("higher_is_better")
    if full_mean is None or baseline_mean is None or not isinstance(higher_is_better, bool):
        return None
    return _directional_improvement(full_mean, baseline_mean, higher_is_better)


def _metric_direction_reasons(main_models: Any) -> tuple[list[str], str | None]:
    if not isinstance(main_models, dict):
        return ["statistics summary main_table must be keyed by model"], None
    reasons: list[str] = []
    directions: set[bool] = set()
    for model, row in main_models.items():
        if not isinstance(row, dict):
            continue
        if "higher_is_better" not in row:
            reasons.append(f"{model} main table missing higher_is_better")
            continue
        higher_is_better = row["higher_is_better"]
        if not isinstance(higher_is_better, bool):
            reasons.append(f"{model} higher_is_better must be boolean")
            continue
        directions.add(higher_is_better)
    if len(directions) > 1:
        reasons.append("statistics summary higher_is_better must be consistent across models")
        return reasons, None
    if len(directions) == 1:
        higher_is_better = next(iter(directions))
        return reasons, "higher_is_better" if higher_is_better else "lower_is_better"
    return reasons, None


def _main_table_reporting_reasons(main_models: dict[str, Any], model: str, seed_count: int) -> list[str]:
    reasons: list[str] = []
    row = main_models.get(model)
    if not isinstance(row, dict):
        reasons.append(f"{model} main table row missing")
        return reasons
    if _finite_float(row.get("mean")) is None:
        reasons.append(f"{model} main table mean must be finite number")
    std = _finite_float(row.get("std"))
    if "std" not in row:
        reasons.append(f"{model} main table missing std")
    elif std is None or std < 0.0:
        reasons.append(f"{model} main table std must be finite non-negative number")
    if "ci95" not in row:
        reasons.append(f"{model} main table missing ci95")
    else:
        ci95 = _finite_interval(row.get("ci95"))
        if ci95 is None:
            reasons.append(f"{model} main table ci95 must be finite length-2 interval")
        elif ci95[0] > ci95[1]:
            reasons.append(f"{model} main table ci95 lower bound must not exceed upper bound")
    per_seed = row.get("per_seed_scores", row.get("per_seed"))
    if not isinstance(per_seed, list) or len(per_seed) < seed_count or seed_count < 3:
        reasons.append(f"{model} main table missing raw per-seed scores")
    elif any(_finite_float(value) is None for value in per_seed):
        reasons.append(f"{model} main table per-seed scores must be finite numbers")
    return reasons


def _summary_reporting_metadata_reasons(
    summary: dict[str, Any],
    models: tuple[str, ...],
    main_models: Any,
    task: str,
) -> list[str]:
    metadata = summary.get("reporting_metadata")
    if not isinstance(metadata, dict):
        return [
            "reporting metadata missing parameter_count",
            "reporting metadata missing training_steps",
            "reporting metadata missing frozen_feature_versions",
            "reporting metadata missing hardware",
            "reporting metadata missing wall_clock_summary",
            "reporting metadata missing per_seed_table",
        ]
    reasons: list[str] = []
    _require_model_metadata(metadata, "parameter_count", models, reasons, require_positive_integer=True)
    _require_model_metadata(metadata, "training_steps", models, reasons, require_positive_integer=True)
    for key in ("frozen_feature_versions", "hardware", "wall_clock_summary", "per_seed_table"):
        value = metadata.get(key)
        if _is_empty_reporting_value(value):
            reasons.append(f"reporting metadata missing {key}")
    reasons.extend(_frozen_feature_versions_reasons(metadata.get("frozen_feature_versions"), task))
    reasons.extend(_hardware_metadata_reasons(metadata.get("hardware")))
    reasons.extend(_wall_clock_summary_reasons(metadata.get("wall_clock_summary")))
    per_seed_table = metadata.get("per_seed_table")
    if isinstance(per_seed_table, list):
        reasons.extend(_reporting_per_seed_table_reasons(per_seed_table, models, main_models))
    return reasons


def _frozen_feature_versions_reasons(value: Any, task: str) -> list[str]:
    if _is_empty_reporting_value(value):
        return []
    if not isinstance(value, dict):
        return ["reporting metadata frozen_feature_versions must be keyed by modality"]
    reasons: list[str] = []
    for modality in _required_feature_modalities(task):
        if _is_empty_reporting_value(value.get(modality)):
            reasons.append(f"reporting metadata frozen_feature_versions missing modality: {modality}")
    return reasons


def _required_feature_modalities(task: str) -> tuple[str, ...]:
    if task == "phrase_region_grounding":
        return ("text", "region")
    if task == "sentiment_emotion":
        return ("text", "audio", "vision")
    return ()


def _hardware_metadata_reasons(value: Any) -> list[str]:
    if _is_empty_reporting_value(value):
        return []
    if not isinstance(value, dict) or _is_empty_reporting_value(value.get("accelerator")):
        return ["reporting metadata hardware must include accelerator"]
    return []


def _wall_clock_summary_reasons(value: Any) -> list[str]:
    if _is_empty_reporting_value(value):
        return []
    if not isinstance(value, dict):
        return ["reporting metadata wall_clock_summary must include positive wall_clock_hours"]
    hours = _finite_float(value.get("wall_clock_hours"))
    if hours is None or hours <= 0.0:
        return ["reporting metadata wall_clock_summary must include positive wall_clock_hours"]
    return []


def _reporting_per_seed_table_reasons(
    per_seed_table: list[Any],
    models: tuple[str, ...],
    main_models: Any,
) -> list[str]:
    reasons: list[str] = []
    valid_seeds_by_model: dict[str, set[int]] = {}
    invalid_seed_models: set[str] = set()
    invalid_score_models: set[str] = set()
    for row in per_seed_table:
        if not isinstance(row, dict):
            continue
        model_value = row.get("model")
        if not model_value:
            continue
        model = str(model_value)
        seed = _safe_int(row.get("seed"))
        score = _finite_float(row.get("score"))
        if seed is None:
            invalid_seed_models.add(model)
            continue
        if score is None:
            invalid_score_models.add(model)
            continue
        valid_seeds_by_model.setdefault(model, set()).add(seed)
    for model in sorted(invalid_seed_models):
        reasons.append(f"reporting metadata per_seed_table seed must be integer for model: {model}")
    for model in sorted(invalid_score_models):
        reasons.append(f"reporting metadata per_seed_table score must be finite for model: {model}")
    for model in models:
        seeds = valid_seeds_by_model.get(model, set())
        if not seeds:
            reasons.append(f"reporting metadata per_seed_table missing model: {model}")
            continue
        expected_seed_count = _seed_count(main_models.get(model, {})) if isinstance(main_models, dict) else 0
        if expected_seed_count > 0 and len(seeds) < expected_seed_count:
            reasons.append(f"reporting metadata per_seed_table must cover main_table seed_count for model: {model}")
    return reasons


def _require_model_metadata(
    metadata: dict[str, Any],
    key: str,
    models: tuple[str, ...],
    reasons: list[str],
    *,
    require_positive_integer: bool = False,
) -> None:
    value = metadata.get(key)
    if not isinstance(value, dict) or not value:
        reasons.append(f"reporting metadata missing {key}")
        return
    for model in models:
        if model not in value or _is_empty_reporting_value(value.get(model)):
            reasons.append(f"reporting metadata {key} missing model: {model}")
            continue
        if require_positive_integer and _positive_integer(value.get(model)) is None:
            reasons.append(f"reporting metadata {key} must be positive integer for model: {model}")


def _is_empty_reporting_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, list, dict, tuple, set)):
        return len(value) == 0
    return False


def _required_baselines_for_summary(task: str) -> tuple[str, ...]:
    try:
        return baseline_names_for_task(task)
    except ValueError:
        return ()


def _seed_count(values: Any) -> int:
    if not isinstance(values, dict):
        return 0
    try:
        return int(values.get("seed_count", 0))
    except (TypeError, ValueError):
        return 0


def _candidate_diag_value(rows: list[dict[str, Any]], setting: str, candidate: str, key: str) -> float | None:
    for row in rows:
        if row.get("setting") != setting:
            continue
        value = _candidate_diag_value_from_row(row, candidate, key)
        if value is not None:
            return value
    return None


def _candidate_diag_value_from_row(row: dict[str, Any], candidate: str, key: str) -> float | None:
    diagnostics = row.get("candidate_diagnostics", {}) or {}
    if not isinstance(diagnostics, dict):
        return None
    candidate_values = diagnostics.get(candidate)
    if not isinstance(candidate_values, dict) or key not in candidate_values:
        return None
    return _finite_float(candidate_values[key])


_VISUAL_STRESS_SETTINGS = {
    "corrupted_visual",
    "missing_visual",
    "visual_corruption",
    "visual_missing",
    "corrupted_regions",
    "missing_regions",
}


def _router_l1_shift(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_load = left.get("router_load_by_candidate", {}) or {}
    right_load = right.get("router_load_by_candidate", {}) or {}
    if not isinstance(left_load, dict) or not isinstance(right_load, dict):
        return 0.0
    names = set(left_load) | set(right_load)
    total = 0.0
    for name in names:
        left_value = _finite_float(left_load.get(name, 0.0))
        right_value = _finite_float(right_load.get(name, 0.0))
        if left_value is None or right_value is None:
            return 0.0
        total += abs(right_value - left_value)
    return total


def _reliability_decreases(clean: dict[str, Any], stress: dict[str, Any]) -> bool:
    clean_reliability = _finite_float(clean.get("rceo_reliability"))
    stress_reliability = _finite_float(stress.get("rceo_reliability"))
    return clean_reliability is not None and stress_reliability is not None and stress_reliability < clean_reliability


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        return int(text) if text.isdigit() and int(text) > 0 else None
    return None


def _finite_interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = _finite_float(value[0])
    high = _finite_float(value[1])
    if low is None or high is None:
        return None
    return (low, high)


def _top_prototype_map_differentiates(value: Any) -> bool:
    if not isinstance(value, Mapping) or len(value) < 2:
        return False
    prototypes = {str(prototype) for prototype in value.values()}
    return len(prototypes) >= 2


def _prototype_load_map_differentiates(value: Any) -> bool:
    if not isinstance(value, Mapping) or len(value) < 2:
        return False
    dominant = {_dominant_prototype(loads) for loads in value.values()}
    dominant.discard(None)
    return len(dominant) >= 2


def _dominant_prototype(loads: Any) -> str | None:
    if isinstance(loads, Mapping) and loads:
        scored = [(str(name), _finite_float(score)) for name, score in loads.items()]
        valid = [(name, score) for name, score in scored if score is not None]
        if not valid:
            return None
        return max(valid, key=lambda item: item[1])[0]
    if isinstance(loads, Sequence) and not isinstance(loads, (str, bytes)) and loads:
        scored = [(str(index), _finite_float(score)) for index, score in enumerate(loads)]
        valid = [(name, score) for name, score in scored if score is not None]
        if not valid:
            return None
        return max(valid, key=lambda item: item[1])[0]
    return None


def _display_key(key: str) -> str:
    return key.replace("_", " ")
