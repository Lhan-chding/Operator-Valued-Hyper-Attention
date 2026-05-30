from __future__ import annotations

from typing import Any


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
        "no_cato_drops": _ablation_drop(statistics_summary, task, split, full_model, no_cato_score, "no-CATO"),
        "cato_router_load_high": _router_load_high(diagnostics_rows, "CATO", minimum=0.35),
        "alignment_entropy_improves": _entropy_improves(diagnostics_rows, "CATO", "alignment_entropy"),
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
        "no_lrio_drops": _ablation_drop(statistics_summary, task, split, full_model, ablation_scores.get("ovha_no_lrio"), "no-LRIO"),
        "no_spo_drops": _ablation_drop(statistics_summary, task, split, full_model, ablation_scores.get("ovha_no_spo"), "no-SPO"),
        "no_rceo_drops": _ablation_drop(statistics_summary, task, split, full_model, ablation_scores.get("ovha_no_rceo"), "no-RCEO"),
        "lrio_router_load_high": _router_load_high(diagnostics_rows, "LRIO", minimum=0.25),
        "spo_router_load_high": _router_load_high(diagnostics_rows, "SPO", minimum=0.20),
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
    if full is None or baseline is None:
        return {"passed": False, "reason": "full or same-feature baseline score missing"}
    passed = full > baseline
    return {
        "passed": passed,
        "value": full - baseline,
        "reason": "full model does not beat same-feature baseline" if not passed else "",
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
    passed = full > float(ablation_score)
    return {
        "passed": passed,
        "value": full - float(ablation_score),
        "reason": f"{ablation_name} ablation does not drop" if not passed else "",
    }


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
    passed = bool(summary.get("full_drop_less_than_baseline")) and bool(summary.get("rceo_reliability_monotonic"))
    return {
        "passed": passed,
        "reason": "robustness summary does not show lower drop and monotonic RCEO reliability" if not passed else "",
    }


def _model_mean(summary: dict[str, Any], task: str, split: str, model: str) -> float | None:
    try:
        return float(summary["main_table"][task][split][model]["mean"])
    except KeyError:
        return None


def _candidate_diag_value(rows: list[dict[str, Any]], setting: str, candidate: str, key: str) -> float | None:
    for row in rows:
        if row.get("setting") != setting:
            continue
        diagnostics = row.get("candidate_diagnostics", {}) or {}
        if candidate in diagnostics and key in diagnostics[candidate]:
            return float(diagnostics[candidate][key])
    return None
