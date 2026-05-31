from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
from moat_ovha_torch.eval.multimodal_public_entry import (
    REGION_TEXT_TASK_TYPES,
    SENTIMENT_EMOTION_TASK_TYPES,
    validate_public_entry_requirements,
)


REGION_TEXT_TOPCONF_CHECKS = (
    "full_beats_same_feature_baseline",
    "full_beats_required_strong_baselines",
    "no_cato_drops",
    "cato_router_load_high",
    "alignment_entropy_improves",
    "cato_top_alignment_accuracy_high",
    "grounding_accuracy_improves_with_entropy",
    "rceo_visual_stress_router_shift",
    "step14_public_diagnostics",
    "rceo_reliability_calibrated",
    "robustness_passes",
)
SENTIMENT_TOPCONF_CHECKS = (
    "full_beats_same_feature_baseline",
    "full_beats_lmf_or_mult_baseline",
    "no_lrio_drops",
    "no_spo_drops",
    "no_rceo_drops",
    "lrio_router_load_high",
    "spo_router_load_high",
    "lrio_rank_entropy_present",
    "spo_prototype_entropy_present",
    "spo_top_prototype_differentiates",
    "step14_public_diagnostics",
    "rceo_reliability_calibrated",
    "robustness_passes",
)


@dataclass(frozen=True)
class CacheValidationTarget:
    layout: MultimodalCacheLayout
    splits: tuple[str, ...]


@dataclass(frozen=True)
class TopConfMainExperimentEntryReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_topconf_main_experiment_entry(
    *,
    controlled_report: dict[str, Any] | None,
    region_gate_report: dict[str, Any] | None,
    sentiment_gate_report: dict[str, Any] | None,
    cache_targets: Mapping[str, CacheValidationTarget],
) -> TopConfMainExperimentEntryReport:
    errors: list[str] = []
    warnings: list[str] = []

    for task_type in ("phrase_region_grounding", "sentiment_emotion"):
        report = validate_public_entry_requirements(task_type, controlled_report)
        if not report.ok:
            errors.extend(f"controlled entry {task_type}: {error}" for error in report.errors)
        warnings.extend(f"controlled entry {task_type}: {warning}" for warning in report.warnings)

    _require_public_gate_report(
        "region_text_public",
        region_gate_report,
        REGION_TEXT_TOPCONF_CHECKS,
        errors,
    )
    _require_public_gate_report(
        "sentiment_emotion_public",
        sentiment_gate_report,
        SENTIMENT_TOPCONF_CHECKS,
        errors,
    )
    _validate_all_cache_targets(cache_targets, errors, warnings)

    return TopConfMainExperimentEntryReport(ok=not errors, errors=errors, warnings=warnings)


def _require_public_gate_report(
    label: str,
    report: dict[str, Any] | None,
    required_checks: tuple[str, ...],
    errors: list[str],
) -> None:
    if not isinstance(report, dict):
        errors.append(f"{label} gate report is required before top-conference main experiments")
        return
    if report.get("name") != label:
        errors.append(f"{label} gate report name mismatch")
    if report.get("passed") is not True:
        errors.append(f"{label} gate must pass before top-conference main experiments")
    reasons = report.get("reasons", [])
    if report.get("passed") is True and isinstance(reasons, (list, tuple)) and reasons:
        errors.append(f"{label} gate reasons must be empty when passed is true")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        errors.append(f"{label} gate report must include checks")
        return
    for check_name, check in sorted(checks.items()):
        if not isinstance(check, dict) or check.get("passed") is not True:
            errors.append(f"{label} gate contains failed check: {check_name}")
    for check_name in required_checks:
        check = checks.get(check_name)
        if not isinstance(check, dict) or check.get("passed") is not True:
            errors.append(f"{label} required check did not pass: {check_name}")


def _validate_all_cache_targets(
    cache_targets: Mapping[str, CacheValidationTarget],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(cache_targets, Mapping) or not cache_targets:
        errors.append("at least one data cache validation target is required for top-conference main experiments")
        return
    observed_region_text_cache = False
    observed_sentiment_cache = False
    for name, target in sorted(cache_targets.items()):
        if not isinstance(target, CacheValidationTarget):
            errors.append(f"data cache validation target must be CacheValidationTarget: {name}")
            continue
        if not target.splits:
            errors.append(f"data cache validation target must include splits: {name}")
            continue
        report = validate_cache_layout(target.layout, splits=target.splits)
        if not report.ok:
            errors.append(f"data cache validation failed for {name}")
            errors.extend(report.errors)
        else:
            tasks = _cache_data_card_tasks(target.layout)
            observed_region_text_cache = observed_region_text_cache or bool(tasks & REGION_TEXT_TASK_TYPES)
            observed_sentiment_cache = observed_sentiment_cache or bool(tasks & SENTIMENT_EMOTION_TASK_TYPES)
        warnings.extend(f"{name}: {warning}" for warning in report.warnings)
    if not observed_region_text_cache:
        errors.append("top-conference main experiments require at least one region-text data cache")
    if not observed_sentiment_cache:
        errors.append("top-conference main experiments require at least one sentiment/emotion data cache")


def _cache_data_card_tasks(layout: MultimodalCacheLayout) -> set[str]:
    try:
        payload = json.loads((layout.root / "data_card.json").read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, dict):
        return set()
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return set()
    return {str(task) for task in tasks if str(task).strip()}
