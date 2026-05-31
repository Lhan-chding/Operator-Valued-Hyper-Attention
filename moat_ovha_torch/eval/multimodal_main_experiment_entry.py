from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements


REGION_TEXT_TOPCONF_CHECKS = (
    "full_beats_same_feature_baseline",
    "no_cato_drops",
    "robustness_passes",
    "rceo_reliability_calibrated",
)
SENTIMENT_TOPCONF_CHECKS = (
    "full_beats_lmf_or_mult_baseline",
    "robustness_passes",
    "rceo_reliability_calibrated",
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
    if report.get("passed") is not True:
        errors.append(f"{label} gate must pass before top-conference main experiments")
    reasons = report.get("reasons", [])
    if report.get("passed") is True and isinstance(reasons, (list, tuple)) and reasons:
        errors.append(f"{label} gate reasons must be empty when passed is true")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        errors.append(f"{label} gate report must include checks")
        return
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
        warnings.extend(f"{name}: {warning}" for warning in report.warnings)
