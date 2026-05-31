from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
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
TOPCONF_GATE_EVIDENCE_ARTIFACTS = ("statistics_summary", "diagnostics", "robustness_summary")
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_EXPECTED_GATE_TASK_TYPES = {
    "region_text_public": REGION_TEXT_TASK_TYPES,
    "sentiment_emotion_public": SENTIMENT_EMOTION_TASK_TYPES,
}


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
    if report.get("passed") is True and (not isinstance(reasons, (list, tuple)) or reasons):
        errors.append(f"{label} gate reasons must be an empty list when passed is true")
    _require_gate_evidence_artifacts(label, report.get("evidence_artifacts"), errors)
    checks = report.get("checks")
    if not isinstance(checks, dict):
        errors.append(f"{label} gate report must include checks")
        return
    for check_name, check in sorted(checks.items()):
        if not isinstance(check, dict) or check.get("passed") is not True:
            errors.append(f"{label} gate contains failed check: {check_name}")
            continue
        if not _empty_gate_reason(check.get("reason", "")):
            errors.append(f"{label} gate check {check_name} reason must be empty when passed is true")
    for check_name in required_checks:
        check = checks.get(check_name)
        if not isinstance(check, dict) or check.get("passed") is not True:
            errors.append(f"{label} required check did not pass: {check_name}")


def _require_gate_evidence_artifacts(label: str, evidence: Any, errors: list[str]) -> None:
    if not isinstance(evidence, Mapping):
        errors.append(f"{label} gate evidence_artifacts is required")
        return

    task = evidence.get("task")
    expected_tasks = _EXPECTED_GATE_TASK_TYPES.get(label, frozenset())
    if not isinstance(task, str) or task.strip() not in expected_tasks:
        errors.append(f"{label} gate evidence_artifacts task must match the public gate task family")
    if not _non_empty_text(evidence.get("split")):
        errors.append(f"{label} gate evidence_artifacts split must be a non-empty string")
    if not _non_empty_text(evidence.get("generated_by")):
        errors.append(f"{label} gate evidence_artifacts generated_by must identify the evaluator")

    for artifact_name in TOPCONF_GATE_EVIDENCE_ARTIFACTS:
        artifact = evidence.get(artifact_name)
        if artifact is None:
            errors.append(f"{label} gate evidence_artifacts missing artifact: {artifact_name}")
            continue
        _validate_artifact_descriptor(label, artifact_name, artifact, errors)

    raw_metrics = evidence.get("raw_metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        errors.append(f"{label} gate evidence_artifacts raw_metrics must be a non-empty list")
        return
    for index, artifact in enumerate(raw_metrics):
        _validate_artifact_descriptor(label, f"raw_metrics[{index}]", artifact, errors)


def _validate_artifact_descriptor(label: str, artifact_name: str, artifact: Any, errors: list[str]) -> None:
    if not isinstance(artifact, Mapping):
        errors.append(f"{label} gate evidence_artifacts {artifact_name} must include path and sha256")
        return
    path_value = artifact.get("path")
    if not _non_empty_text(path_value):
        errors.append(f"{label} gate evidence_artifacts {artifact_name}.path must be a non-empty string")
        artifact_path = None
    else:
        artifact_path = Path(path_value)
    sha256 = artifact.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX_RE.fullmatch(sha256):
        errors.append(f"{label} gate evidence_artifacts {artifact_name}.sha256 must be lowercase SHA-256")
        return
    if artifact_path is None:
        return
    if not artifact_path.exists():
        errors.append(f"{label} gate evidence_artifacts {artifact_name}.path does not exist")
        return
    if not artifact_path.is_file():
        errors.append(f"{label} gate evidence_artifacts {artifact_name}.path must point to a file")
        return
    if _sha256(artifact_path) != sha256:
        errors.append(f"{label} gate evidence_artifacts {artifact_name}.sha256 does not match file content")


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _empty_gate_reason(reason: Any) -> bool:
    if reason is None:
        return True
    if isinstance(reason, str):
        return not reason.strip()
    if isinstance(reason, (list, tuple)):
        return not reason
    return False
