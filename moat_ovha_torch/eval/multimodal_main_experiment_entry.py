from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
from moat_ovha_torch.eval.multimodal_public_entry import (
    REGION_TEXT_TASK_TYPES,
    SENTIMENT_EMOTION_TASK_TYPES,
    validate_public_entry_requirements,
)
from moat_ovha_torch.eval.multimodal_public_gates import (
    evaluate_region_text_gate,
    evaluate_sentiment_gate,
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

    artifact_paths: dict[str, Path] = {}
    for artifact_name in TOPCONF_GATE_EVIDENCE_ARTIFACTS:
        artifact = evidence.get(artifact_name)
        if artifact is None:
            errors.append(f"{label} gate evidence_artifacts missing artifact: {artifact_name}")
            continue
        artifact_path = _validate_artifact_descriptor(label, artifact_name, artifact, errors)
        if artifact_path is not None:
            artifact_paths[artifact_name] = artifact_path

    raw_metrics = evidence.get("raw_metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        errors.append(f"{label} gate evidence_artifacts raw_metrics must be a non-empty list")
        return
    raw_metric_paths: list[Path] = []
    for index, artifact in enumerate(raw_metrics):
        artifact_path = _validate_artifact_descriptor(label, f"raw_metrics[{index}]", artifact, errors)
        if artifact_path is not None:
            raw_metric_paths.append(artifact_path)
    if isinstance(task, str) and _non_empty_text(evidence.get("split")) and "statistics_summary" in artifact_paths:
        _validate_statistics_summary_content(
            label,
            artifact_paths["statistics_summary"],
            task.strip(),
            str(evidence["split"]).strip(),
            raw_metric_paths,
            errors,
        )
    if "diagnostics" in artifact_paths:
        _validate_diagnostics_artifact_content(label, artifact_paths["diagnostics"], errors)
    if "robustness_summary" in artifact_paths:
        _validate_robustness_artifact_content(label, artifact_paths["robustness_summary"], errors)
    if (
        isinstance(task, str)
        and _non_empty_text(evidence.get("split"))
        and all(artifact_name in artifact_paths for artifact_name in TOPCONF_GATE_EVIDENCE_ARTIFACTS)
    ):
        _validate_recomputed_public_gate(
            label,
            task.strip(),
            str(evidence["split"]).strip(),
            artifact_paths,
            errors,
        )


def _validate_artifact_descriptor(label: str, artifact_name: str, artifact: Any, errors: list[str]) -> Path | None:
    if not isinstance(artifact, Mapping):
        errors.append(f"{label} gate evidence_artifacts {artifact_name} must include path and sha256")
        return None
    path_value = artifact.get("path")
    if not _non_empty_text(path_value):
        errors.append(f"{label} gate evidence_artifacts {artifact_name}.path must be a non-empty string")
        artifact_path = None
    else:
        artifact_path = Path(path_value)
    sha256 = artifact.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX_RE.fullmatch(sha256):
        errors.append(f"{label} gate evidence_artifacts {artifact_name}.sha256 must be lowercase SHA-256")
        return None
    if artifact_path is None:
        return None
    if not artifact_path.exists():
        errors.append(f"{label} gate evidence_artifacts {artifact_name}.path does not exist")
        return None
    if not artifact_path.is_file():
        errors.append(f"{label} gate evidence_artifacts {artifact_name}.path must point to a file")
        return None
    if _sha256(artifact_path) != sha256:
        errors.append(f"{label} gate evidence_artifacts {artifact_name}.sha256 does not match file content")
        return None
    return artifact_path


def _validate_statistics_summary_content(
    label: str,
    path: Path,
    task: str,
    split: str,
    raw_metric_paths: list[Path],
    errors: list[str],
) -> None:
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} gate statistics_summary must be valid JSON: {exc}")
        return
    if not isinstance(summary, Mapping):
        errors.append(f"{label} gate statistics_summary must be a JSON object")
        return

    main_table = summary.get("main_table")
    task_table = main_table.get(task) if isinstance(main_table, Mapping) else None
    split_table = task_table.get(split) if isinstance(task_table, Mapping) else None
    if not isinstance(split_table, Mapping) or not split_table:
        errors.append(f"{label} gate statistics_summary missing main_table entry for task/split: {task}/{split}")

    referenced_paths = _statistics_raw_metric_paths(summary, base_dir=path.parent)
    for index, raw_metric_path in enumerate(raw_metric_paths):
        if raw_metric_path.resolve() not in referenced_paths:
            errors.append(f"{label} gate statistics_summary does not reference raw_metrics[{index}]")


def _statistics_raw_metric_paths(summary: Mapping[str, Any], *, base_dir: Path) -> set[Path]:
    paths: set[Path] = set()
    metadata = summary.get("metadata")
    if isinstance(metadata, Mapping):
        paths.update(_path_values(metadata.get("raw_metric_paths"), base_dir=base_dir))
    paths.update(_raw_metric_paths_from_rows(summary.get("per_seed_appendix"), base_dir=base_dir))
    reporting_metadata = summary.get("reporting_metadata")
    if isinstance(reporting_metadata, Mapping):
        paths.update(_raw_metric_paths_from_rows(reporting_metadata.get("per_seed_table"), base_dir=base_dir))
    return paths


def _path_values(value: Any, *, base_dir: Path) -> set[Path]:
    if not isinstance(value, list):
        return set()
    return {
        _resolve_artifact_path(str(path), base_dir=base_dir)
        for path in value
        if _non_empty_text(path)
    }


def _raw_metric_paths_from_rows(rows: Any, *, base_dir: Path) -> set[Path]:
    if not isinstance(rows, list):
        return set()
    paths: set[Path] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw_path = row.get("raw_metric_path")
        if _non_empty_text(raw_path):
            paths.add(_resolve_artifact_path(str(raw_path), base_dir=base_dir))
    return paths


def _resolve_artifact_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _validate_diagnostics_artifact_content(label: str, path: Path, errors: list[str]) -> None:
    rows = _read_jsonl_artifact(label, "diagnostics", path, errors)
    if not rows:
        errors.append(f"{label} gate diagnostics artifact must contain JSONL rows")
        return
    public_payloads = [row.get("public_diagnostics") for row in rows if isinstance(row.get("public_diagnostics"), Mapping)]
    if label == "region_text_public":
        _require_any_public_payload_key(
            label,
            public_payloads,
            "cato_router_load_by_phrase_type",
            "CATO router load by phrase type",
            errors,
        )
        _require_any_public_payload_key(
            label,
            public_payloads,
            "no_cato_delta_by_object_size",
            "no-CATO delta by object size",
            errors,
        )
        _require_any_public_payload_key(
            label,
            public_payloads,
            "no_cato_delta_by_phrase_length",
            "no-CATO delta by phrase length",
            errors,
        )
        _require_any_public_payload_key(
            label,
            public_payloads,
            "rceo_reliability_shift_under_blurred_regions",
            "RCEO reliability shift under blurred regions",
            errors,
        )
    elif label == "sentiment_emotion_public":
        _require_any_public_payload_key(
            label,
            public_payloads,
            "lrio_rank_entropy_by_modality_pair",
            "LRIO rank entropy by modality pair",
            errors,
        )
        _require_any_public_payload_key(
            label,
            public_payloads,
            "spo_prototype_load_by_emotion_class",
            "SPO prototype load by emotion class",
            errors,
        )
        _require_any_public_payload_key(
            label,
            public_payloads,
            "rceo_reliability_shift_under_missing_noisy_modality",
            "RCEO reliability shift under missing/noisy modality",
            errors,
        )
        _require_any_public_payload_key(
            label,
            public_payloads,
            "router_load_by_condition",
            "router load by clean/corrupted/missing split",
            errors,
        )


def _read_jsonl_artifact(label: str, artifact_name: str, path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{label} gate {artifact_name} line {line_number} must be valid JSON: {exc}")
                    continue
                if not isinstance(payload, dict):
                    errors.append(f"{label} gate {artifact_name} line {line_number} must be a JSON object")
                    continue
                rows.append(payload)
    except OSError as exc:
        errors.append(f"{label} gate {artifact_name} artifact could not be read: {exc}")
    return rows


def _read_json_artifact(label: str, artifact_name: str, path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} gate {artifact_name} must be valid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label} gate {artifact_name} must be a JSON object")
        return None
    return payload


def _validate_recomputed_public_gate(
    label: str,
    task: str,
    split: str,
    artifact_paths: dict[str, Path],
    errors: list[str],
) -> None:
    summary = _read_json_artifact(label, "statistics_summary", artifact_paths["statistics_summary"], errors)
    diagnostics = _read_jsonl_artifact(label, "diagnostics", artifact_paths["diagnostics"], errors)
    robustness = _read_json_artifact(label, "robustness_summary", artifact_paths["robustness_summary"], errors)
    if summary is None or robustness is None:
        return

    if label == "region_text_public":
        recomputed = evaluate_region_text_gate(
            statistics_summary=summary,
            diagnostics_rows=diagnostics,
            no_cato_score=_summary_model_mean(summary, task, split, "ovha_no_cato"),
            robustness_summary=robustness,
            task=task,
            split=split,
        )
    elif label == "sentiment_emotion_public":
        recomputed = evaluate_sentiment_gate(
            statistics_summary=summary,
            diagnostics_rows=diagnostics,
            ablation_scores={
                "ovha_no_lrio": _summary_model_mean(summary, task, split, "ovha_no_lrio"),
                "ovha_no_spo": _summary_model_mean(summary, task, split, "ovha_no_spo"),
                "ovha_no_rceo": _summary_model_mean(summary, task, split, "ovha_no_rceo"),
            },
            robustness_summary=robustness,
            task=task,
            split=split,
        )
    else:
        return

    if recomputed.get("passed") is True:
        return
    reasons = recomputed.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        errors.append(f"{label} gate artifact recomputation failed")
        return
    errors.extend(f"{label} gate artifact recomputation failed: {reason}" for reason in reasons)


def _summary_model_mean(summary: Mapping[str, Any], task: str, split: str, model: str) -> float | None:
    main_table = summary.get("main_table")
    task_table = main_table.get(task) if isinstance(main_table, Mapping) else None
    split_table = task_table.get(split) if isinstance(task_table, Mapping) else None
    row = split_table.get(model) if isinstance(split_table, Mapping) else None
    if not isinstance(row, Mapping):
        return None
    return _finite_float(row.get("mean"))


def _require_any_public_payload_key(
    label: str,
    payloads: list[Any],
    key: str,
    description: str,
    errors: list[str],
) -> None:
    if any(_non_empty_public_value(payload.get(key)) for payload in payloads if isinstance(payload, Mapping)):
        return
    errors.append(f"{label} gate diagnostics missing {description}")


def _non_empty_public_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, tuple, dict, set)):
        return bool(value)
    return True


def _validate_robustness_artifact_content(label: str, path: Path, errors: list[str]) -> None:
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} gate robustness_summary must be valid JSON: {exc}")
        return
    if not isinstance(summary, Mapping):
        errors.append(f"{label} gate robustness_summary must be a JSON object")
        return
    if summary.get("full_drop_less_than_baseline") is not True:
        errors.append(f"{label} gate robustness_summary does not show lower full-model drop")
    if summary.get("rceo_reliability_monotonic") is not True:
        errors.append(f"{label} gate robustness_summary missing monotonic RCEO reliability")
    coverage = summary.get("required_stress_coverage")
    if not isinstance(coverage, Mapping) or coverage.get("passed") is not True:
        errors.append(f"{label} gate robustness_summary missing required stress coverage pass")
    ablation = summary.get("required_ablation_degradation")
    if not isinstance(ablation, Mapping) or ablation.get("passed") is not True:
        errors.append(f"{label} gate robustness_summary missing required ablation degradation pass")
    if not isinstance(summary.get("rceo_reliability_calibration"), Mapping):
        errors.append(f"{label} gate robustness_summary missing RCEO reliability calibration")


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


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
