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
from moat_ovha_torch.eval.multimodal_controlled_report import (
    CANDIDATE_ORACLE_GAP_KEYS,
    CONTROLLED_REQUIRED_FAMILIES,
    OPERATOR_DIAGNOSTIC_REQUIREMENTS,
    ORACLE_MATRIX_CELLS,
    ROUTER_DECOMPOSITION_ABLATION_KEYS,
    build_controlled_report,
)
from moat_ovha_torch.eval.multimodal_public_entry import (
    REGION_TEXT_TASK_TYPES,
    SENTIMENT_EMOTION_TASK_TYPES,
    validate_public_entry_requirements,
)
from moat_ovha_torch.eval.multimodal_public_gates import (
    evaluate_region_text_gate,
    evaluate_sentiment_gate,
)
from moat_ovha_torch.eval.multimodal_robustness import DEFAULT_REQUIRED_STRESS_TARGETS, summarize_robustness_rows
from moat_ovha_torch.eval.multimodal_statistics import validate_public_summary


CONTROLLED_TOPCONF_EVIDENCE_ARTIFACTS = ("controlled_rows", "diagnostics_report")
CONTROLLED_DIAGNOSTIC_ARTIFACT_TYPES = frozenset({"controlled_diagnostics", "controlled_training_diagnostics"})
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
TOPCONF_GATE_EVIDENCE_ARTIFACTS = ("statistics_summary", "diagnostics", "robustness_summary", "robustness_rows")
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
    _require_controlled_artifact_evidence(controlled_report, errors)

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


def _require_controlled_artifact_evidence(controlled_report: dict[str, Any] | None, errors: list[str]) -> None:
    report = _controlled_report_payload(controlled_report)
    if report is None:
        return
    evidence = report.get("evidence_artifacts")
    if not isinstance(evidence, Mapping):
        errors.append("controlled report evidence_artifacts is required for top-conference main experiments")
        return
    task = evidence.get("task")
    if not isinstance(task, str) or task.strip() != "controlled_multimodal":
        errors.append("controlled report evidence_artifacts task must be controlled_multimodal")
    if not _non_empty_text(evidence.get("generated_by")):
        errors.append("controlled report evidence_artifacts generated_by must identify the evaluator")

    artifact_paths: dict[str, Path] = {}
    for artifact_name in CONTROLLED_TOPCONF_EVIDENCE_ARTIFACTS:
        artifact = evidence.get(artifact_name)
        if artifact is None:
            errors.append(f"controlled report evidence_artifacts missing artifact: {artifact_name}")
            continue
        artifact_path = _validate_controlled_artifact_descriptor(artifact_name, artifact, errors)
        if artifact_path is not None:
            artifact_paths[artifact_name] = artifact_path
    controlled_rows_sha = _artifact_sha256(evidence.get("controlled_rows"))
    diagnostics_sha = _artifact_sha256(evidence.get("diagnostics_report"))
    if controlled_rows_sha is not None and diagnostics_sha is not None and controlled_rows_sha == diagnostics_sha:
        errors.append(
            "controlled report evidence_artifacts diagnostics_report.sha256 must differ from controlled_rows.sha256"
        )

    controlled_rows_path = artifact_paths.get("controlled_rows")
    diagnostics_path = artifact_paths.get("diagnostics_report")
    if controlled_rows_path is None or diagnostics_path is None:
        return
    rows = _read_jsonl_artifact("controlled report", "controlled_rows", controlled_rows_path, errors)
    diagnostics_rows = _read_jsonl_artifact("controlled report", "diagnostics_report", diagnostics_path, errors)
    if not rows:
        errors.append("controlled report controlled_rows artifact must contain JSONL rows")
        return
    if not diagnostics_rows:
        errors.append("controlled report diagnostics_report artifact must contain JSONL rows")
        return
    _validate_controlled_diagnostics_report_content(rows, diagnostics_rows, errors)

    recomputed = build_controlled_report(rows, evidence_artifacts=dict(evidence))
    _validate_controlled_report_matches_artifacts(report, recomputed, errors)
    for task_type in ("phrase_region_grounding", "sentiment_emotion"):
        validation = validate_public_entry_requirements(task_type, recomputed)
        if not validation.ok:
            errors.extend(
                f"controlled report artifact recomputation failed: {error}"
                for error in validation.errors
            )


def _validate_controlled_report_matches_artifacts(
    supplied_report: Mapping[str, Any],
    recomputed_report: Mapping[str, Any],
    errors: list[str],
) -> None:
    supplied_families = supplied_report.get("families")
    recomputed_families = recomputed_report.get("families")
    if not isinstance(supplied_families, Mapping) or not isinstance(recomputed_families, Mapping):
        return

    for family in CONTROLLED_REQUIRED_FAMILIES:
        supplied_row = supplied_families.get(family)
        recomputed_row = recomputed_families.get(family)
        if not isinstance(supplied_row, Mapping) or not isinstance(recomputed_row, Mapping):
            continue
        _validate_controlled_report_field_matches_artifacts(
            family,
            "stackability_passed",
            supplied_row,
            recomputed_row,
            errors,
        )
        for key in _required_controlled_diagnostic_keys(family):
            _validate_controlled_report_field_matches_artifacts(family, key, supplied_row, recomputed_row, errors)
        _validate_controlled_report_oracle_matrix_matches_artifacts(
            family,
            supplied_row,
            recomputed_row,
            errors,
        )


def _validate_controlled_report_field_matches_artifacts(
    family: str,
    key: str,
    supplied_row: Mapping[str, Any],
    recomputed_row: Mapping[str, Any],
    errors: list[str],
) -> None:
    if key not in recomputed_row:
        return
    if key not in supplied_row:
        errors.append(f"controlled report {family}.{key} missing but present in artifact recomputation")
        return
    if key in {"rceo_reliability_monotonic", "rceo_reliability_curve"}:
        if supplied_row.get(key) != recomputed_row.get(key):
            errors.append(f"controlled report {family}.{key} disagrees with artifact recomputation")
        return
    supplied_value = _finite_float(supplied_row.get(key))
    recomputed_value = _finite_float(recomputed_row.get(key))
    if supplied_value is None or recomputed_value is None:
        if supplied_row.get(key) != recomputed_row.get(key):
            errors.append(f"controlled report {family}.{key} disagrees with artifact recomputation")
        return
    if not math.isclose(supplied_value, recomputed_value, rel_tol=1e-9, abs_tol=1e-9):
        errors.append(f"controlled report {family}.{key} disagrees with artifact recomputation")


def _validate_controlled_report_oracle_matrix_matches_artifacts(
    family: str,
    supplied_row: Mapping[str, Any],
    recomputed_row: Mapping[str, Any],
    errors: list[str],
) -> None:
    supplied_matrix = supplied_row.get("oracle_matrix")
    recomputed_matrix = recomputed_row.get("oracle_matrix")
    if not isinstance(supplied_matrix, Mapping) or not isinstance(recomputed_matrix, Mapping):
        return
    for cell in ORACLE_MATRIX_CELLS:
        supplied_loss = _oracle_cell_loss_from_mapping(supplied_matrix, cell)
        recomputed_loss = _oracle_cell_loss_from_mapping(recomputed_matrix, cell)
        if supplied_loss is None or recomputed_loss is None:
            continue
        if not math.isclose(supplied_loss, recomputed_loss, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(
                f"controlled report {family}.oracle_matrix.{cell}.loss disagrees with artifact recomputation"
            )


def _validate_controlled_diagnostics_report_content(
    controlled_rows: list[dict[str, Any]],
    diagnostics_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    controlled_by_family = _controlled_rows_by_family(controlled_rows)
    rows_by_family: dict[str, dict[str, Any]] = {}
    for row in diagnostics_rows:
        family = str(row.get("family", "")).strip()
        if not family:
            errors.append("controlled report diagnostics_report row missing controlled family")
            continue
        if family not in CONTROLLED_REQUIRED_FAMILIES:
            errors.append(f"controlled report diagnostics_report contains unknown controlled family: {family}")
            continue
        if family in rows_by_family:
            errors.append(f"controlled report diagnostics_report duplicate controlled family: {family}")
            continue
        if row.get("artifact_type") not in CONTROLLED_DIAGNOSTIC_ARTIFACT_TYPES:
            errors.append(
                f"controlled report diagnostics_report {family} artifact_type must be controlled_diagnostics "
                "or controlled_training_diagnostics"
            )
        if "active_operator" in row:
            errors.append(
                f"controlled report diagnostics_report {family} must not duplicate controlled_rows active_operator"
            )
        rows_by_family[family] = row

    for family in CONTROLLED_REQUIRED_FAMILIES:
        row = rows_by_family.get(family)
        if row is None:
            errors.append(f"controlled report diagnostics_report missing controlled family: {family}")
            continue
        if row.get("stackability_passed") is not True:
            errors.append(f"controlled report diagnostics_report {family} stackability_passed must be explicit true")
        controlled_row = controlled_by_family.get(family)
        if controlled_row is not None and row.get("stackability_passed") != controlled_row.get("stackability_passed"):
            errors.append(
                f"controlled report diagnostics_report {family} stackability_passed disagrees with controlled_rows"
            )
        for key in _required_controlled_diagnostic_keys(family):
            _validate_controlled_diagnostic_value(family, row, controlled_row, key, errors)
        matrix = row.get("oracle_matrix")
        if not isinstance(matrix, Mapping):
            errors.append(f"controlled report diagnostics_report {family} oracle_matrix must be an object")
            continue
        for cell in ORACLE_MATRIX_CELLS:
            cell_payload = matrix.get(cell)
            if not isinstance(cell_payload, Mapping):
                errors.append(f"controlled report diagnostics_report {family} missing oracle_matrix cell: {cell}")
                continue
            controlled_loss = _oracle_cell_loss(controlled_row, cell) if controlled_row is not None else None
            diagnostics_loss = _finite_float(cell_payload.get("loss"))
            if controlled_loss is not None and diagnostics_loss is not None:
                if not math.isclose(diagnostics_loss, controlled_loss, rel_tol=1e-9, abs_tol=1e-9):
                    errors.append(
                        f"controlled report diagnostics_report {family} oracle_matrix.{cell}.loss "
                        "disagrees with controlled_rows"
                    )


def _controlled_rows_by_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = str(row.get("family", "")).strip()
        if family and family in CONTROLLED_REQUIRED_FAMILIES and family not in by_family:
            by_family[family] = row
    return by_family


def _required_controlled_diagnostic_keys(family: str) -> tuple[str, ...]:
    keys: list[str] = [
        *CANDIDATE_ORACLE_GAP_KEYS,
        *ROUTER_DECOMPOSITION_ABLATION_KEYS,
        "no_operator_memory_delta",
        "no_hyper_adapter_delta",
    ]
    if family == "spo_global_prototype":
        keys.extend(OPERATOR_DIAGNOSTIC_REQUIREMENTS["SPO"])
    if family == "lrio_low_rank_interaction":
        keys.extend(OPERATOR_DIAGNOSTIC_REQUIREMENTS["LRIO"])
        keys.append("no_lrio_delta")
    if family == "cato_alignment_transport":
        keys.extend(OPERATOR_DIAGNOSTIC_REQUIREMENTS["CATO"])
    if family == "rceo_reliability_corruption":
        keys.extend(
            (
                "rceo_prior_effect",
                "rceo_reliability_monotonic",
                "rceo_router_load_shift",
                "rceo_reliability_curve",
                "no_rceo_delta",
            )
        )
    if family == "mixed_relation_operator":
        keys.extend(("router_accuracy", "no_lrio_delta", "no_rceo_delta"))
    return tuple(dict.fromkeys(keys))


def _validate_controlled_diagnostic_value(
    family: str,
    diagnostic_row: Mapping[str, Any],
    controlled_row: Mapping[str, Any] | None,
    key: str,
    errors: list[str],
) -> None:
    if key not in diagnostic_row:
        errors.append(f"controlled report diagnostics_report {family} missing gate diagnostic: {key}")
        return
    value = diagnostic_row.get(key)
    if key == "rceo_reliability_monotonic":
        if value is not True:
            errors.append(f"controlled report diagnostics_report {family} {key} must be explicit true")
        if controlled_row is not None and value != controlled_row.get(key):
            errors.append(f"controlled report diagnostics_report {family} {key} disagrees with controlled_rows")
        return
    if key == "rceo_reliability_curve":
        if not _rceo_reliability_curve_valid(value):
            errors.append(f"controlled report diagnostics_report {family} {key} must be a valid monotonic curve")
        if controlled_row is not None and value != controlled_row.get(key):
            errors.append(f"controlled report diagnostics_report {family} {key} disagrees with controlled_rows")
        return

    numeric = _finite_float(value)
    if numeric is None:
        errors.append(f"controlled report diagnostics_report {family} {key} must be finite")
    elif key == "router_accuracy":
        if numeric < 0.80 or numeric > 1.0:
            errors.append(f"controlled report diagnostics_report {family} {key} must be in [0.80, 1.0]")
    elif numeric <= 0.0 and key not in CANDIDATE_ORACLE_GAP_KEYS:
        errors.append(f"controlled report diagnostics_report {family} {key} must be positive")
    elif numeric < 0.0 and key in CANDIDATE_ORACLE_GAP_KEYS:
        errors.append(f"controlled report diagnostics_report {family} {key} must be non-negative")

    if controlled_row is not None and key in controlled_row:
        controlled_numeric = _finite_float(controlled_row.get(key))
        if controlled_numeric is not None and numeric is not None:
            if not math.isclose(numeric, controlled_numeric, rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"controlled report diagnostics_report {family} {key} disagrees with controlled_rows")


def _rceo_reliability_curve_valid(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    previous_strength: float | None = None
    previous_reliability: float | None = None
    for point in value:
        if not isinstance(point, Mapping):
            return False
        strength = _finite_float(point.get("corruption_strength"))
        reliability = _finite_float(point.get("mean_reliability"))
        if strength is None or reliability is None or reliability < 0.0 or reliability > 1.0:
            return False
        if previous_strength is not None and strength <= previous_strength:
            return False
        if previous_reliability is not None and reliability > previous_reliability + 1e-12:
            return False
        previous_strength = strength
        previous_reliability = reliability
    return True


def _oracle_cell_loss(row: dict[str, Any] | None, cell: str) -> float | None:
    if row is None:
        return None
    matrix = row.get("oracle_matrix")
    if not isinstance(matrix, Mapping):
        return None
    return _oracle_cell_loss_from_mapping(matrix, cell)


def _oracle_cell_loss_from_mapping(matrix: Mapping[str, Any], cell: str) -> float | None:
    cell_payload = matrix.get(cell)
    if not isinstance(cell_payload, Mapping):
        return None
    return _finite_float(cell_payload.get("loss"))


def _controlled_report_payload(controlled_report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(controlled_report, dict):
        return None
    nested = controlled_report.get("controlled_report")
    if nested is None:
        return controlled_report
    return nested if isinstance(nested, dict) else None


def _validate_controlled_artifact_descriptor(
    artifact_name: str,
    artifact: Any,
    errors: list[str],
) -> Path | None:
    if not isinstance(artifact, Mapping):
        errors.append(f"controlled report evidence_artifacts {artifact_name} must include path and sha256")
        return None
    path_value = artifact.get("path")
    if not _non_empty_text(path_value):
        errors.append(f"controlled report evidence_artifacts {artifact_name}.path must be a non-empty string")
        artifact_path = None
    else:
        artifact_path = Path(path_value)
    sha256 = artifact.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX_RE.fullmatch(sha256):
        errors.append(f"controlled report evidence_artifacts {artifact_name}.sha256 must be lowercase SHA-256")
        return None
    if artifact_path is None:
        return None
    if not artifact_path.exists():
        errors.append(f"controlled report evidence_artifacts {artifact_name}.path does not exist")
        return None
    if not artifact_path.is_file():
        errors.append(f"controlled report evidence_artifacts {artifact_name}.path must point to a file")
        return None
    if _sha256(artifact_path) != sha256:
        errors.append(f"controlled report evidence_artifacts {artifact_name}.sha256 does not match file content")
        return None
    return artifact_path


def _artifact_sha256(artifact: Any) -> str | None:
    if not isinstance(artifact, Mapping):
        return None
    sha256 = artifact.get("sha256")
    if isinstance(sha256, str) and _SHA256_HEX_RE.fullmatch(sha256):
        return sha256
    return None


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
    recomputed_report = _require_gate_evidence_artifacts(label, report.get("evidence_artifacts"), errors)
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
    if recomputed_report is not None:
        _validate_public_gate_report_matches_artifacts(label, report, recomputed_report, required_checks, errors)


def _require_gate_evidence_artifacts(label: str, evidence: Any, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(evidence, Mapping):
        errors.append(f"{label} gate evidence_artifacts is required")
        return None

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
        return None
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
    if "robustness_rows" in artifact_paths:
        _validate_robustness_rows_artifact_content(label, artifact_paths["robustness_rows"], errors)
    if (
        isinstance(task, str)
        and _non_empty_text(evidence.get("split"))
        and all(artifact_name in artifact_paths for artifact_name in TOPCONF_GATE_EVIDENCE_ARTIFACTS)
    ):
        return _validate_recomputed_public_gate(
            label,
            task.strip(),
            str(evidence["split"]).strip(),
            artifact_paths,
            errors,
        )
    return None


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
    validation = validate_public_summary(dict(summary))
    if not validation.ok:
        errors.extend(
            f"{label} gate statistics_summary validation failed: {error}"
            for error in validation.errors
        )

    main_table = summary.get("main_table")
    task_table = main_table.get(task) if isinstance(main_table, Mapping) else None
    split_table = task_table.get(split) if isinstance(task_table, Mapping) else None
    if not isinstance(split_table, Mapping) or not split_table:
        errors.append(f"{label} gate statistics_summary missing main_table entry for task/split: {task}/{split}")

    base_dir = path.parent
    referenced_paths = _statistics_raw_metric_paths(summary, base_dir=base_dir)
    for index, raw_metric_path in enumerate(raw_metric_paths):
        if raw_metric_path.resolve() not in referenced_paths:
            errors.append(f"{label} gate statistics_summary does not reference raw_metrics[{index}]")
        _validate_raw_metrics_cover_statistics_appendix(
            label,
            index,
            raw_metric_path,
            summary,
            base_dir=base_dir,
            errors=errors,
        )


def _validate_raw_metrics_cover_statistics_appendix(
    label: str,
    index: int,
    raw_metric_path: Path,
    summary: Mapping[str, Any],
    *,
    base_dir: Path,
    errors: list[str],
) -> None:
    rows = _read_jsonl_artifact(label, f"raw_metrics[{index}]", raw_metric_path, errors)
    if not rows:
        errors.append(f"{label} gate raw_metrics[{index}] artifact must contain JSONL rows")
        return
    appendix_rows = _per_seed_rows_for_raw_metric(summary, raw_metric_path, base_dir=base_dir)
    expected = _metric_row_signature(appendix_rows)
    observed = _metric_row_signature(rows)
    if not expected:
        return
    if not expected.issubset(observed):
        errors.append(f"{label} gate raw_metrics[{index}] rows must cover statistics_summary per_seed_appendix")


def _per_seed_rows_for_raw_metric(
    summary: Mapping[str, Any],
    raw_metric_path: Path,
    *,
    base_dir: Path,
) -> list[Mapping[str, Any]]:
    appendix_rows = summary.get("per_seed_appendix")
    if not isinstance(appendix_rows, list):
        return []
    resolved_raw_metric_path = raw_metric_path.resolve()
    matched_rows: list[Mapping[str, Any]] = []
    for row in appendix_rows:
        if not isinstance(row, Mapping):
            continue
        raw_path = row.get("raw_metric_path")
        if not _non_empty_text(raw_path):
            continue
        if _resolve_artifact_path(str(raw_path), base_dir=base_dir) == resolved_raw_metric_path:
            matched_rows.append(row)
    return matched_rows


def _metric_row_signature(rows: Any) -> set[tuple[str, str, str, int, float]]:
    if not isinstance(rows, list):
        return set()
    signature: set[tuple[str, str, str, int, float]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if not (
            _non_empty_text(row.get("task"))
            and _non_empty_text(row.get("split"))
            and _non_empty_text(row.get("model"))
        ):
            continue
        seed = _integer_value(row.get("seed"))
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
            )
        )
    return signature


def _integer_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


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
) -> dict[str, Any] | None:
    summary = _read_json_artifact(label, "statistics_summary", artifact_paths["statistics_summary"], errors)
    diagnostics = _read_jsonl_artifact(label, "diagnostics", artifact_paths["diagnostics"], errors)
    robustness = _read_json_artifact(label, "robustness_summary", artifact_paths["robustness_summary"], errors)
    robustness_rows = _read_jsonl_artifact(label, "robustness_rows", artifact_paths["robustness_rows"], errors)
    if summary is None or robustness is None:
        return None
    if robustness_rows:
        _validate_robustness_summary_matches_rows(label, robustness, robustness_rows, errors)

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
        return None

    if recomputed.get("passed") is not True:
        reasons = recomputed.get("reasons")
        if not isinstance(reasons, list) or not reasons:
            errors.append(f"{label} gate artifact recomputation failed")
        else:
            errors.extend(f"{label} gate artifact recomputation failed: {reason}" for reason in reasons)
    return recomputed


def _validate_robustness_summary_matches_rows(
    label: str,
    supplied_summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    full_model = str(supplied_summary.get("full_model") or "ovha_full")
    baseline_model = str(supplied_summary.get("baseline_model") or "cross_attention_transformer")
    recomputed = summarize_robustness_rows(rows, full_model=full_model, baseline_model=baseline_model)
    for field in (
        "clean_score",
        "corrupted_score",
        "relative_drop",
        "auc_over_corruption_strength",
        "full_drop_less_than_baseline",
        "rceo_reliability_monotonic",
        "rceo_reliability_shift",
        "rceo_reliability_curve",
        "operator_load_shift",
        "candidate_loss_shift",
        "required_stress_coverage",
        "required_ablation_degradation",
        "rceo_reliability_calibration",
        "robustness_significance",
    ):
        if field not in recomputed:
            continue
        if field not in supplied_summary:
            errors.append(f"{label} gate robustness_summary.{field} missing but present in robustness_rows recomputation")
            continue
        _validate_robustness_value_matches_rows(
            label,
            f"robustness_summary.{field}",
            supplied_summary.get(field),
            recomputed.get(field),
            errors,
        )


def _validate_robustness_value_matches_rows(
    label: str,
    field_path: str,
    supplied_value: Any,
    recomputed_value: Any,
    errors: list[str],
) -> None:
    if isinstance(recomputed_value, Mapping):
        if not isinstance(supplied_value, Mapping):
            errors.append(f"{label} gate {field_path} disagrees with robustness_rows recomputation")
            return
        for key in sorted(set(supplied_value) | set(recomputed_value)):
            next_path = f"{field_path}.{key}"
            if key not in recomputed_value:
                errors.append(f"{label} gate {next_path} not present in robustness_rows recomputation")
            elif key not in supplied_value:
                errors.append(f"{label} gate {next_path} missing but present in robustness_rows recomputation")
            else:
                _validate_robustness_value_matches_rows(
                    label,
                    next_path,
                    supplied_value.get(key),
                    recomputed_value.get(key),
                    errors,
                )
        return
    if isinstance(recomputed_value, (list, tuple)):
        if not isinstance(supplied_value, (list, tuple)) or len(supplied_value) != len(recomputed_value):
            errors.append(f"{label} gate {field_path} disagrees with robustness_rows recomputation")
            return
        for index, (supplied_item, recomputed_item) in enumerate(zip(supplied_value, recomputed_value)):
            _validate_robustness_value_matches_rows(
                label,
                f"{field_path}[{index}]",
                supplied_item,
                recomputed_item,
                errors,
            )
        return
    if isinstance(supplied_value, bool) or isinstance(recomputed_value, bool):
        if supplied_value is not recomputed_value:
            errors.append(f"{label} gate {field_path} disagrees with robustness_rows recomputation")
        return
    supplied_number = _finite_float(supplied_value)
    recomputed_number = _finite_float(recomputed_value)
    if supplied_number is not None and recomputed_number is not None:
        if not math.isclose(supplied_number, recomputed_number, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"{label} gate {field_path} disagrees with robustness_rows recomputation")
        return
    if supplied_value != recomputed_value:
        errors.append(f"{label} gate {field_path} disagrees with robustness_rows recomputation")


def _validate_public_gate_report_matches_artifacts(
    label: str,
    supplied_report: Mapping[str, Any],
    recomputed_report: Mapping[str, Any],
    required_checks: tuple[str, ...],
    errors: list[str],
) -> None:
    if supplied_report.get("passed") != recomputed_report.get("passed"):
        errors.append(f"{label} gate passed disagrees with artifact recomputation")
    if _normalized_gate_reasons(supplied_report.get("reasons")) != _normalized_gate_reasons(recomputed_report.get("reasons")):
        errors.append(f"{label} gate reasons disagree with artifact recomputation")

    supplied_checks = supplied_report.get("checks")
    recomputed_checks = recomputed_report.get("checks")
    if not isinstance(supplied_checks, Mapping) or not isinstance(recomputed_checks, Mapping):
        return
    for check_name in sorted(set(required_checks) | set(recomputed_checks)):
        supplied_check = supplied_checks.get(check_name)
        recomputed_check = recomputed_checks.get(check_name)
        if not isinstance(recomputed_check, Mapping):
            continue
        if not isinstance(supplied_check, Mapping):
            errors.append(f"{label} gate check {check_name} missing but present in artifact recomputation")
            continue
        if supplied_check.get("passed") != recomputed_check.get("passed"):
            errors.append(f"{label} gate check {check_name}.passed disagrees with artifact recomputation")
        if _normalized_gate_reason(supplied_check.get("reason")) != _normalized_gate_reason(recomputed_check.get("reason")):
            errors.append(f"{label} gate check {check_name}.reason disagrees with artifact recomputation")
        if "value" in recomputed_check:
            if "value" not in supplied_check:
                errors.append(f"{label} gate check {check_name}.value missing but present in artifact recomputation")
            else:
                _validate_gate_value_matches_artifacts(
                    label,
                    f"{check_name}.value",
                    supplied_check.get("value"),
                    recomputed_check.get("value"),
                    errors,
                )


def _validate_gate_value_matches_artifacts(
    label: str,
    field_path: str,
    supplied_value: Any,
    recomputed_value: Any,
    errors: list[str],
) -> None:
    if isinstance(recomputed_value, Mapping):
        if not isinstance(supplied_value, Mapping):
            errors.append(f"{label} gate check {field_path} disagrees with artifact recomputation")
            return
        for key in sorted(set(supplied_value) | set(recomputed_value)):
            next_path = f"{field_path}.{key}"
            if key not in recomputed_value:
                errors.append(f"{label} gate check {next_path} not present in artifact recomputation")
            elif key not in supplied_value:
                errors.append(f"{label} gate check {next_path} missing but present in artifact recomputation")
            else:
                _validate_gate_value_matches_artifacts(
                    label,
                    next_path,
                    supplied_value.get(key),
                    recomputed_value.get(key),
                    errors,
                )
        return
    if isinstance(recomputed_value, (list, tuple)):
        if not isinstance(supplied_value, (list, tuple)) or len(supplied_value) != len(recomputed_value):
            errors.append(f"{label} gate check {field_path} disagrees with artifact recomputation")
            return
        for index, (supplied_item, recomputed_item) in enumerate(zip(supplied_value, recomputed_value)):
            _validate_gate_value_matches_artifacts(
                label,
                f"{field_path}[{index}]",
                supplied_item,
                recomputed_item,
                errors,
            )
        return
    if isinstance(supplied_value, bool) or isinstance(recomputed_value, bool):
        if supplied_value is not recomputed_value:
            errors.append(f"{label} gate check {field_path} disagrees with artifact recomputation")
        return
    supplied_number = _finite_float(supplied_value)
    recomputed_number = _finite_float(recomputed_value)
    if supplied_number is not None and recomputed_number is not None:
        if not math.isclose(supplied_number, recomputed_number, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"{label} gate check {field_path} disagrees with artifact recomputation")
        return
    if supplied_value != recomputed_value:
        errors.append(f"{label} gate check {field_path} disagrees with artifact recomputation")


def _normalized_gate_reasons(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(reason).strip() for reason in value if str(reason).strip())
    reason = _normalized_gate_reason(value)
    return (reason,) if reason else ()


def _normalized_gate_reason(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)) and not value:
        return ""
    return str(value).strip()


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
    elif isinstance(coverage, Mapping):
        _validate_robustness_stress_coverage(label, coverage, errors)
    ablation = summary.get("required_ablation_degradation")
    if not isinstance(ablation, Mapping) or ablation.get("passed") is not True:
        errors.append(f"{label} gate robustness_summary missing required ablation degradation pass")
    if not isinstance(summary.get("rceo_reliability_calibration"), Mapping):
        errors.append(f"{label} gate robustness_summary missing RCEO reliability calibration")


def _validate_robustness_rows_artifact_content(label: str, path: Path, errors: list[str]) -> None:
    rows = _read_jsonl_artifact(label, "robustness_rows", path, errors)
    if not rows:
        errors.append(f"{label} gate robustness_rows artifact must contain JSONL rows")


def _validate_robustness_stress_coverage(label: str, coverage: Mapping[str, Any], errors: list[str]) -> None:
    observed = _normalized_text_set(coverage.get("observed"))
    required = _normalized_text_set(coverage.get("required"))
    if observed is None or not observed:
        errors.append(f"{label} gate robustness_summary required_stress_coverage must list observed stress targets")
        return
    if required is None or not required:
        errors.append(f"{label} gate robustness_summary required_stress_coverage must list required stress targets")
        return
    canonical_required = set(DEFAULT_REQUIRED_STRESS_TARGETS)
    missing_canonical = sorted(canonical_required - required)
    if missing_canonical:
        errors.append(
            f"{label} gate robustness_summary required_stress_coverage "
            "required missing canonical Step 6 stress targets: "
            + ", ".join(missing_canonical)
        )
    missing = sorted(required - observed)
    if missing:
        errors.append(
            f"{label} gate robustness_summary required_stress_coverage observed missing required stress targets: "
            + ", ".join(missing)
        )


def _normalized_text_set(value: Any) -> set[str] | None:
    if not isinstance(value, (list, tuple, set)):
        return None
    normalized: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        normalized.add(item.strip())
    return normalized


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
