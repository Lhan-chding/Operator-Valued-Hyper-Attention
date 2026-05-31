from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from moat_ovha_torch.eval.multimodal_controlled_report import (
    CANDIDATE_ORACLE_GAP_KEYS,
    CONTROLLED_REQUIRED_FAMILIES,
    CONTROLLED_REQUIRED_GATES,
    OPERATOR_DIAGNOSTIC_REQUIREMENTS,
    ORACLE_MATRIX_CELLS,
    REGION_TEXT_ENTRY_GATES,
    ROUTER_DECOMPOSITION_ABLATION_KEYS,
    SENTIMENT_ENTRY_GATES,
    build_controlled_report,
)


REGION_TEXT_TASK_TYPES = {
    "phrase_region_grounding",
    "region_text_grounding",
    "refcoco",
    "flickr30k_entities",
    "visual_genome",
}
SENTIMENT_EMOTION_TASK_TYPES = {
    "sentiment_emotion",
    "sentiment_regression",
    "emotion_classification",
    "cmu_mosei",
    "cmu_mosi",
    "meld",
    "iemocap",
}
CONTROLLED_EVIDENCE_ARTIFACTS = ("controlled_rows", "diagnostics_report")
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class PublicEntryValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_public_entry_requirements(
    task_type: str,
    controlled_report: dict[str, Any] | None,
    *,
    require_artifact_files: bool = False,
) -> PublicEntryValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    if controlled_report is None:
        return PublicEntryValidationReport(
            ok=False,
            errors=["controlled go/no-go report is required before public multimodal entry"],
            warnings=warnings,
        )
    controlled_report = _public_entry_report(controlled_report, errors)
    if controlled_report is None:
        return PublicEntryValidationReport(ok=False, errors=errors, warnings=warnings)

    go_no_go = controlled_report.get("go_no_go", {})
    if not isinstance(go_no_go, dict) or go_no_go.get("controlled_multimodal_passed") is not True:
        errors.append("controlled_multimodal_passed must be true before public multimodal entry")
    if not isinstance(go_no_go, dict) or go_no_go.get("enter_public_multimodal") is not True:
        errors.append("enter_public_multimodal must be true before public multimodal entry")
    if (
        isinstance(go_no_go, dict)
        and go_no_go.get("controlled_multimodal_passed") is True
        and go_no_go.get("enter_public_multimodal") is True
    ):
        reasons = go_no_go.get("reasons", [])
        if not isinstance(reasons, (list, tuple)) or reasons:
            errors.append("controlled report go_no_go.reasons must be empty when public entry flags are true")

    _require_complete_controlled_report(controlled_report, errors)
    artifact_paths = _require_controlled_evidence_artifacts(
        controlled_report.get("evidence_artifacts"),
        errors,
        require_artifact_files=require_artifact_files,
    )
    if require_artifact_files:
        _require_controlled_artifact_report_match(task_type, controlled_report, artifact_paths, errors)

    gates = controlled_report.get("gate_table", {})
    if task_type in REGION_TEXT_TASK_TYPES:
        _require_gate(gates, "Stackability", "region-text public entry requires stackability", errors)
        _require_gate(gates, "CATO collapse", "region-text public entry requires CATO collapse", errors)
        _require_gate(
            gates,
            "CATO alignment diagnostics",
            "region-text public entry requires CATO alignment diagnostics",
            errors,
        )
    elif task_type in SENTIMENT_EMOTION_TASK_TYPES:
        _require_gate(gates, "LRIO collapse", "sentiment/emotion public entry requires LRIO collapse", errors)
        _require_gate(gates, "SPO collapse", "sentiment/emotion public entry requires SPO collapse", errors)
        _require_gate(gates, "RCEO gate", "sentiment/emotion public entry requires RCEO gate", errors)
        _require_gate(
            gates,
            "no-LRIO ablation",
            "sentiment/emotion public entry requires no-LRIO ablation degradation",
            errors,
        )
        _require_gate(
            gates,
            "no-RCEO ablation",
            "sentiment/emotion public entry requires no-RCEO ablation degradation",
            errors,
        )
    else:
        errors.append(f"unknown public entry task_type: {task_type}")

    return PublicEntryValidationReport(ok=not errors, errors=errors, warnings=warnings)


def _public_entry_report(payload: dict[str, Any], errors: list[str]) -> dict[str, Any] | None:
    mode = str(payload.get("mode", "")).strip()
    if mode == "oracle_smoke_only":
        errors.append("oracle_smoke_only is not valid public-entry evidence")
        return None
    nested = payload.get("controlled_report")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        errors.append("controlled_report wrapper must contain an object")
        return None
    nested_mode = str(nested.get("mode", "")).strip()
    if nested_mode == "oracle_smoke_only":
        errors.append("oracle_smoke_only is not valid public-entry evidence")
        return None
    return nested


def _require_gate(gates: Any, gate_name: str, message: str, errors: list[str]) -> None:
    gate = gates.get(gate_name, {}) if isinstance(gates, dict) else {}
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        errors.append(message)


def _require_complete_controlled_report(controlled_report: dict[str, Any], errors: list[str]) -> None:
    observed_cells = controlled_report.get("oracle_matrix_cells")
    if not isinstance(observed_cells, (list, tuple)) or tuple(observed_cells) != ORACLE_MATRIX_CELLS:
        errors.append("controlled report must include full oracle_matrix_cells")

    families = controlled_report.get("families")
    if not isinstance(families, dict):
        families = {}
        errors.append("controlled report must include controlled families")
    for family in CONTROLLED_REQUIRED_FAMILIES:
        family_row = families.get(family)
        if not isinstance(family_row, dict):
            errors.append(f"controlled report missing controlled family: {family}")
        else:
            _require_family_oracle_evidence(family, family_row, errors)
    for family in sorted(set(families) - set(CONTROLLED_REQUIRED_FAMILIES)):
        errors.append(f"controlled report contains unknown controlled family: {family}")

    gates = controlled_report.get("gate_table")
    if not isinstance(gates, dict):
        gates = {}
        errors.append("controlled report must include gate_table")
    for gate_name in CONTROLLED_REQUIRED_GATES:
        gate = gates.get(gate_name)
        if not isinstance(gate, dict):
            errors.append(f"controlled report missing required gate: {gate_name}")
        elif gate.get("passed") is not True:
            errors.append(f"controlled report required gate did not pass: {gate_name}")


def _require_controlled_evidence_artifacts(
    evidence: Any,
    errors: list[str],
    *,
    require_artifact_files: bool,
) -> dict[str, Path]:
    artifact_paths: dict[str, Path] = {}
    if not isinstance(evidence, Mapping):
        errors.append("controlled report evidence_artifacts is required")
        return artifact_paths
    task = evidence.get("task")
    if not isinstance(task, str) or task.strip() != "controlled_multimodal":
        errors.append("controlled report evidence_artifacts task must be controlled_multimodal")
    if not _non_empty_text(evidence.get("generated_by")):
        errors.append("controlled report evidence_artifacts generated_by must identify the evaluator")
    for artifact_name in CONTROLLED_EVIDENCE_ARTIFACTS:
        artifact = evidence.get(artifact_name)
        if artifact is None:
            errors.append(f"controlled report evidence_artifacts missing artifact: {artifact_name}")
            continue
        artifact_path = _validate_artifact_descriptor(
            artifact_name,
            artifact,
            errors,
            require_artifact_files=require_artifact_files,
        )
        if artifact_path is not None:
            artifact_paths[artifact_name] = artifact_path
    controlled_rows_sha = _artifact_sha256(evidence.get("controlled_rows"))
    diagnostics_sha = _artifact_sha256(evidence.get("diagnostics_report"))
    if controlled_rows_sha is not None and diagnostics_sha is not None and controlled_rows_sha == diagnostics_sha:
        errors.append(
            "controlled report evidence_artifacts diagnostics_report.sha256 must differ from controlled_rows.sha256"
        )
    return artifact_paths


def _validate_artifact_descriptor(
    artifact_name: str,
    artifact: Any,
    errors: list[str],
    *,
    require_artifact_files: bool,
) -> Path | None:
    if not isinstance(artifact, Mapping):
        errors.append(f"controlled report evidence_artifacts {artifact_name} must include path and sha256")
        return None
    path_value = artifact.get("path")
    if not _non_empty_text(path_value):
        errors.append(f"controlled report evidence_artifacts {artifact_name}.path must be a non-empty string")
    sha256 = artifact.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX_RE.fullmatch(sha256):
        errors.append(f"controlled report evidence_artifacts {artifact_name}.sha256 must be lowercase SHA-256")
        return None
    if not require_artifact_files or not _non_empty_text(path_value):
        return None
    path = Path(path_value)
    if not path.exists():
        errors.append(f"controlled report evidence_artifacts {artifact_name}.path does not exist")
        return None
    if not path.is_file():
        errors.append(f"controlled report evidence_artifacts {artifact_name}.path must point to a file")
        return None
    if _sha256(path) != sha256:
        errors.append(f"controlled report evidence_artifacts {artifact_name}.sha256 does not match file content")
        return None
    return path


def _require_controlled_artifact_report_match(
    task_type: str,
    controlled_report: dict[str, Any],
    artifact_paths: Mapping[str, Path],
    errors: list[str],
) -> None:
    controlled_rows_path = artifact_paths.get("controlled_rows")
    if controlled_rows_path is None:
        return
    controlled_rows = _read_jsonl_dicts("controlled_rows", controlled_rows_path, errors)
    if not controlled_rows:
        errors.append("controlled report controlled_rows artifact must contain at least one JSON object")
        return
    diagnostics_path = artifact_paths.get("diagnostics_report")
    if diagnostics_path is not None:
        diagnostics_rows = _read_jsonl_dicts("diagnostics_report", diagnostics_path, errors)
        if not diagnostics_rows:
            errors.append("controlled report diagnostics_report artifact must contain at least one JSON object")
        else:
            _validate_controlled_diagnostics_report_content(controlled_rows, diagnostics_rows, errors)
    recomputed_report = build_controlled_report(
        controlled_rows,
        evidence_artifacts=_evidence_artifacts_copy(controlled_report.get("evidence_artifacts")),
    )
    _compare_recomputed_controlled_report(controlled_report, recomputed_report, errors)
    recomputed_entry = validate_public_entry_requirements(
        task_type,
        recomputed_report,
        require_artifact_files=False,
    )
    for recomputed_error in recomputed_entry.errors:
        errors.append(f"controlled report artifact recomputation failed: {recomputed_error}")


def _read_jsonl_dicts(label: str, path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        errors.append(f"controlled report {label} artifact could not be read: {exc}")
        return rows
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"controlled report {label} artifact line {line_number} is not valid JSON: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"controlled report {label} artifact line {line_number} must be a JSON object")
            continue
        rows.append(payload)
    return rows


def _validate_controlled_diagnostics_report_content(
    controlled_rows: list[dict[str, Any]],
    diagnostics_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    controlled_by_family = _controlled_rows_by_family(controlled_rows)
    diagnostics_by_family: dict[str, dict[str, Any]] = {}
    for row in diagnostics_rows:
        family = str(row.get("family", "")).strip()
        if not family:
            errors.append("controlled report diagnostics_report row missing controlled family")
            continue
        if family not in CONTROLLED_REQUIRED_FAMILIES:
            errors.append(f"controlled report diagnostics_report contains unknown controlled family: {family}")
            continue
        if family in diagnostics_by_family:
            errors.append(f"controlled report diagnostics_report duplicate controlled family: {family}")
            continue
        if row.get("artifact_type") != "controlled_diagnostics":
            errors.append(
                f"controlled report diagnostics_report {family} artifact_type must be controlled_diagnostics"
            )
        if "active_operator" in row:
            errors.append(
                f"controlled report diagnostics_report {family} must not duplicate controlled_rows active_operator"
            )
        diagnostics_by_family[family] = row

    for family in CONTROLLED_REQUIRED_FAMILIES:
        diagnostics_row = diagnostics_by_family.get(family)
        if diagnostics_row is None:
            errors.append(f"controlled report diagnostics_report missing controlled family: {family}")
            continue
        controlled_row = controlled_by_family.get(family)
        if diagnostics_row.get("stackability_passed") is not True:
            errors.append(f"controlled report diagnostics_report {family} stackability_passed must be explicit true")
        if controlled_row is not None and diagnostics_row.get("stackability_passed") != controlled_row.get("stackability_passed"):
            errors.append(
                f"controlled report diagnostics_report {family} stackability_passed disagrees with controlled_rows"
            )
        for key in _required_controlled_diagnostic_keys(family):
            _validate_controlled_diagnostic_value(family, diagnostics_row, controlled_row, key, errors)
        _validate_controlled_diagnostic_oracle_matrix(family, diagnostics_row, controlled_row, errors)


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
    diagnostics_row: Mapping[str, Any],
    controlled_row: Mapping[str, Any] | None,
    key: str,
    errors: list[str],
) -> None:
    if key not in diagnostics_row:
        errors.append(f"controlled report diagnostics_report {family} missing gate diagnostic: {key}")
        return
    value = diagnostics_row.get(key)
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


def _validate_controlled_diagnostic_oracle_matrix(
    family: str,
    diagnostics_row: Mapping[str, Any],
    controlled_row: Mapping[str, Any] | None,
    errors: list[str],
) -> None:
    matrix = diagnostics_row.get("oracle_matrix")
    if not isinstance(matrix, Mapping):
        errors.append(f"controlled report diagnostics_report {family} oracle_matrix must be an object")
        return
    for cell in ORACLE_MATRIX_CELLS:
        cell_payload = matrix.get(cell)
        if not isinstance(cell_payload, Mapping):
            errors.append(f"controlled report diagnostics_report {family} missing oracle_matrix cell: {cell}")
            continue
        diagnostics_loss = _finite_float(cell_payload.get("loss"))
        if diagnostics_loss is None:
            errors.append(f"controlled report diagnostics_report {family} oracle_matrix.{cell}.loss must be finite")
            continue
        controlled_loss = _oracle_cell_loss(controlled_row, cell)
        if controlled_loss is not None and not math.isclose(diagnostics_loss, controlled_loss, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(
                f"controlled report diagnostics_report {family} oracle_matrix.{cell}.loss disagrees with controlled_rows"
            )


def _oracle_cell_loss(row: Mapping[str, Any] | None, cell: str) -> float | None:
    if row is None:
        return None
    matrix = row.get("oracle_matrix")
    if not isinstance(matrix, Mapping):
        return None
    cell_payload = matrix.get(cell)
    if not isinstance(cell_payload, Mapping):
        return None
    return _finite_float(cell_payload.get("loss"))


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


def _evidence_artifacts_copy(evidence_artifacts: Any) -> dict[str, Any] | None:
    if not isinstance(evidence_artifacts, Mapping):
        return None
    return dict(evidence_artifacts)


def _compare_recomputed_controlled_report(
    supplied_report: dict[str, Any],
    recomputed_report: dict[str, Any],
    errors: list[str],
) -> None:
    _compare_go_no_go(supplied_report, recomputed_report, errors)
    _compare_gate_table(supplied_report, recomputed_report, errors)
    supplied_families = supplied_report.get("families")
    recomputed_families = recomputed_report.get("families")
    if not isinstance(supplied_families, Mapping) or not isinstance(recomputed_families, Mapping):
        return
    for family in CONTROLLED_REQUIRED_FAMILIES:
        supplied_family = supplied_families.get(family)
        recomputed_family = recomputed_families.get(family)
        if not isinstance(supplied_family, Mapping) or not isinstance(recomputed_family, Mapping):
            continue
        for gap_key in CANDIDATE_ORACLE_GAP_KEYS:
            _compare_float_field(
                f"controlled report {family}.{gap_key}",
                supplied_family.get(gap_key),
                recomputed_family.get(gap_key),
                errors,
            )
        supplied_matrix = supplied_family.get("oracle_matrix")
        recomputed_matrix = recomputed_family.get("oracle_matrix")
        if not isinstance(supplied_matrix, Mapping) or not isinstance(recomputed_matrix, Mapping):
            continue
        for cell in ORACLE_MATRIX_CELLS:
            supplied_cell = supplied_matrix.get(cell)
            recomputed_cell = recomputed_matrix.get(cell)
            if not isinstance(supplied_cell, Mapping) or not isinstance(recomputed_cell, Mapping):
                continue
            _compare_float_field(
                f"controlled report {family}.oracle_matrix.{cell}.loss",
                supplied_cell.get("loss"),
                recomputed_cell.get("loss"),
                errors,
            )


def _compare_go_no_go(
    supplied_report: Mapping[str, Any],
    recomputed_report: Mapping[str, Any],
    errors: list[str],
) -> None:
    supplied = supplied_report.get("go_no_go")
    recomputed = recomputed_report.get("go_no_go")
    if not isinstance(supplied, Mapping) or not isinstance(recomputed, Mapping):
        return
    for key in ("controlled_multimodal_passed", "enter_public_multimodal"):
        if supplied.get(key) != recomputed.get(key):
            errors.append(f"controlled report go_no_go.{key} disagrees with artifact recomputation")
    supplied_reasons = supplied.get("reasons", [])
    recomputed_reasons = recomputed.get("reasons", [])
    if _normalized_sequence(supplied_reasons) != _normalized_sequence(recomputed_reasons):
        errors.append("controlled report go_no_go.reasons disagrees with artifact recomputation")


def _compare_gate_table(
    supplied_report: Mapping[str, Any],
    recomputed_report: Mapping[str, Any],
    errors: list[str],
) -> None:
    supplied_gates = supplied_report.get("gate_table")
    recomputed_gates = recomputed_report.get("gate_table")
    if not isinstance(supplied_gates, Mapping) or not isinstance(recomputed_gates, Mapping):
        return
    for gate_name in CONTROLLED_REQUIRED_GATES + REGION_TEXT_ENTRY_GATES + SENTIMENT_ENTRY_GATES:
        supplied_gate = supplied_gates.get(gate_name)
        recomputed_gate = recomputed_gates.get(gate_name)
        if not isinstance(supplied_gate, Mapping) or not isinstance(recomputed_gate, Mapping):
            continue
        if supplied_gate.get("passed") != recomputed_gate.get("passed"):
            errors.append(f"controlled report gate_table.{gate_name}.passed disagrees with artifact recomputation")
        _compare_optional_float_field(
            f"controlled report gate_table.{gate_name}.value",
            supplied_gate.get("value"),
            recomputed_gate.get("value"),
            errors,
        )


def _compare_optional_float_field(
    label: str,
    supplied_value: Any,
    recomputed_value: Any,
    errors: list[str],
) -> None:
    if supplied_value is None and recomputed_value is None:
        return
    if isinstance(supplied_value, Mapping) or isinstance(recomputed_value, Mapping):
        return
    if isinstance(supplied_value, (list, tuple)) or isinstance(recomputed_value, (list, tuple)):
        return
    supplied_float = _finite_float(supplied_value)
    recomputed_float = _finite_float(recomputed_value)
    if supplied_float is None and recomputed_float is None:
        return
    _compare_float_field(label, supplied_value, recomputed_value, errors)


def _compare_float_field(
    label: str,
    supplied_value: Any,
    recomputed_value: Any,
    errors: list[str],
) -> None:
    supplied_float = _finite_float(supplied_value)
    recomputed_float = _finite_float(recomputed_value)
    if supplied_float is None or recomputed_float is None:
        errors.append(f"{label} disagrees with artifact recomputation")
        return
    if not math.isclose(supplied_float, recomputed_float, rel_tol=1e-12, abs_tol=1e-12):
        errors.append(f"{label} disagrees with artifact recomputation")


def _normalized_sequence(value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        return [value]
    return list(value)


def _require_family_oracle_evidence(family: str, row: dict[str, Any], errors: list[str]) -> None:
    matrix = row.get("oracle_matrix")
    if not isinstance(matrix, dict):
        errors.append(f"controlled report family {family} missing oracle_matrix")
    else:
        for cell in ORACLE_MATRIX_CELLS:
            cell_payload = matrix.get(cell)
            if not isinstance(cell_payload, dict):
                errors.append(f"controlled report family {family} missing oracle_matrix cell: {cell}")
                continue
            loss = _finite_float(cell_payload.get("loss"))
            if loss is None or loss < 0.0:
                errors.append(f"controlled report family {family} oracle_matrix.{cell}.loss must be finite non-negative")
    for key in CANDIDATE_ORACLE_GAP_KEYS:
        value = _finite_float(row.get(key))
        if value is None or value < 0.0:
            errors.append(f"controlled report family {family} missing oracle gap evidence: {key}")
    if family == "rceo_reliability_corruption":
        prior_effect = _finite_float(row.get("rceo_prior_effect"))
        if prior_effect is None or prior_effect <= 0.0:
            errors.append("controlled report family rceo_reliability_corruption missing positive rceo_prior_effect")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _artifact_sha256(artifact: Any) -> str | None:
    if not isinstance(artifact, Mapping):
        return None
    sha256 = artifact.get("sha256")
    if isinstance(sha256, str) and _SHA256_HEX_RE.fullmatch(sha256):
        return sha256
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
