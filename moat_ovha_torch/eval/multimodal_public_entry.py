from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from moat_ovha_torch.eval.multimodal_controlled_report import (
    CANDIDATE_ORACLE_GAP_KEYS,
    CONTROLLED_REQUIRED_FAMILIES,
    CONTROLLED_REQUIRED_GATES,
    ORACLE_MATRIX_CELLS,
)


REGION_TEXT_TASK_TYPES = {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities"}
SENTIMENT_EMOTION_TASK_TYPES = {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "meld"}


@dataclass(frozen=True)
class PublicEntryValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_public_entry_requirements(task_type: str, controlled_report: dict[str, Any] | None) -> PublicEntryValidationReport:
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
