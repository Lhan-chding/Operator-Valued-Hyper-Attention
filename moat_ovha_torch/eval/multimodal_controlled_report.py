from __future__ import annotations

import math
from typing import Any


ORACLE_MATRIX_CELLS = ("learned_learned", "true_learned", "learned_true", "true_true")
CONTROLLED_REQUIRED_FAMILIES = (
    "tleo_local_evidence",
    "spo_global_prototype",
    "lrio_low_rank_interaction",
    "cato_alignment_transport",
    "rceo_reliability_corruption",
    "mixed_relation_operator",
)
CONTROLLED_REQUIRED_GATES = (
    "Stackability",
    "TLEO collapse",
    "SPO collapse",
    "LRIO collapse",
    "CATO collapse",
    "Router gate",
    "RCEO gate",
    "Memory gate",
    "Adapter gate",
)
REGION_TEXT_ENTRY_GATES = (
    "CATO alignment diagnostics",
)
SENTIMENT_ENTRY_GATES = (
    "no-LRIO ablation",
    "no-RCEO ablation",
)
CANDIDATE_ORACLE_GAP_KEYS = (
    "TLEO_oracle_gap",
    "SPO_oracle_gap",
    "LRIO_oracle_gap",
    "CATO_oracle_gap",
)
OPERATOR_TO_FAMILY = {
    "TLEO": "tleo_local_evidence",
    "SPO": "spo_global_prototype",
    "LRIO": "lrio_low_rank_interaction",
    "CATO": "cato_alignment_transport",
}
OPERATOR_DIAGNOSTIC_REQUIREMENTS = {
    "SPO": ("prototype_kl_delta",),
    "LRIO": ("rank_logits_kl_delta",),
    "CATO": ("alignment_entropy_delta", "alignment_topk_delta"),
}


def build_controlled_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_rows = {str(row["family"]): row for row in rows}
    gate_table = {
        "Stackability": _stackability_gate(rows),
        "TLEO collapse": _collapse_gate(family_rows, "TLEO"),
        "SPO collapse": _collapse_gate(family_rows, "SPO"),
        "LRIO collapse": _collapse_gate(family_rows, "LRIO"),
        "CATO collapse": _collapse_gate(family_rows, "CATO"),
        REGION_TEXT_ENTRY_GATES[0]: _operator_diagnostic_gate(
            family_rows,
            "CATO",
            REGION_TEXT_ENTRY_GATES[0],
        ),
        "Router gate": _router_gate(family_rows),
        "RCEO gate": _rceo_gate(family_rows),
        "Memory gate": _delta_gate(rows, "no_operator_memory_delta", "Memory gate"),
        "Adapter gate": _delta_gate(rows, "no_hyper_adapter_delta", "Adapter gate"),
        SENTIMENT_ENTRY_GATES[0]: _targeted_delta_gate(
            family_rows,
            "no_lrio_delta",
            SENTIMENT_ENTRY_GATES[0],
            ("lrio_low_rank_interaction", "mixed_relation_operator"),
        ),
        SENTIMENT_ENTRY_GATES[1]: _targeted_delta_gate(
            family_rows,
            "no_rceo_delta",
            SENTIMENT_ENTRY_GATES[1],
            ("rceo_reliability_corruption", "mixed_relation_operator"),
        ),
    }
    reasons: list[str] = []
    for family in CONTROLLED_REQUIRED_FAMILIES:
        if family not in family_rows:
            reasons.append(f"missing controlled family: {family}")
    for row in rows:
        missing_cells = _missing_oracle_cells(row)
        if missing_cells:
            reasons.append(f"{row.get('family', '<unknown>')} missing oracle matrix cells: {', '.join(missing_cells)}")
        for matrix_reason in _oracle_matrix_value_reasons(row):
            reasons.append(matrix_reason)
        for oracle_reason in _oracle_gap_reasons(row):
            reasons.append(oracle_reason)
        for rceo_reason in _rceo_prior_effect_reasons(row):
            reasons.append(rceo_reason)
    for name in CONTROLLED_REQUIRED_GATES:
        gate = gate_table[name]
        if not gate["passed"]:
            reasons.append(f"gate failed: {name}")
        for diagnostic_reason in gate.get("diagnostic_reasons", ()):
            reasons.append(f"{name} {diagnostic_reason}")
    return {
        "oracle_matrix_cells": ORACLE_MATRIX_CELLS,
        "families": family_rows,
        "gate_table": gate_table,
        "go_no_go": {
            "controlled_multimodal_passed": not reasons,
            "enter_public_multimodal": not reasons,
            "reasons": reasons,
        },
    }


def _missing_oracle_cells(row: dict[str, Any]) -> list[str]:
    matrix = row.get("oracle_matrix", {})
    return [cell for cell in ORACLE_MATRIX_CELLS if cell not in matrix]


def _oracle_gap_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in CANDIDATE_ORACLE_GAP_KEYS:
        if key not in row:
            reasons.append(f"{row.get('family', '<unknown>')} missing oracle gap evidence: {key}")
            continue
        value = _finite_float(row[key])
        if value is None or value < 0.0:
            reasons.append(f"{row.get('family', '<unknown>')} oracle gap must be finite non-negative: {key}")
    return reasons


def _rceo_prior_effect_reasons(row: dict[str, Any]) -> list[str]:
    if row.get("family") != "rceo_reliability_corruption":
        return []
    if "rceo_prior_effect" not in row:
        return ["rceo_reliability_corruption missing RCEO prior effect"]
    value = _finite_float(row["rceo_prior_effect"])
    if value is None or value <= 0.0:
        return ["rceo_reliability_corruption RCEO prior effect must be positive"]
    return []


def _stackability_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = bool(rows) and all(bool(row.get("stackability_passed")) for row in rows)
    return {"passed": passed, "condition": "all candidate values are [B,Q,Dy] and stackability_passed is true"}


def _collapse_gate(family_rows: dict[str, dict[str, Any]], operator: str) -> dict[str, Any]:
    family = OPERATOR_TO_FAMILY[operator]
    row = family_rows.get(family)
    if row is None:
        return {"passed": False, "condition": f"{operator} family exists"}
    true_router_learned_adapter_loss = _loss(row, "true_learned")
    specialist_loss = float(row.get("specialist_loss", _loss(row, "true_learned")))
    threshold = specialist_loss * 1.05 + 1e-6
    diagnostic_reasons = _operator_diagnostic_reasons(row, operator)
    loss_passed = true_router_learned_adapter_loss <= threshold
    return {
        "passed": loss_passed and not diagnostic_reasons,
        "value": true_router_learned_adapter_loss,
        "threshold": threshold,
        "oracle_cell": "true_learned",
        "loss_passed": loss_passed,
        "diagnostic_reasons": diagnostic_reasons,
        "condition": f"{operator} true-router + learned-adapter loss <= specialist * 1.05 + eps",
    }


def _router_gate(family_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = family_rows.get("mixed_relation_operator")
    accuracy = float(row.get("router_accuracy", 0.0)) if row is not None else 0.0
    return {
        "passed": accuracy >= 0.80,
        "value": accuracy,
        "threshold": 0.80,
        "condition": "mixed relation active_operator accuracy >= 80% test",
    }


def _rceo_gate(family_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = family_rows.get("rceo_reliability_corruption")
    monotonic = bool(row.get("rceo_reliability_monotonic")) if row is not None else False
    load_shift = float(row.get("rceo_router_load_shift", 0.0)) if row is not None else 0.0
    prior_effect = float(row.get("rceo_prior_effect", 0.0)) if row is not None else 0.0
    return {
        "passed": monotonic and load_shift > 0.0 and prior_effect > 0.0,
        "value": {"monotonic": monotonic, "router_load_shift": load_shift, "prior_effect": prior_effect},
        "condition": "reliability decreases with corruption, router load shifts coherently, and RCEO prior improves loss",
    }


def _delta_gate(rows: list[dict[str, Any]], key: str, name: str) -> dict[str, Any]:
    values = [float(row.get(key, 0.0)) for row in rows]
    min_value = min(values) if values else 0.0
    return {
        "passed": bool(values) and min_value > 0.0,
        "value": min_value,
        "condition": f"{name} ablation is worse than full model for every controlled row",
    }


def _targeted_delta_gate(
    family_rows: dict[str, dict[str, Any]],
    key: str,
    name: str,
    required_families: tuple[str, ...],
) -> dict[str, Any]:
    values: list[float] = []
    reasons: list[str] = []
    for family in required_families:
        row = family_rows.get(family)
        if row is None or key not in row:
            reasons.append(f"missing {name} evidence for {family}")
            continue
        value = float(row[key])
        values.append(value)
        if value <= 0.0:
            reasons.append(f"{name} does not degrade for {family}")
    return {
        "passed": not reasons,
        "value": min(values) if values else 0.0,
        "required_families": required_families,
        "reasons": reasons,
        "condition": f"{name} is worse than full model on targeted sentiment-entry controlled families",
    }


def _operator_diagnostic_gate(
    family_rows: dict[str, dict[str, Any]],
    operator: str,
    name: str,
) -> dict[str, Any]:
    family = OPERATOR_TO_FAMILY[operator]
    row = family_rows.get(family)
    reasons = [f"missing {operator} controlled family"] if row is None else _operator_diagnostic_reasons(row, operator)
    return {
        "passed": not reasons,
        "operator": operator,
        "required_diagnostics": OPERATOR_DIAGNOSTIC_REQUIREMENTS.get(operator, ()),
        "reasons": reasons,
        "condition": f"{name} are present and improve on the {operator} controlled family",
    }


def _loss(row: dict[str, Any], cell: str) -> float:
    return float(row.get("oracle_matrix", {}).get(cell, {}).get("loss", float("inf")))


def _operator_diagnostic_reasons(row: dict[str, Any], operator: str) -> list[str]:
    reasons: list[str] = []
    for key in OPERATOR_DIAGNOSTIC_REQUIREMENTS.get(operator, ()):
        if key not in row:
            reasons.append(f"missing diagnostic: {key}")
            continue
        if float(row[key]) <= 0.0:
            reasons.append(f"diagnostic must improve: {key}")
    return reasons


def _oracle_matrix_value_reasons(row: dict[str, Any]) -> list[str]:
    family = str(row.get("family", "<unknown>"))
    matrix = row.get("oracle_matrix", {})
    if not isinstance(matrix, dict):
        return [f"{family} oracle_matrix must be an object keyed by oracle cell"]
    reasons: list[str] = []
    for cell in ORACLE_MATRIX_CELLS:
        if cell not in matrix:
            continue
        cell_values = matrix.get(cell)
        if not isinstance(cell_values, dict):
            reasons.append(f"{family} oracle_matrix.{cell} must be an object with loss")
            continue
        loss = _finite_float(cell_values.get("loss"))
        if loss is None or loss < 0.0:
            reasons.append(f"{family} oracle_matrix.{cell}.loss must be finite non-negative")
    return reasons


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
