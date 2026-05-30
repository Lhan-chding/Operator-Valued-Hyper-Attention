from __future__ import annotations

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
OPERATOR_TO_FAMILY = {
    "TLEO": "tleo_local_evidence",
    "SPO": "spo_global_prototype",
    "LRIO": "lrio_low_rank_interaction",
    "CATO": "cato_alignment_transport",
}


def build_controlled_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_rows = {str(row["family"]): row for row in rows}
    gate_table = {
        "Stackability": _stackability_gate(rows),
        "TLEO collapse": _collapse_gate(family_rows, "TLEO"),
        "SPO collapse": _collapse_gate(family_rows, "SPO"),
        "LRIO collapse": _collapse_gate(family_rows, "LRIO"),
        "CATO collapse": _collapse_gate(family_rows, "CATO"),
        "Router gate": _router_gate(family_rows),
        "RCEO gate": _rceo_gate(family_rows),
        "Memory gate": _delta_gate(rows, "no_operator_memory_delta", "Memory gate"),
        "Adapter gate": _delta_gate(rows, "no_hyper_adapter_delta", "Adapter gate"),
    }
    reasons: list[str] = []
    for family in CONTROLLED_REQUIRED_FAMILIES:
        if family not in family_rows:
            reasons.append(f"missing controlled family: {family}")
    for row in rows:
        missing_cells = _missing_oracle_cells(row)
        if missing_cells:
            reasons.append(f"{row.get('family', '<unknown>')} missing oracle matrix cells: {', '.join(missing_cells)}")
    for name, gate in gate_table.items():
        if not gate["passed"]:
            reasons.append(f"gate failed: {name}")
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


def _stackability_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = bool(rows) and all(bool(row.get("stackability_passed")) for row in rows)
    return {"passed": passed, "condition": "all candidate values are [B,Q,Dy] and stackability_passed is true"}


def _collapse_gate(family_rows: dict[str, dict[str, Any]], operator: str) -> dict[str, Any]:
    family = OPERATOR_TO_FAMILY[operator]
    row = family_rows.get(family)
    if row is None:
        return {"passed": False, "condition": f"{operator} family exists"}
    full_loss = _loss(row, "learned_learned")
    specialist_loss = float(row.get("specialist_loss", _loss(row, "true_learned")))
    threshold = specialist_loss * 1.05 + 1e-6
    return {
        "passed": full_loss <= threshold,
        "value": full_loss,
        "threshold": threshold,
        "condition": f"{operator} learned loss <= specialist * 1.05 + eps",
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
    return {
        "passed": monotonic and load_shift > 0.0,
        "value": {"monotonic": monotonic, "router_load_shift": load_shift},
        "condition": "reliability decreases with corruption and router load shifts coherently",
    }


def _delta_gate(rows: list[dict[str, Any]], key: str, name: str) -> dict[str, Any]:
    values = [float(row.get(key, 0.0)) for row in rows]
    min_value = min(values) if values else 0.0
    return {
        "passed": bool(values) and min_value > 0.0,
        "value": min_value,
        "condition": f"{name} ablation is worse than full model for every controlled row",
    }


def _loss(row: dict[str, Any], cell: str) -> float:
    return float(row.get("oracle_matrix", {}).get(cell, {}).get("loss", float("inf")))
