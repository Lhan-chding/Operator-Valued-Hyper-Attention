from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
