from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_DIAGNOSTIC_KEYS = (
    "router_entropy",
    "router_load_by_candidate",
    "router_logit_parts",
    "candidate_loss",
    "adapter_params",
    "memory_slot_norm",
    "stackability_passed",
)
MULTIMODAL_CANDIDATE_NAMES = ("TLEO", "SPO", "LRIO", "CATO")
REQUIRED_ROUTER_LOGIT_PARTS = ("memory", "evidence", "reliability")
REQUIRED_ADAPTER_PARAM_KEYS = (
    "TLEO_lengthscale",
    "SPO_temperature",
    "LRIO_rank_entropy",
    "CATO_alignment_temperature",
)

CANDIDATE_DIAGNOSTIC_KEYS = {
    "TLEO": ("lengthscale", "local_entropy", "candidate_loss"),
    "SPO": ("prototype_entropy", "prototype_temperature", "candidate_loss"),
    "LRIO": ("rank_entropy", "interaction_temperature", "candidate_loss"),
    "CATO": ("alignment_entropy", "alignment_temperature", "transport_scale", "candidate_loss"),
    "RCEO": ("modality_reliability", "reliability_bias_norm", "corruption_response"),
}


@dataclass(frozen=True)
class DiagnosticValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


def required_diagnostic_keys() -> tuple[str, ...]:
    return REQUIRED_DIAGNOSTIC_KEYS


def validate_diagnostic_row(row: dict[str, Any]) -> DiagnosticValidationReport:
    errors = [f"missing diagnostic key: {key}" for key in REQUIRED_DIAGNOSTIC_KEYS if key not in row]
    warnings: list[str] = []
    if row.get("stackability_passed") is False:
        errors.append("stackability_passed is false")
    load = row.get("router_load_by_candidate")
    if isinstance(load, dict):
        for name in MULTIMODAL_CANDIDATE_NAMES:
            if name not in load:
                errors.append(f"router_load_by_candidate missing {name}")
    _require_nested_keys(row, "router_logit_parts", REQUIRED_ROUTER_LOGIT_PARTS, errors)
    _require_nested_keys(row, "candidate_loss", MULTIMODAL_CANDIDATE_NAMES, errors)
    _require_nested_keys(row, "adapter_params", REQUIRED_ADAPTER_PARAM_KEYS, errors)
    _require_nested_keys(row, "memory_slot_norm", MULTIMODAL_CANDIDATE_NAMES, errors)
    return DiagnosticValidationReport(ok=not errors, errors=errors, warnings=warnings)


def summarize_diagnostic_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [validate_diagnostic_row(row) for row in rows]
    return {
        "row_count": len(rows),
        "valid_row_count": sum(1 for report in reports if report.ok),
        "errors": [error for report in reports for error in report.errors],
        "required_keys": REQUIRED_DIAGNOSTIC_KEYS,
        "candidate_diagnostic_keys": CANDIDATE_DIAGNOSTIC_KEYS,
    }


def _require_nested_keys(row: dict[str, Any], parent: str, keys: tuple[str, ...], errors: list[str]) -> None:
    value = row.get(parent)
    if not isinstance(value, dict):
        return
    for key in keys:
        if key not in value:
            errors.append(f"{parent} missing {key}")
