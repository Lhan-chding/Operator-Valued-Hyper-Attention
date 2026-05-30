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
        for name in ("TLEO", "SPO", "LRIO", "CATO"):
            if name not in load:
                errors.append(f"router_load_by_candidate missing {name}")
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
