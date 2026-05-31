from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


REQUIRED_DIAGNOSTIC_KEYS = (
    "router_entropy",
    "router_load_by_candidate",
    "router_logit_parts",
    "candidate_loss",
    "adapter_params",
    "memory_slot_norm",
    "candidate_diagnostics",
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
    "TLEO": ("lengthscale", "local_entropy", "local_window_size", "candidate_loss"),
    "SPO": ("prototype_entropy", "top_prototype", "prototype_temperature", "candidate_loss"),
    "LRIO": ("rank_entropy", "rank_top_k", "pair_interaction_strength", "candidate_loss"),
    "CATO": ("alignment_entropy", "top_k_alignment", "transport_marginal_error", "candidate_loss"),
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
    if "stackability_passed" in row and row.get("stackability_passed") is not True:
        errors.append("stackability_passed must be true")
    if "router_entropy" in row and _finite_float(row.get("router_entropy")) is None:
        errors.append("router_entropy must be finite")
    load = row.get("router_load_by_candidate")
    if isinstance(load, dict):
        for name in MULTIMODAL_CANDIDATE_NAMES:
            if name not in load:
                errors.append(f"router_load_by_candidate missing {name}")
        _validate_probability_map(load, "router_load_by_candidate", MULTIMODAL_CANDIDATE_NAMES, errors)
    elif "router_load_by_candidate" in row:
        errors.append("router_load_by_candidate must be an object keyed by candidate")
    _require_nested_keys(row, "router_logit_parts", REQUIRED_ROUTER_LOGIT_PARTS, errors)
    _require_nested_keys(row, "candidate_loss", MULTIMODAL_CANDIDATE_NAMES, errors)
    _require_nested_keys(row, "adapter_params", REQUIRED_ADAPTER_PARAM_KEYS, errors)
    _require_nested_keys(row, "memory_slot_norm", MULTIMODAL_CANDIDATE_NAMES, errors)
    _validate_finite_map(row.get("router_logit_parts"), "router_logit_parts", REQUIRED_ROUTER_LOGIT_PARTS, errors)
    _validate_non_negative_map(row.get("candidate_loss"), "candidate_loss", MULTIMODAL_CANDIDATE_NAMES, errors)
    _validate_finite_map(row.get("adapter_params"), "adapter_params", REQUIRED_ADAPTER_PARAM_KEYS, errors)
    _validate_non_negative_map(row.get("memory_slot_norm"), "memory_slot_norm", MULTIMODAL_CANDIDATE_NAMES, errors)
    _validate_candidate_diagnostics(row, errors)
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


def _validate_candidate_diagnostics(row: dict[str, Any], errors: list[str]) -> None:
    diagnostics = row.get("candidate_diagnostics")
    if not isinstance(diagnostics, dict):
        errors.append("candidate_diagnostics must be an object keyed by candidate/support module")
        return
    for candidate, keys in CANDIDATE_DIAGNOSTIC_KEYS.items():
        candidate_values = diagnostics.get(candidate)
        if not isinstance(candidate_values, dict):
            errors.append(f"candidate_diagnostics missing {candidate}")
            continue
        for key in keys:
            if key not in candidate_values:
                errors.append(f"candidate_diagnostics.{candidate} missing {key}")
        _validate_candidate_diagnostic_values(candidate, candidate_values, errors)


def _validate_candidate_diagnostic_values(candidate: str, values: dict[str, Any], errors: list[str]) -> None:
    if candidate == "RCEO" and "modality_reliability" in values:
        value = _finite_float(values.get("modality_reliability"))
        if value is None or value < 0.0 or value > 1.0:
            errors.append("candidate_diagnostics.RCEO.modality_reliability must be a finite probability")
    for key in (
        "candidate_loss",
        "lengthscale",
        "local_entropy",
        "prototype_entropy",
        "prototype_temperature",
        "rank_entropy",
        "pair_interaction_strength",
        "alignment_entropy",
        "transport_marginal_error",
        "reliability_bias_norm",
        "corruption_response",
    ):
        if key not in values:
            continue
        numeric = _finite_float(values.get(key))
        if numeric is None or numeric < 0.0:
            errors.append(f"candidate_diagnostics.{candidate}.{key} must be finite non-negative")


def _validate_probability_map(
    values: dict[str, Any],
    parent: str,
    required_keys: tuple[str, ...],
    errors: list[str],
    *,
    tolerance: float = 1e-6,
) -> None:
    total = 0.0
    has_all_finite_values = True
    for key in required_keys:
        numeric = _finite_float(values.get(key))
        if numeric is None:
            errors.append(f"{parent}.{key} must be a finite probability")
            has_all_finite_values = False
            continue
        if numeric < 0.0 or numeric > 1.0:
            errors.append(f"{parent}.{key} must be a finite probability")
        total += numeric
    if has_all_finite_values and not math.isclose(total, 1.0, rel_tol=tolerance, abs_tol=tolerance):
        errors.append(f"{parent} values must sum to 1")


def _validate_finite_map(value: Any, parent: str, keys: tuple[str, ...], errors: list[str]) -> None:
    if not isinstance(value, dict):
        return
    for key in keys:
        if key in value and _finite_float(value.get(key)) is None:
            errors.append(f"{parent}.{key} must be finite")


def _validate_non_negative_map(value: Any, parent: str, keys: tuple[str, ...], errors: list[str]) -> None:
    if not isinstance(value, dict):
        return
    for key in keys:
        if key not in value:
            continue
        numeric = _finite_float(value.get(key))
        if numeric is None or numeric < 0.0:
            errors.append(f"{parent}.{key} must be finite non-negative")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
