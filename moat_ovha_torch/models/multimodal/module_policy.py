from __future__ import annotations

from dataclasses import dataclass
from typing import Any


V1_CANDIDATE_STACK = ("TLEO", "SPO", "LRIO", "CATO")
V1_FORBIDDEN_STACK_NAMES = ("RCEO", "MMRO", "CTRO", "TLDO", "OMRO")
SUPPORT_MODULE_POSITIONS = {
    "RCEO": "router_prior_adapter_condition_diagnostics",
    "MMRO": "upstream_conditioner",
    "CTRO": "second_stage_residual",
    "OMRO": "memory_infrastructure",
}
TEMPORAL_MODALITIES = {"audio", "video"}


@dataclass(frozen=True)
class ModulePolicyReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_module_policy(policy: dict[str, Any]) -> ModulePolicyReport:
    version = str(policy.get("version", "v1"))
    candidate_stack = tuple(policy.get("candidate_stack", ()))
    support_modules = dict(policy.get("support_modules", {}))
    modalities = set(policy.get("task_modalities", ()))
    errors: list[str] = []
    warnings: list[str] = []
    if version == "v1":
        _validate_v1(candidate_stack, support_modules, errors)
    elif version == "v2":
        _validate_v2(candidate_stack, support_modules, modalities, errors)
    else:
        errors.append(f"unknown multimodal module policy version: {version}")
    return ModulePolicyReport(ok=not errors, errors=errors, warnings=warnings)


def _validate_v1(candidate_stack: tuple[str, ...], support_modules: dict[str, str], errors: list[str]) -> None:
    if candidate_stack != V1_CANDIDATE_STACK:
        errors.append(f"v1 candidate_stack must be exactly {V1_CANDIDATE_STACK}")
    for forbidden in V1_FORBIDDEN_STACK_NAMES:
        if forbidden in candidate_stack:
            errors.append(f"{forbidden} must not enter v1 candidate_stack")
    rceo_position = support_modules.get("RCEO")
    if rceo_position is not None and rceo_position != SUPPORT_MODULE_POSITIONS["RCEO"]:
        errors.append("RCEO must be router_prior_adapter_condition_diagnostics")
    for module in ("MMRO", "CTRO", "TLDO"):
        if module in candidate_stack or module in support_modules:
            errors.append(f"{module} is v2+ only")
    if support_modules.get("OMRO") not in (None, SUPPORT_MODULE_POSITIONS["OMRO"]):
        errors.append("OMRO is memory infrastructure")


def _validate_v2(
    candidate_stack: tuple[str, ...],
    support_modules: dict[str, str],
    modalities: set[str],
    errors: list[str],
) -> None:
    base = candidate_stack[:4]
    if base != V1_CANDIDATE_STACK:
        errors.append(f"v2 candidate_stack must start with {V1_CANDIDATE_STACK}")
    if "TLDO" in candidate_stack and not (modalities & TEMPORAL_MODALITIES):
        errors.append("TLDO requires temporal modality")
    for module, expected_position in SUPPORT_MODULE_POSITIONS.items():
        actual = support_modules.get(module)
        if actual is not None and actual != expected_position:
            if module == "OMRO":
                errors.append("OMRO is memory infrastructure")
            else:
                errors.append(f"{module} must be {expected_position}")
    if "MMRO" in candidate_stack:
        errors.append("MMRO must be upstream_conditioner, not candidate_stack")
    if "CTRO" in candidate_stack:
        errors.append("CTRO must be second_stage_residual, not candidate_stack")
    if "OMRO" in candidate_stack:
        errors.append("OMRO is memory infrastructure")
