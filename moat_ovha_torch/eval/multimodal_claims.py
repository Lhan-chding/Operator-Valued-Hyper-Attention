from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PDE_FORBIDDEN_PATTERNS = (
    "strongest neural operator",
    "outperforms fno",
    "beats all fno",
    "beats fno",
    "outperforms deeponet",
    "beats deeponet",
    "outperforms icon",
    "beats icon",
    "proves multimodal",
    "proves the multimodal",
)

PDE_REQUIRED_FRAMING = (
    "architecture feasibility",
    "not used as the main top-conference benchmark claim",
    "multimodal typed-token relation-operator",
)

MULTIMODAL_MAIN_EVIDENCE_KEYS = (
    "controlled_multimodal_passed",
    "region_text_public_passed",
    "sentiment_public_passed",
    "robustness_passed",
    "multi_seed_statistics_passed",
)


@dataclass(frozen=True)
class ClaimValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_claim_text(
    text: str,
    *,
    evidence_scope: str,
    evidence: dict[str, Any] | None = None,
) -> ClaimValidationReport:
    lowered = " ".join(text.lower().split())
    errors: list[str] = []
    warnings: list[str] = []
    if evidence_scope == "pdebench":
        _validate_pde_claims(lowered, errors, warnings)
    elif evidence_scope == "multimodal_main":
        _validate_multimodal_main_claims(lowered, evidence or {}, errors)
    else:
        errors.append(f"unknown evidence_scope: {evidence_scope}")
    return ClaimValidationReport(ok=not errors, errors=errors, warnings=warnings)


def _validate_pde_claims(lowered: str, errors: list[str], warnings: list[str]) -> None:
    for pattern in PDE_FORBIDDEN_PATTERNS:
        if _contains_unnegated(lowered, pattern):
            errors.append("PDEBench may not support main benchmark superiority claims")
            break
    if "pdebench" in lowered:
        for phrase in PDE_REQUIRED_FRAMING:
            if phrase not in lowered:
                warnings.append(f"PDEBench framing should mention: {phrase}")


def _contains_unnegated(text: str, pattern: str) -> bool:
    start = text.find(pattern)
    while start != -1:
        prefix = text[max(0, start - 32) : start]
        if not any(marker in prefix for marker in ("not ", "does not ", "cannot ", "must not ", "not as ")):
            return True
        start = text.find(pattern, start + len(pattern))
    return False


def _validate_multimodal_main_claims(lowered: str, evidence: dict[str, Any], errors: list[str]) -> None:
    if "typed-token operator-valued attention" not in lowered and "typed multimodal token" not in lowered:
        return
    missing = [key for key in MULTIMODAL_MAIN_EVIDENCE_KEYS if evidence.get(key) is not True]
    if missing:
        errors.append("missing evidence for multimodal main claim: " + ", ".join(missing))
