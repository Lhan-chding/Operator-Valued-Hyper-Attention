from __future__ import annotations

from dataclasses import dataclass

from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float
    threshold: float
    direction: str


def controlled_stackability_gate(output) -> GateResult:
    expected = len(MULTIMODAL_CANDIDATE_NAMES)
    actual = output.candidate_values.shape[-2]
    return GateResult("stackability", actual == expected, float(actual), float(expected), "equals")


def router_accuracy_gate(batch, router_weights, threshold: float = 0.80) -> GateResult:
    if batch.hidden is None or "true_active_operator" not in batch.hidden:
        raise ValueError("router accuracy gate requires true_active_operator hidden truth")
    prediction = router_weights.argmax(dim=-1)
    accuracy = (prediction == batch.hidden["true_active_operator"]).float().mean()
    return GateResult("router_active_operator_accuracy", bool(accuracy >= threshold), float(accuracy), threshold, "greater_equal")


def reliability_monotonic_gate(corruption_strength, reliability, tolerance: float = 1e-6) -> GateResult:
    order = corruption_strength.reshape(-1).argsort()
    sorted_reliability = reliability.reshape(-1)[order]
    deltas = sorted_reliability[1:] - sorted_reliability[:-1]
    max_violation = deltas.max() if deltas.numel() else sorted_reliability.new_tensor(0.0)
    return GateResult("rceo_reliability_monotonic", bool(max_violation <= tolerance), float(max_violation), tolerance, "less_equal")


def candidate_oracle_gap_gate(candidate_loss: float, specialist_loss: float, eps: float = 1e-6) -> GateResult:
    threshold = specialist_loss * 1.05 + eps
    return GateResult("candidate_specialist_collapse", candidate_loss <= threshold, candidate_loss, threshold, "less_equal")
