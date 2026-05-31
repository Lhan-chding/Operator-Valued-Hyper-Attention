from __future__ import annotations

from typing import Any


MULTIMODAL_CANDIDATE_NAMES = ("TLEO", "SPO", "LRIO", "CATO")
ORACLE_MATRIX_CELLS = ("learned_learned", "true_learned", "learned_true", "true_true")


def evaluate_oracle_matrix(
    batch,
    *,
    learned_candidate_values: Any | None = None,
    learned_router_weights: Any | None = None,
) -> dict[str, dict[str, object] | float]:
    if batch.hidden is None:
        raise ValueError("controlled oracle matrix requires batch.hidden")
    required = ("true_candidate_values", "true_router_weights")
    missing = [name for name in required if name not in batch.hidden]
    if missing:
        raise ValueError(f"controlled oracle matrix missing hidden truth: {missing}")
    true_candidate_values = batch.hidden["true_candidate_values"]
    true_router_weights = batch.hidden["true_router_weights"]
    candidate_gap_values = learned_candidate_values if learned_candidate_values is not None else true_candidate_values
    if learned_candidate_values is None:
        learned_candidate_values = true_candidate_values
    if learned_router_weights is None:
        learned_router_weights = true_router_weights
    _validate_oracle_tensor_shapes(
        target_y=batch.target_y,
        true_candidate_values=true_candidate_values,
        true_router_weights=true_router_weights,
        learned_candidate_values=learned_candidate_values,
        learned_router_weights=learned_router_weights,
    )
    report: dict[str, dict[str, object] | float] = {
        "learned_learned": _oracle_cell(
            _mse(_mix(learned_router_weights, learned_candidate_values), batch.target_y),
            "learned router + learned adapter/candidates",
        ),
        "true_learned": _oracle_cell(
            _mse(_mix(true_router_weights, learned_candidate_values), batch.target_y),
            "true router + learned adapter/candidates",
        ),
        "learned_true": _oracle_cell(
            _mse(_mix(learned_router_weights, true_candidate_values), batch.target_y),
            "learned router + true adapter/candidates",
        ),
        "true_true": _oracle_cell(
            _mse(_mix(true_router_weights, true_candidate_values), batch.target_y),
            "true router + true adapter/candidates; candidate expressivity upper bound",
        ),
    }
    for index, name in enumerate(MULTIMODAL_CANDIDATE_NAMES):
        candidate_mse = _mse(candidate_gap_values[..., index, :], batch.target_y)
        report[f"{name}_oracle_gap"] = candidate_mse
    if getattr(batch, "task_type", "") == "rceo_reliability_corruption":
        report["rceo_prior_effect"] = _rceo_prior_effect(batch)
    return report


def _mix(router_weights: Any, candidate_values: Any) -> Any:
    return (router_weights.unsqueeze(-1) * candidate_values).sum(dim=-2)


def _mse(prediction: Any, target: Any) -> Any:
    return (prediction - target).square().mean()


def _validate_oracle_tensor_shapes(
    *,
    target_y: Any,
    true_candidate_values: Any,
    true_router_weights: Any,
    learned_candidate_values: Any,
    learned_router_weights: Any,
) -> None:
    target_shape = _tensor_shape(target_y)
    if len(target_shape) != 3:
        raise ValueError(f"controlled oracle target_y must have shape [B,Q,Dy], got {target_shape}")
    _validate_candidate_value_shape(
        "true_candidate_values",
        true_candidate_values,
        expected_target_shape=target_shape,
    )
    _validate_candidate_value_shape(
        "learned_candidate_values",
        learned_candidate_values,
        expected_target_shape=target_shape,
    )
    _validate_router_weight_shape(
        "true_router_weights",
        true_router_weights,
        expected_prefix=target_shape[:2],
    )
    _validate_router_weight_shape(
        "learned_router_weights",
        learned_router_weights,
        expected_prefix=target_shape[:2],
    )


def _validate_candidate_value_shape(
    name: str,
    value: Any,
    *,
    expected_target_shape: tuple[int, ...],
) -> None:
    shape = _tensor_shape(value)
    expected = (
        expected_target_shape[0],
        expected_target_shape[1],
        len(MULTIMODAL_CANDIDATE_NAMES),
        expected_target_shape[2],
    )
    if shape != expected:
        raise ValueError(f"controlled oracle {name} must have shape [B,Q,P,Dy]={expected}, got {shape}")


def _validate_router_weight_shape(
    name: str,
    value: Any,
    *,
    expected_prefix: tuple[int, int],
) -> None:
    shape = _tensor_shape(value)
    expected = (expected_prefix[0], expected_prefix[1], len(MULTIMODAL_CANDIDATE_NAMES))
    if shape != expected:
        raise ValueError(f"controlled oracle {name} must have shape [B,Q,P]={expected}, got {shape}")


def _tensor_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


def controlled_row_from_oracle_report(
    family: str,
    active_operator: str,
    oracle_report: dict[str, Any],
    *,
    router_accuracy: float = 1.0,
    stackability_passed: bool = True,
    no_operator_memory_delta: float = 0.0,
    no_hyper_adapter_delta: float = 0.0,
    no_evidence_router_delta: float = 0.0,
    no_reliability_prior_delta: float = 0.0,
    memory_only_router_delta: float = 0.0,
    evidence_only_router_delta: float = 0.0,
    no_lrio_delta: float | None = None,
    no_rceo_delta: float | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "family": family,
        "active_operator": active_operator,
        "oracle_matrix": {
            cell: _controlled_matrix_cell(oracle_report.get(cell, {}))
            for cell in ORACLE_MATRIX_CELLS
        },
        "specialist_loss": _loss_value(oracle_report.get("true_learned", {})),
        "router_accuracy": float(router_accuracy),
        "stackability_passed": bool(stackability_passed),
        "no_operator_memory_delta": float(no_operator_memory_delta),
        "no_hyper_adapter_delta": float(no_hyper_adapter_delta),
        "no_evidence_router_delta": float(no_evidence_router_delta),
        "no_reliability_prior_delta": float(no_reliability_prior_delta),
        "memory_only_router_delta": float(memory_only_router_delta),
        "evidence_only_router_delta": float(evidence_only_router_delta),
    }
    if no_lrio_delta is not None:
        row["no_lrio_delta"] = float(no_lrio_delta)
    if no_rceo_delta is not None:
        row["no_rceo_delta"] = float(no_rceo_delta)
    for name in MULTIMODAL_CANDIDATE_NAMES:
        row[f"{name}_oracle_gap"] = _as_float(oracle_report.get(f"{name}_oracle_gap", 0.0))
    if "rceo_prior_effect" in oracle_report:
        row["rceo_prior_effect"] = _as_float(oracle_report["rceo_prior_effect"])
    if diagnostics:
        row.update(diagnostics)
    return row


def _oracle_cell(loss: Any, purpose: str) -> dict[str, Any]:
    return {"mse": loss, "loss": loss, "purpose": purpose}


def _controlled_matrix_cell(value: Any) -> dict[str, float]:
    return {"loss": _loss_value(value)}


def _loss_value(value: Any) -> float:
    if isinstance(value, dict):
        if "loss" in value:
            return _as_float(value["loss"])
        if "mse" in value:
            return _as_float(value["mse"])
        return float("inf")
    return _as_float(value)


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    shape = getattr(value, "shape", ())
    if shape not in ((), None) and hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _rceo_prior_effect(batch) -> Any:
    hidden = batch.hidden or {}
    corruption = hidden.get("true_corruption_level")
    reliability = hidden.get("true_reliability")
    if corruption is None or reliability is None:
        return 0.0
    corruption_mean = corruption.reshape(corruption.shape[0], -1).mean(dim=1)
    reliability_mean = reliability.reshape(reliability.shape[0], -1).mean(dim=1)
    return (corruption_mean * (1.0 - reliability_mean)).mean()
