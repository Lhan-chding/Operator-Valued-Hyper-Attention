from __future__ import annotations

from typing import Any


MULTIMODAL_CANDIDATE_NAMES = ("TLEO", "SPO", "LRIO", "CATO")
ORACLE_MATRIX_CELLS = ("learned_learned", "true_learned", "learned_true", "true_true")


def evaluate_oracle_matrix(batch) -> dict[str, dict[str, object] | float]:
    if batch.hidden is None:
        raise ValueError("controlled oracle matrix requires batch.hidden")
    required = ("true_candidate_values", "true_router_weights")
    missing = [name for name in required if name not in batch.hidden]
    if missing:
        raise ValueError(f"controlled oracle matrix missing hidden truth: {missing}")
    candidate_values = batch.hidden["true_candidate_values"]
    router_weights = batch.hidden["true_router_weights"]
    true_true = (router_weights.unsqueeze(-1) * candidate_values).sum(dim=-2)
    mse = (true_true - batch.target_y).square().mean()
    report: dict[str, dict[str, object] | float] = {
        "true_true": _oracle_cell(mse, "candidate expressivity upper bound"),
        "true_learned": _oracle_cell(mse, "adapter/candidate isolation scaffold"),
        "learned_true": _oracle_cell(mse, "router isolation scaffold"),
        "learned_learned": _oracle_cell(mse, "full model placeholder for trained eval"),
    }
    for index, name in enumerate(MULTIMODAL_CANDIDATE_NAMES):
        candidate_mse = (candidate_values[..., index, :] - batch.target_y).square().mean()
        report[f"{name}_oracle_gap"] = candidate_mse
    if getattr(batch, "task_type", "") == "rceo_reliability_corruption":
        report["rceo_prior_effect"] = _rceo_prior_effect(batch)
    return report


def controlled_row_from_oracle_report(
    family: str,
    active_operator: str,
    oracle_report: dict[str, Any],
    *,
    router_accuracy: float = 1.0,
    stackability_passed: bool = True,
    no_operator_memory_delta: float = 0.0,
    no_hyper_adapter_delta: float = 0.0,
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
