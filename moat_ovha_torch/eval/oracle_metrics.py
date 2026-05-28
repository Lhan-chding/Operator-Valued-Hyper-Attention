from __future__ import annotations

from typing import Any

import torch

from moat_ovha_torch.data.episodes import EpisodeHiddenInfo
from moat_ovha_torch.models.ovha import OVHAOutput
from moat_ovha_torch.train.metrics import relative_l2


def controlled_oracle_metrics(
    output: OVHAOutput,
    target_y: torch.Tensor,
    hidden: EpisodeHiddenInfo,
    primitive_names: tuple[str, ...],
) -> dict[str, float | None]:
    hints = hidden.oracle_hints or {}
    true_weights = hints.get("true_component_weight_by_q")
    true_outputs = hints.get("true_primitive_outputs_by_q")
    true_order = tuple(hints.get("primitive_order") or ())
    if true_weights is None or true_outputs is None or not true_order:
        return _empty_metrics()

    true_weights = _to_device(true_weights, output.y_hat.device)
    true_outputs = _to_device(true_outputs, output.y_hat.device)
    reorder = _reorder_indices(true_order, primitive_names)
    if reorder is None:
        return _empty_metrics()

    true_weights = true_weights.index_select(-1, torch.tensor(reorder, device=true_weights.device))
    true_outputs = true_outputs.index_select(-2, torch.tensor(reorder, device=true_outputs.device))
    if true_weights.shape != output.primitive_weights.shape:
        return _empty_metrics()

    predicted_weights = output.primitive_weights.clamp_min(1e-8)
    normalized_true = true_weights.clamp_min(1e-8)
    normalized_true = normalized_true / normalized_true.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    router_mae = (predicted_weights - normalized_true).abs().mean()
    router_kl = (normalized_true * (normalized_true.log() - predicted_weights.log())).sum(dim=-1).mean()
    router_ce = -(normalized_true * predicted_weights.log()).sum(dim=-1).mean()

    model_outputs = output.diagnostics.get("per_primitive_outputs")
    oracle_router_rel = None
    if isinstance(model_outputs, torch.Tensor) and model_outputs.shape[-2] == len(primitive_names):
        oracle_router_y = (normalized_true.unsqueeze(-1) * model_outputs).sum(dim=-2)
        oracle_router_rel = float(relative_l2(oracle_router_y, target_y).mean().detach().cpu())

    oracle_adapter_y = (predicted_weights.unsqueeze(-1) * true_outputs).sum(dim=-2)
    oracle_adapter_rel = float(relative_l2(oracle_adapter_y, target_y).mean().detach().cpu())
    oracle_router_adapter_y = (normalized_true.unsqueeze(-1) * true_outputs).sum(dim=-2)
    oracle_router_adapter_rel = float(relative_l2(oracle_router_adapter_y, target_y).mean().detach().cpu())
    return {
        "router_true_weight_mae": float(router_mae.detach().cpu()),
        "router_true_weight_kl": float(router_kl.detach().cpu()),
        "router_true_weight_ce": float(router_ce.detach().cpu()),
        "oracle_router_upper_bound_relative_l2": oracle_router_rel,
        "oracle_adapter_upper_bound_relative_l2": oracle_adapter_rel,
        "oracle_router_adapter_upper_bound_relative_l2": oracle_router_adapter_rel,
    }


def _empty_metrics() -> dict[str, None]:
    return {
        "router_true_weight_mae": None,
        "router_true_weight_kl": None,
        "router_true_weight_ce": None,
        "oracle_router_upper_bound_relative_l2": None,
        "oracle_adapter_upper_bound_relative_l2": None,
        "oracle_router_adapter_upper_bound_relative_l2": None,
    }


def _to_device(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _reorder_indices(true_order: tuple[str, ...], model_order: tuple[str, ...]) -> list[int] | None:
    indices = []
    for name in model_order:
        if name not in true_order:
            return None
        indices.append(true_order.index(name))
    return indices
