from __future__ import annotations

from typing import Any

import torch

from moat_ovha_torch.data.episodes import EpisodeHiddenInfo, MetaOperatorBatch
from moat_ovha_torch.models.ovha import OVHAOutput
from moat_ovha_torch.models.primitives.base import PrimitiveParams
from moat_ovha_torch.models.primitives.registry import make_primitive_registry
from moat_ovha_torch.train.metrics import relative_l2


def controlled_oracle_metrics(
    output: OVHAOutput,
    target_y: torch.Tensor,
    hidden: EpisodeHiddenInfo,
    primitive_names: tuple[str, ...],
    batch: MetaOperatorBatch | None = None,
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
    model_true_param_rel = None
    model_router_true_param_rel = None
    if batch is not None:
        model_true_outputs = model_primitive_outputs_from_true_params(batch, hidden, primitive_names)
        if isinstance(model_true_outputs, torch.Tensor) and model_true_outputs.shape == true_outputs.shape:
            model_true_outputs = model_true_outputs.to(output.y_hat.device)
            model_true_param_y = (normalized_true.unsqueeze(-1) * model_true_outputs).sum(dim=-2)
            model_true_param_rel = float(relative_l2(model_true_param_y, target_y).mean().detach().cpu())
            model_router_true_param_y = (predicted_weights.unsqueeze(-1) * model_true_outputs).sum(dim=-2)
            model_router_true_param_rel = float(relative_l2(model_router_true_param_y, target_y).mean().detach().cpu())
    return {
        "router_true_weight_mae": float(router_mae.detach().cpu()),
        "router_true_weight_kl": float(router_kl.detach().cpu()),
        "router_true_weight_ce": float(router_ce.detach().cpu()),
        "oracle_router_upper_bound_relative_l2": oracle_router_rel,
        "oracle_adapter_upper_bound_relative_l2": oracle_adapter_rel,
        "oracle_router_adapter_upper_bound_relative_l2": oracle_router_adapter_rel,
        "model_primitive_true_param_relative_l2": model_true_param_rel,
        "model_router_true_param_relative_l2": model_router_true_param_rel,
    }


def _empty_metrics() -> dict[str, None]:
    return {
        "router_true_weight_mae": None,
        "router_true_weight_kl": None,
        "router_true_weight_ce": None,
        "oracle_router_upper_bound_relative_l2": None,
        "oracle_adapter_upper_bound_relative_l2": None,
        "oracle_router_adapter_upper_bound_relative_l2": None,
        "model_primitive_true_param_relative_l2": None,
        "model_router_true_param_relative_l2": None,
    }


def model_primitive_outputs_from_true_params(
    batch: MetaOperatorBatch,
    hidden: EpisodeHiddenInfo,
    primitive_names: tuple[str, ...],
) -> torch.Tensor | None:
    hints = hidden.oracle_hints or {}
    tensors = hints.get("true_operator_tensors")
    if not tensors or any(name not in {"spectral", "local", "separable"} for name in primitive_names):
        return None
    registry = make_primitive_registry(primitive_names).to(batch.target_q.device)
    params_by_name = {
        name: _primitive_params_from_true_tensors(
            name,
            tensors,
            batch.target_q,
            str(hints.get("controlled_generator_variant", "model_aligned")),
        )
        for name in primitive_names
    }
    outputs = [
        registry[name](batch.target_u, batch.support_grid, batch.target_q, params_by_name[name], memory=None)
        for name in primitive_names
    ]
    return torch.stack(outputs, dim=-2)


def _primitive_params_from_true_tensors(
    primitive_name: str,
    tensors: dict[str, Any],
    target_q: torch.Tensor,
    generator_variant: str,
) -> PrimitiveParams:
    scale = _expand_scalar(tensors["gain"], target_q)
    bias = _expand_scalar(tensors["bias"], target_q)
    if primitive_name == "spectral":
        if generator_variant == "full":
            mode_logits = torch.full(
                (target_q.shape[0], target_q.shape[1], 4),
                -30.0,
                dtype=target_q.dtype,
                device=target_q.device,
            )
            mode_logits[..., 0] = 30.0
            frequency = _expand_scalar(tensors["frequency"], target_q)
            phase = _expand_scalar(tensors["phase"], target_q)
        else:
            mode_logits = _expand_vector(tensors["spectral_mode_logits"], target_q)
            frequency = None
            phase = None
        return PrimitiveParams(
            scale=scale,
            bias=bias,
            spectral_frequency=frequency,
            spectral_phase=phase,
            spectral_mode_logits=mode_logits,
            kernel_params={"frequency": frequency, "phase": phase, "mode_logits": mode_logits},
        )
    if primitive_name == "local":
        lengthscale = _expand_scalar(tensors["lengthscale"], target_q)
        lengthscale_raw = _inverse_softplus(lengthscale.clamp_min(1e-6))
        shift = _expand_scalar(tensors["shift"], target_q) if generator_variant == "full" else torch.zeros_like(lengthscale)
        return PrimitiveParams(
            scale=scale,
            bias=bias,
            local_lengthscale=lengthscale_raw,
            local_shift=shift,
            kernel_params={"lengthscale": lengthscale_raw, "shift": shift},
        )
    rank_logits = _expand_vector(tensors["separable_rank_logits"], target_q)
    return PrimitiveParams(
        scale=scale,
        bias=bias,
        separable_rank_logits=rank_logits,
        kernel_params={"rank_logits": rank_logits},
    )


def _expand_scalar(value: Any, target_q: torch.Tensor) -> torch.Tensor:
    tensor = _to_device(value, target_q.device).to(dtype=target_q.dtype)
    return tensor.reshape(target_q.shape[0], 1, 1).expand(-1, target_q.shape[1], -1)


def _expand_vector(value: Any, target_q: torch.Tensor) -> torch.Tensor:
    tensor = _to_device(value, target_q.device).to(dtype=target_q.dtype)
    return tensor.reshape(target_q.shape[0], 1, -1).expand(-1, target_q.shape[1], -1)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value).clamp_min(1e-12))


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
