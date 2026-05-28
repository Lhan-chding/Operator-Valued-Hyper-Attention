from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from moat_ovha_torch.data.episodes import EpisodeHiddenInfo, MetaOperatorBatch
from moat_ovha_torch.models.ovha import OVHAOutput


def controlled_v2_adapter_losses(
    output: OVHAOutput,
    hidden: EpisodeHiddenInfo,
    primitive_names: tuple[str, ...],
    batch: MetaOperatorBatch,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    zero = output.y_hat.sum() * 0.0
    hints = hidden.oracle_hints or {}
    tensors = hints.get("true_operator_tensors") or {}
    params = output.adapter_params or {}
    has_controlled_adapter = any(params.get(name) is not None for name in ("spectral", "local", "separable"))
    true_weights = _reordered_true_weights(hints, primitive_names, output.y_hat.device)
    active_weights = _primitive_activity_weights(true_weights, primitive_names, output.y_hat)
    losses: dict[str, torch.Tensor] = {
        "spectral_mode_kl": zero,
        "separable_rank_kl": zero,
        "gain_huber": zero,
        "bias_huber": zero,
        "local_lengthscale_log_huber": zero,
        "local_shift_huber": zero,
        "param_scope_loss": zero,
        "primitive_output_loss": zero,
        "oracle_routed_prediction_loss": zero,
    }

    if tensors and params:
        if "spectral" in params and params["spectral"] is not None and params["spectral"].spectral_mode_logits is not None:
            weights = active_weights.get("spectral")
            losses["spectral_mode_kl"] = _distribution_kl(
                params["spectral"].spectral_mode_logits,
                _expand_vector(tensors["spectral_mode_logits"], batch.target_q),
                temperature,
                weights,
            )
            losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(
                params["spectral"].spectral_mode_logits,
                weights,
            )
        if "separable" in params and params["separable"] is not None and params["separable"].separable_rank_logits is not None:
            weights = active_weights.get("separable")
            losses["separable_rank_kl"] = _distribution_kl(
                params["separable"].separable_rank_logits,
                _expand_vector(tensors["separable_rank_logits"], batch.target_q),
                temperature,
                weights,
            )
            losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(
                params["separable"].separable_rank_logits,
                weights,
            )
        scale_terms: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        bias_terms: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for name in primitive_names:
            if name not in {"spectral", "local", "separable"}:
                continue
            param = params.get(name)
            if param is None:
                continue
            weights = active_weights.get(name)
            if param.scale is not None and "gain" in tensors:
                scale_terms.append((_smooth_l1_per_query(param.scale, _expand_scalar(tensors["gain"], batch.target_q)), weights))
                losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(param.scale, weights)
            if param.bias is not None and "bias" in tensors:
                bias_terms.append((_smooth_l1_per_query(param.bias, _expand_scalar(tensors["bias"], batch.target_q)), weights))
                losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(param.bias, weights)
        if scale_terms:
            losses["gain_huber"] = _weighted_stack_mean(scale_terms)
        if bias_terms:
            losses["bias_huber"] = _weighted_stack_mean(bias_terms)
        local = params.get("local")
        if local is not None and local.local_lengthscale is not None and "lengthscale" in tensors:
            weights = active_weights.get("local")
            pred_lengthscale = F.softplus(local.local_lengthscale) + 1e-3
            true_lengthscale = _expand_scalar(tensors["lengthscale"], batch.target_q).clamp_min(1e-6)
            losses["local_lengthscale_log_huber"] = _smooth_l1_weighted(
                pred_lengthscale.log(),
                true_lengthscale.log(),
                weights,
            )
            losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(local.local_lengthscale, weights)
        if local is not None and local.local_shift is not None and "shift" in tensors:
            weights = active_weights.get("local")
            losses["local_shift_huber"] = _smooth_l1_weighted(
                local.local_shift,
                _expand_scalar(tensors["shift"], batch.target_q),
                weights,
            )
            losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(local.local_shift, weights)

    true_outputs = _reordered_true_outputs(hints, primitive_names, output.y_hat.device)
    learned_outputs = output.diagnostics.get("per_primitive_outputs_train")
    if has_controlled_adapter and isinstance(learned_outputs, torch.Tensor) and true_outputs is not None and learned_outputs.shape == true_outputs.shape:
        losses["primitive_output_loss"] = _relative_mse(learned_outputs, true_outputs, true_weights)
    if has_controlled_adapter and isinstance(learned_outputs, torch.Tensor) and true_weights is not None and true_weights.shape == output.primitive_weights.shape:
        oracle_y = (true_weights.unsqueeze(-1) * learned_outputs).sum(dim=-2)
        losses["oracle_routed_prediction_loss"] = _relative_mse(oracle_y, batch.target_y)

    adapter_terms = [
        losses["spectral_mode_kl"],
        losses["separable_rank_kl"],
        losses["gain_huber"],
        losses["bias_huber"],
        losses["local_lengthscale_log_huber"],
        losses["local_shift_huber"],
    ]
    losses["adapter_param_loss"] = torch.stack(adapter_terms).sum()
    return losses


def _distribution_kl(
    pred_logits: torch.Tensor,
    true_logits: torch.Tensor,
    temperature: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    true_probs = torch.softmax(true_logits / temp, dim=-1)
    pred_log_probs = torch.log_softmax(pred_logits / temp, dim=-1)
    per_query = (true_probs * (true_probs.clamp_min(1e-12).log() - pred_log_probs)).sum(dim=-1)
    return _weighted_mean(per_query, weights)


def _relative_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is not None:
        weights = _expand_weights(weights, prediction)
    numerator = _weighted_mean((prediction - target).square(), weights)
    denominator = _weighted_mean(target.square(), weights).clamp_min(1e-8)
    return numerator / denominator


def _smooth_l1_per_query(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)


def _smooth_l1_weighted(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    return _weighted_mean(_smooth_l1_per_query(prediction, target), weights)


def _weighted_stack_mean(values: list[tuple[torch.Tensor, torch.Tensor | None]]) -> torch.Tensor:
    stacked_values = torch.stack([value for value, _ in values], dim=-1)
    if all(weights is None for _, weights in values):
        return stacked_values.mean()
    stacked_weights = torch.stack(
        [
            torch.ones_like(value) if weights is None else weights.to(device=value.device, dtype=value.dtype)
            for value, weights in values
        ],
        dim=-1,
    )
    return _weighted_mean(stacked_values, stacked_weights)


def _q_variance(value: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    active = _activity_scalar(weights, value)
    if active is not None and bool((active <= 0.0).all()):
        return value.sum() * 0.0
    if value.shape[1] <= 1:
        return value.sum() * 0.0
    variance = value.var(dim=1, unbiased=False).mean(dim=-1)
    if active is None:
        return variance.mean()
    return _weighted_mean(variance, active)


def _weighted_mean(value: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return value.mean()
    weights = _expand_weights(weights.to(device=value.device, dtype=value.dtype), value)
    denominator = weights.sum().clamp_min(1e-8)
    return (value * weights).sum() / denominator


def _expand_weights(weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    expanded = weights
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    return expanded.expand_as(value)


def _activity_scalar(weights: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor | None:
    if weights is None:
        return None
    activity = weights.to(device=value.device, dtype=value.dtype)
    if activity.ndim > 1:
        activity = activity.mean(dim=tuple(range(1, activity.ndim)))
    return activity


def _expand_scalar(value: Any, target_q: torch.Tensor) -> torch.Tensor:
    tensor = _to_device(value, target_q.device).to(dtype=target_q.dtype)
    return tensor.reshape(target_q.shape[0], 1, 1).expand(-1, target_q.shape[1], -1)


def _expand_vector(value: Any, target_q: torch.Tensor) -> torch.Tensor:
    tensor = _to_device(value, target_q.device).to(dtype=target_q.dtype)
    return tensor.reshape(target_q.shape[0], 1, -1).expand(-1, target_q.shape[1], -1)


def _to_device(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _reordered_true_weights(hints: dict[str, Any], primitive_names: tuple[str, ...], device: torch.device) -> torch.Tensor | None:
    true_weights = hints.get("true_component_weight_by_q")
    true_order = tuple(hints.get("primitive_order") or ())
    reorder = _reorder_indices(true_order, primitive_names)
    if true_weights is None or reorder is None:
        return None
    target = _to_device(true_weights, device)
    target = target.index_select(-1, torch.tensor(reorder, device=target.device))
    return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _primitive_activity_weights(
    true_weights: torch.Tensor | None,
    primitive_names: tuple[str, ...],
    reference: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    if true_weights is None:
        return {name: None for name in primitive_names}
    weights = true_weights.to(device=reference.device, dtype=reference.dtype)
    return {name: weights[..., index] for index, name in enumerate(primitive_names)}


def _reordered_true_outputs(hints: dict[str, Any], primitive_names: tuple[str, ...], device: torch.device) -> torch.Tensor | None:
    true_outputs = hints.get("true_primitive_outputs_by_q")
    true_order = tuple(hints.get("primitive_order") or ())
    reorder = _reorder_indices(true_order, primitive_names)
    if true_outputs is None or reorder is None:
        return None
    target = _to_device(true_outputs, device)
    return target.index_select(-2, torch.tensor(reorder, device=target.device))


def _reorder_indices(true_order: tuple[str, ...], model_order: tuple[str, ...]) -> list[int] | None:
    indices = []
    for name in model_order:
        if name not in true_order:
            return None
        indices.append(true_order.index(name))
    return indices
