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
            losses["spectral_mode_kl"] = _distribution_kl(
                params["spectral"].spectral_mode_logits,
                _expand_vector(tensors["spectral_mode_logits"], batch.target_q),
                temperature,
            )
            losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(params["spectral"].spectral_mode_logits)
        if "separable" in params and params["separable"] is not None and params["separable"].separable_rank_logits is not None:
            losses["separable_rank_kl"] = _distribution_kl(
                params["separable"].separable_rank_logits,
                _expand_vector(tensors["separable_rank_logits"], batch.target_q),
                temperature,
            )
            losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(params["separable"].separable_rank_logits)
        scale_terms = []
        bias_terms = []
        for name in primitive_names:
            if name not in {"spectral", "local", "separable"}:
                continue
            param = params.get(name)
            if param is None:
                continue
            if param.scale is not None and "gain" in tensors:
                scale_terms.append(F.smooth_l1_loss(param.scale, _expand_scalar(tensors["gain"], batch.target_q)))
                losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(param.scale)
            if param.bias is not None and "bias" in tensors:
                bias_terms.append(F.smooth_l1_loss(param.bias, _expand_scalar(tensors["bias"], batch.target_q)))
                losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(param.bias)
        if scale_terms:
            losses["gain_huber"] = torch.stack(scale_terms).mean()
        if bias_terms:
            losses["bias_huber"] = torch.stack(bias_terms).mean()
        local = params.get("local")
        if local is not None and local.local_lengthscale is not None and "lengthscale" in tensors:
            pred_lengthscale = F.softplus(local.local_lengthscale) + 1e-3
            true_lengthscale = _expand_scalar(tensors["lengthscale"], batch.target_q).clamp_min(1e-6)
            losses["local_lengthscale_log_huber"] = F.smooth_l1_loss(pred_lengthscale.log(), true_lengthscale.log())
            losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(local.local_lengthscale)
        if local is not None and local.local_shift is not None and "shift" in tensors:
            losses["local_shift_huber"] = F.smooth_l1_loss(local.local_shift, _expand_scalar(tensors["shift"], batch.target_q))
            losses["param_scope_loss"] = losses["param_scope_loss"] + _q_variance(local.local_shift)

    true_weights = _reordered_true_weights(hints, primitive_names, output.y_hat.device)
    true_outputs = _reordered_true_outputs(hints, primitive_names, output.y_hat.device)
    learned_outputs = output.diagnostics.get("per_primitive_outputs_train")
    if has_controlled_adapter and isinstance(learned_outputs, torch.Tensor) and true_outputs is not None and learned_outputs.shape == true_outputs.shape:
        losses["primitive_output_loss"] = _relative_mse(learned_outputs, true_outputs)
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


def _distribution_kl(pred_logits: torch.Tensor, true_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    true_probs = torch.softmax(true_logits / temp, dim=-1)
    pred_log_probs = torch.log_softmax(pred_logits / temp, dim=-1)
    return (true_probs * (true_probs.clamp_min(1e-12).log() - pred_log_probs)).sum(dim=-1).mean()


def _relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction - target).square().mean()
    denominator = target.square().mean().clamp_min(1e-8)
    return numerator / denominator


def _q_variance(value: torch.Tensor) -> torch.Tensor:
    if value.shape[1] <= 1:
        return value.sum() * 0.0
    return value.var(dim=1, unbiased=False).mean()


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
