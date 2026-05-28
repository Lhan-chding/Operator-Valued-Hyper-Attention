from __future__ import annotations

import math
from typing import Any, Optional

from moat_ovha_torch.data.episodes import EpisodeHiddenInfo, MetaOperatorBatch, hash_context, hash_model_inputs, hash_target
from moat_ovha_torch.runtime import require_torch


PRIMITIVE_ORDER = ("spectral", "local", "separable")
CONTROLLED_V2_FAMILIES = (
    "single_primitive_representable",
    "query_piecewise_router",
    "context_identifiable_mixture",
    "hyper_parameter_family",
    "same_target_counterfactual",
    "modality_reliability_conflict",
    "history_session_preference",
)
LEGACY_CONTROLLED_FAMILY_ALIASES = {
    "query_piecewise_composition_family": "query_piecewise_router",
    "context_identifiable_mixture_family": "context_identifiable_mixture",
    "anti_single_primitive_family": "anti_single_primitive",
    "confounded_family_pair": "confounded_family_pair",
}
CONTROLLED_STRESS_FAMILIES = CONTROLLED_V2_FAMILIES + tuple(LEGACY_CONTROLLED_FAMILY_ALIASES)


def sample_component_stress_batch(
    seed: int,
    batch_size: int,
    num_demos: int,
    context_points: int,
    support_points: int,
    query_points: int,
    family: str,
    split: str,
    mode: str,
    device: str = "cpu",
    resolution_multiplier: int = 1,
    episode_id: Optional[int] = None,
    generator_variant: str = "model_aligned",
) -> tuple[MetaOperatorBatch, EpisodeHiddenInfo]:
    torch = require_torch()
    if family not in CONTROLLED_STRESS_FAMILIES:
        raise ValueError(f"unknown controlled stress family: {family}")
    if mode not in {"same_function_field", "operator_transfer"}:
        raise ValueError(f"unknown mode: {mode}")
    if generator_variant not in {"model_aligned", "full"}:
        raise ValueError(f"unknown controlled generator variant: {generator_variant}")

    canonical_family = _canonical_family(family)
    episode_id_value = 0 if episode_id is None else int(episode_id)
    generator = torch.Generator(device=_generator_device(device))
    generator.manual_seed(
        _stable_seed(seed, family, split, mode, episode_id_value, batch_size, support_points, query_points, generator_variant)
    )
    support_points = support_points * max(1, resolution_multiplier)
    query_points = query_points * max(1, resolution_multiplier)
    support_grid = torch.linspace(0.0, 1.0, support_points, device=device).view(1, support_points, 1)
    target_q = torch.linspace(0.0, 1.0, query_points, device=device).view(1, query_points, 1).repeat(batch_size, 1, 1)
    context_q = _sample_context_q(torch, batch_size, num_demos, context_points, split, generator, device)
    params = _sample_params(torch, batch_size, split, episode_id_value, generator, device)

    context_u = _sample_functions(torch, batch_size, num_demos, support_grid, generator, device)
    if mode == "same_function_field":
        target_u = context_u[:, 0]
    elif canonical_family == "same_target_counterfactual":
        target_u = _sample_counterfactual_target_functions(torch, seed, batch_size, support_grid, device)
    else:
        target_u = _sample_functions(torch, batch_size, 1, support_grid, generator, device)[:, 0]

    context_y, _, _ = _apply_stress_operator(torch, canonical_family, context_u, support_grid, context_q, params, generator_variant)
    target_y, true_weights, true_outputs = _apply_stress_operator(
        torch,
        canonical_family,
        target_u.unsqueeze(1),
        support_grid,
        target_q.unsqueeze(1),
        params,
        generator_variant,
    )
    target_y = target_y[:, 0]
    true_weights = true_weights[:, 0]
    true_outputs = true_outputs[:, 0]

    context_mask = torch.ones(batch_size, num_demos, context_points, dtype=torch.bool, device=device)
    target_mask = torch.ones(batch_size, target_q.shape[1], dtype=torch.bool, device=device)
    if split in {"confusable_context", "sparse_context", "noisy_context"}:
        context_mask[:, :, max(1, context_points // 2) :] = False
        context_y = context_y.masked_fill(~context_mask.unsqueeze(-1), 0.0)

    batch = MetaOperatorBatch(
        context_u=context_u,
        context_q=context_q,
        context_y=context_y,
        target_u=target_u,
        target_q=target_q,
        target_y=target_y,
        support_grid=support_grid,
        context_mask=context_mask,
        target_mask=target_mask,
    )
    hidden = EpisodeHiddenInfo(
        family=family,
        latent_params={
            "stress_family": family,
            "controlled_v2_family": canonical_family,
            "split": split,
            "generator_variant": generator_variant,
            "true_operator_params": _public_param_summary(params),
        },
        mixture_weights={"spectral_local_separable": _to_python(params["mixture"])},
        oracle_hints={
            "episode_id": episode_id_value,
            "primitive_order": PRIMITIVE_ORDER,
            "true_component_weight_by_q": true_weights.detach().cpu(),
            "true_primitive_outputs_by_q": true_outputs.detach().cpu(),
            "true_operator_params": _public_param_summary(params),
            "modality_reliability": _to_python(params["modality_reliability"]),
            "history_session_preference": _to_python(params["history_session_preference"]),
            "batch_hash": hash_model_inputs(batch),
            "context_hash": hash_context(batch),
            "target_hash": hash_target(batch),
        },
    )
    return batch, hidden


def _canonical_family(family: str) -> str:
    return LEGACY_CONTROLLED_FAMILY_ALIASES.get(family, family)


def _apply_stress_operator(
    torch: Any,
    family: str,
    u: Any,
    support_grid: Any,
    query: Any,
    params: dict[str, Any],
    generator_variant: str,
):
    primitive_outputs = _primitive_outputs(torch, u, support_grid, query, params, generator_variant)
    weights = _family_weights(torch, family, query, params)
    return (primitive_outputs * weights.unsqueeze(-1)).sum(dim=-2), weights, primitive_outputs


def _primitive_outputs(torch: Any, u: Any, support_grid: Any, query: Any, params: dict[str, Any], generator_variant: str):
    return torch.cat(
        [
            _spectral(torch, u, support_grid, query, params, generator_variant).unsqueeze(-2),
            _local(torch, u, support_grid, query, params, generator_variant).unsqueeze(-2),
            _separable(torch, u, support_grid, query, params).unsqueeze(-2),
        ],
        dim=-2,
    )


def _family_weights(torch: Any, family: str, query: Any, params: dict[str, Any]):
    if family == "single_primitive_representable":
        weights = torch.nn.functional.one_hot(params["primitive_index"], num_classes=3).float()
        return weights.view(weights.shape[0], 1, 1, 3).expand(query.shape[0], query.shape[1], query.shape[2], 3)
    if family == "query_piecewise_router":
        return _query_piecewise_weights(torch, query)
    if family in {"context_identifiable_mixture", "hyper_parameter_family", "same_target_counterfactual"}:
        return params["mixture"].view(query.shape[0], 1, 1, 3).expand(query.shape[0], query.shape[1], query.shape[2], 3)
    if family == "modality_reliability_conflict":
        reliability = params["modality_reliability"]
        weights = reliability / reliability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights.view(query.shape[0], 1, 1, 3).expand(query.shape[0], query.shape[1], query.shape[2], 3)
    if family == "history_session_preference":
        weights = params["history_session_preference"]
        return weights.view(query.shape[0], 1, 1, 3).expand(query.shape[0], query.shape[1], query.shape[2], 3)
    if family == "anti_single_primitive":
        return _anti_single_weights(torch, query)
    if family == "confounded_family_pair":
        base = params["mixture"].view(query.shape[0], 1, 1, 3).expand(query.shape[0], query.shape[1], query.shape[2], 3)
        contrast = _confounded_contrast(torch, query, params["confound_sign"])
        weights = (base + contrast).clamp_min(1e-4)
        return weights / weights.sum(dim=-1, keepdim=True)
    raise ValueError(f"unknown controlled v2 family: {family}")


def _query_piecewise_weights(torch: Any, query: Any):
    q = query[..., 0]
    sharpness = 30.0
    spectral = torch.sigmoid(sharpness * (1.0 / 3.0 - q))
    local = torch.sigmoid(sharpness * (q - 1.0 / 3.0)) * torch.sigmoid(sharpness * (2.0 / 3.0 - q))
    separable = torch.sigmoid(sharpness * (q - 2.0 / 3.0))
    weights = torch.stack([spectral, local, separable], dim=-1).clamp_min(1e-4)
    return weights / weights.sum(dim=-1, keepdim=True)


def _anti_single_weights(torch: Any, query: Any):
    q = query[..., 0]
    raw = torch.stack(
        [
            0.35 + 0.25 * torch.sin(2.0 * math.pi * q),
            0.35 + 0.25 * torch.sin(2.0 * math.pi * q + 2.0 * math.pi / 3.0),
            0.35 + 0.25 * torch.sin(2.0 * math.pi * q + 4.0 * math.pi / 3.0),
        ],
        dim=-1,
    ).clamp_min(1e-4)
    return raw / raw.sum(dim=-1, keepdim=True)


def _confounded_contrast(torch: Any, query: Any, sign: Any):
    q = query[..., 0]
    gate = torch.sigmoid(40.0 * (q - 0.62)).unsqueeze(-1)
    direction = torch.tensor([0.22, -0.16, -0.06], device=query.device).view(1, 1, 1, 3)
    return gate * sign.view(sign.shape[0], 1, 1, 1) * direction


def _sample_context_q(torch: Any, batch_size: int, num_demos: int, context_points: int, split: str, generator: Any, device: str):
    if split in {"confusable_context", "sparse_context"}:
        base = torch.linspace(0.2, 0.4, context_points, device=device)
    else:
        base = torch.linspace(0.0, 1.0, context_points, device=device)
    jitter = 0.005 * torch.randn(batch_size, num_demos, context_points, 1, generator=generator, device=device)
    return (base.view(1, 1, context_points, 1) + jitter).clamp(0.0, 1.0)


def _sample_functions(torch: Any, batch_size: int, count: int, support_grid: Any, generator: Any, device: str):
    grid = support_grid.view(1, 1, -1, 1)
    freq = torch.randint(1, 5, (batch_size, count, 1, 1), generator=generator, device=device).float()
    phase = 2.0 * math.pi * torch.rand(batch_size, count, 1, 1, generator=generator, device=device)
    amp = 0.7 + 0.6 * torch.rand(batch_size, count, 1, 1, generator=generator, device=device)
    return amp * torch.sin(2.0 * math.pi * freq * grid + phase) + 0.25 * torch.cos(math.pi * (freq + 1.0) * grid)


def _sample_counterfactual_target_functions(torch: Any, seed: int, batch_size: int, support_grid: Any, device: str):
    generator = torch.Generator(device=_generator_device(device))
    generator.manual_seed(_stable_seed(seed, "same_target_counterfactual", "target_u"))
    return _sample_functions(torch, batch_size, 1, support_grid, generator, device)[:, 0]


def _sample_params(torch: Any, batch_size: int, split: str, episode_id: int, generator: Any, device: str) -> dict[str, Any]:
    holdout = 0.7 if split == "parameter_holdout" else 0.0
    raw_mix = torch.rand(batch_size, 3, generator=generator, device=device)
    raw_reliability = 0.1 + torch.rand(batch_size, 3, generator=generator, device=device)
    raw_history = 0.1 + torch.rand(batch_size, 3, generator=generator, device=device)
    return {
        "gain": 0.8 + holdout + 0.4 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "bias": 0.05 * torch.randn(batch_size, 1, 1, 1, generator=generator, device=device),
        "frequency": 1.0 + holdout + 2.0 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "phase": 2.0 * math.pi * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "spectral_mode_logits": torch.randn(batch_size, 4, generator=generator, device=device),
        "lengthscale": 0.08 + 0.25 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "shift": -0.15 + 0.3 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "separable_rank_logits": torch.randn(batch_size, 4, generator=generator, device=device),
        "mixture": raw_mix / raw_mix.sum(dim=-1, keepdim=True),
        "primitive_index": (torch.arange(batch_size, device=device) + episode_id) % 3,
        "modality_reliability": raw_reliability / raw_reliability.sum(dim=-1, keepdim=True),
        "history_session_preference": raw_history / raw_history.sum(dim=-1, keepdim=True),
        "confound_sign": torch.where(torch.rand(batch_size, generator=generator, device=device) > 0.5, 1.0, -1.0),
    }


def _spectral(torch: Any, u: Any, support_grid: Any, query: Any, params: dict[str, Any], generator_variant: str):
    grid = support_grid.view(1, 1, 1, -1, 1)
    q = query.unsqueeze(-2)
    if generator_variant == "full":
        frequency = params["frequency"].view(u.shape[0], 1, 1, 1, 1)
        phase = params["phase"].view(u.shape[0], 1, 1, 1, 1)
        kernel = torch.cos(2.0 * math.pi * frequency * (q - grid) + phase)
        value = (kernel * u.unsqueeze(-3)).mean(dim=-2)
    else:
        outputs = []
        for index in range(1, 5):
            kernel = torch.cos(2.0 * math.pi * index * (q - grid))
            outputs.append((kernel * u.unsqueeze(-3)).mean(dim=-2))
        stacked = torch.stack(outputs, dim=-2)
        weights = torch.softmax(params["spectral_mode_logits"], dim=-1).view(u.shape[0], 1, 1, 4, 1)
        value = (stacked * weights).sum(dim=-2)
    return params["gain"].view(u.shape[0], 1, 1, 1) * value + params["bias"].view(u.shape[0], 1, 1, 1)


def _local(torch: Any, u: Any, support_grid: Any, query: Any, params: dict[str, Any], generator_variant: str):
    grid = support_grid.view(1, 1, 1, -1, 1)
    q = query.unsqueeze(-2)
    lengthscale = params["lengthscale"].view(u.shape[0], 1, 1, 1, 1).abs() + 1e-3
    shift = params["shift"].view(u.shape[0], 1, 1, 1, 1) if generator_variant == "full" else 0.0
    kernel = torch.exp(-((q - grid - shift) ** 2) / (2.0 * lengthscale**2))
    kernel = kernel / kernel.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return params["gain"].view(u.shape[0], 1, 1, 1) * (kernel * u.unsqueeze(-3)).sum(dim=-2) + params["bias"].view(
        u.shape[0], 1, 1, 1
    )


def _separable(torch: Any, u: Any, support_grid: Any, query: Any, params: dict[str, Any]):
    grid = support_grid.view(1, 1, -1, 1)
    branches = [
        u.mean(dim=-2, keepdim=True).expand(-1, -1, query.shape[2], -1),
        (u * grid).mean(dim=-2, keepdim=True) * query,
        (u * torch.sin(math.pi * grid)).mean(dim=-2, keepdim=True) * torch.sin(math.pi * query),
        (u * torch.cos(2.0 * math.pi * grid)).mean(dim=-2, keepdim=True) * torch.cos(2.0 * math.pi * query),
    ]
    stacked = torch.stack(branches, dim=-2)
    weights = torch.softmax(params["separable_rank_logits"], dim=-1).view(u.shape[0], 1, 1, 4, 1)
    value = (stacked * weights).sum(dim=-2)
    return params["gain"].view(u.shape[0], 1, 1, 1) * value + params["bias"].view(u.shape[0], 1, 1, 1)


def _generator_device(device: str) -> str:
    device_text = str(device)
    if device_text.startswith("cuda"):
        return device_text
    return "cpu"


def _stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % (2**31)


def _public_param_summary(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _to_python(value)
        for key, value in params.items()
        if key
        in {
            "gain",
            "bias",
            "frequency",
            "phase",
            "spectral_mode_logits",
            "lengthscale",
            "shift",
            "separable_rank_logits",
            "mixture",
            "primitive_index",
            "modality_reliability",
            "history_session_preference",
            "confound_sign",
        }
    }


def _to_python(value: Optional[Any]) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.reshape(-1)[:32].tolist()
    return value
