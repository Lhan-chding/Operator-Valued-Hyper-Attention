from __future__ import annotations

import math
from typing import Any

from moat_ovha_torch.data.episodes import EpisodeHiddenInfo, MetaOperatorBatch
from moat_ovha_torch.runtime import require_torch


CONTROLLED_STRESS_FAMILIES = (
    "query_piecewise_composition_family",
    "context_identifiable_mixture_family",
    "anti_single_primitive_family",
    "confounded_family_pair",
)


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
) -> tuple[MetaOperatorBatch, EpisodeHiddenInfo]:
    torch = require_torch()
    if family not in CONTROLLED_STRESS_FAMILIES:
        raise ValueError(f"unknown controlled stress family: {family}")
    if mode not in {"same_function_field", "operator_transfer"}:
        raise ValueError(f"unknown mode: {mode}")

    generator = torch.Generator(device=_generator_device(device))
    generator.manual_seed(_stable_seed(seed, family, split, mode, batch_size, support_points, query_points))
    support_points = support_points * max(1, resolution_multiplier)
    query_points = query_points * max(1, resolution_multiplier)
    support_grid = torch.linspace(0.0, 1.0, support_points, device=device).view(1, support_points, 1)
    target_q = torch.linspace(0.0, 1.0, query_points, device=device).view(1, query_points, 1).repeat(batch_size, 1, 1)
    context_q = _sample_context_q(torch, batch_size, num_demos, context_points, split, generator, device)
    params = _sample_params(torch, batch_size, split, generator, device)

    context_u = _sample_functions(torch, batch_size, num_demos, support_grid, generator, device)
    if mode == "same_function_field":
        target_u = context_u[:, 0]
    else:
        target_u = _sample_functions(torch, batch_size, 1, support_grid, generator, device)[:, 0]

    context_y, _ = _apply_stress_operator(torch, family, context_u, support_grid, context_q, params)
    target_y, true_weights = _apply_stress_operator(torch, family, target_u.unsqueeze(1), support_grid, target_q.unsqueeze(1), params)
    target_y = target_y[:, 0]
    true_weights = true_weights[:, 0]

    context_mask = torch.ones(batch_size, num_demos, context_points, dtype=torch.bool, device=device)
    target_mask = torch.ones(batch_size, target_q.shape[1], dtype=torch.bool, device=device)
    if split in {"confusable_context", "sparse_context"}:
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
            "split": split,
            "mixture": params["mixture"].detach().cpu().tolist(),
        },
        mixture_weights={"spectral_local_separable": params["mixture"].detach().cpu().tolist()},
        oracle_hints={"true_component_weight_by_q": true_weights.detach().cpu()},
    )
    return batch, hidden


def _apply_stress_operator(torch: Any, family: str, u: Any, support_grid: Any, query: Any, params: dict[str, Any]):
    spectral = _spectral(torch, u, support_grid, query, params)
    local = _local(torch, u, support_grid, query, params)
    separable = _separable(torch, u, support_grid, query, params)
    if family == "query_piecewise_composition_family":
        weights = _query_piecewise_weights(torch, query)
    elif family == "context_identifiable_mixture_family":
        weights = params["mixture"].view(u.shape[0], 1, 1, 3).expand(u.shape[0], u.shape[1], query.shape[2], 3)
    elif family == "anti_single_primitive_family":
        weights = _anti_single_weights(torch, query)
    elif family == "confounded_family_pair":
        base = params["mixture"].view(u.shape[0], 1, 1, 3).expand(u.shape[0], u.shape[1], query.shape[2], 3)
        contrast = _confounded_contrast(torch, query, params["confound_sign"])
        weights = (base + contrast).clamp_min(1e-4)
        weights = weights / weights.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown controlled stress family: {family}")
    stacked = torch.cat([spectral, local, separable], dim=-1)
    return (stacked * weights).sum(dim=-1, keepdim=True), weights


def _query_piecewise_weights(torch: Any, query: Any):
    q = query[..., 0]
    sharpness = 30.0
    spectral = torch.sigmoid(sharpness * (1.0 / 3.0 - q))
    separable = torch.sigmoid(sharpness * (q - 2.0 / 3.0))
    local = torch.sigmoid(sharpness * (q - 1.0 / 3.0)) * torch.sigmoid(sharpness * (2.0 / 3.0 - q))
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


def _sample_params(torch: Any, batch_size: int, split: str, generator: Any, device: str) -> dict[str, Any]:
    holdout = 0.7 if split == "parameter_holdout" else 0.0
    raw_mix = torch.rand(batch_size, 3, generator=generator, device=device)
    return {
        "gain": 0.8 + holdout + 0.4 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "frequency": 1.0 + holdout + 2.0 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "phase": 2.0 * math.pi * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "lengthscale": 0.08 + 0.25 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "shift": -0.15 + 0.3 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "rank_weight": 0.4 + 0.8 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "mixture": raw_mix / raw_mix.sum(dim=-1, keepdim=True),
        "confound_sign": torch.where(torch.rand(batch_size, generator=generator, device=device) > 0.5, 1.0, -1.0),
    }


def _spectral(torch: Any, u: Any, support_grid: Any, query: Any, params: dict[str, Any]):
    grid = support_grid.view(1, 1, 1, -1, 1)
    q = query.unsqueeze(-2)
    frequency = params["frequency"].view(u.shape[0], 1, 1, 1, 1)
    phase = params["phase"].view(u.shape[0], 1, 1, 1, 1)
    kernel = torch.cos(2.0 * math.pi * frequency * (q - grid) + phase)
    return params["gain"].view(u.shape[0], 1, 1, 1) * (kernel * u.unsqueeze(-3)).mean(dim=-2)


def _local(torch: Any, u: Any, support_grid: Any, query: Any, params: dict[str, Any]):
    grid = support_grid.view(1, 1, 1, -1, 1)
    q = query.unsqueeze(-2)
    lengthscale = params["lengthscale"].view(u.shape[0], 1, 1, 1, 1).abs() + 1e-3
    shift = params["shift"].view(u.shape[0], 1, 1, 1, 1)
    kernel = torch.exp(-((q - grid - shift) ** 2) / (2.0 * lengthscale**2))
    kernel = kernel / kernel.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return params["gain"].view(u.shape[0], 1, 1, 1) * (kernel * u.unsqueeze(-3)).sum(dim=-2)


def _separable(torch: Any, u: Any, support_grid: Any, query: Any, params: dict[str, Any]):
    grid = support_grid.view(1, 1, -1, 1)
    mean = u.mean(dim=-2, keepdim=True)
    moment = (u * grid).mean(dim=-2, keepdim=True)
    sine = (u * torch.sin(math.pi * grid)).mean(dim=-2, keepdim=True)
    return params["gain"].view(u.shape[0], 1, 1, 1) * (
        mean + params["rank_weight"] * moment * query + 0.5 * sine * torch.sin(math.pi * query)
    )


def _generator_device(device: str) -> str:
    device_text = str(device)
    if device_text.startswith("cuda"):
        return device_text
    return "cpu"


def _stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % (2**31)
