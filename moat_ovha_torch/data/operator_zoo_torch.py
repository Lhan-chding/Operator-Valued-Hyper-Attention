from __future__ import annotations

import math
import random
from typing import Any, Optional

from moat_ovha_torch.data.episodes import EpisodeHiddenInfo, MetaOperatorBatch
from moat_ovha_torch.runtime import require_torch


class MetadataFreeOperatorZoo:
    """Metadata-free episodic operator generator backed by torch tensors."""

    families = (
        "spectral_family",
        "local_green_family",
        "separable_lowrank_family",
        "nonlinear_family",
        "compositional_mixed_family",
    )

    def __init__(self, seed: int = 0):
        self.seed = seed

    def sample_batch(
        self,
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
        if family not in self.families:
            raise ValueError(f"unknown family: {family}")
        if mode not in {"same_function_field", "operator_transfer"}:
            raise ValueError(f"unknown mode: {mode}")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(self.seed, family, split, mode, batch_size, support_points, query_points))
        support_points = support_points * max(1, resolution_multiplier)
        support_grid = torch.linspace(0.0, 1.0, support_points, device=device).view(1, support_points, 1)
        target_q = torch.linspace(0.0, 1.0, query_points * max(1, resolution_multiplier), device=device)
        target_q = target_q.view(1, -1, 1).repeat(batch_size, 1, 1)
        context_q = _sample_context_q(torch, batch_size, num_demos, context_points, split, generator, device)
        params = _sample_params(torch, batch_size, family, split, generator, device)

        context_u = _sample_functions(torch, batch_size, num_demos, support_grid, generator, device)
        context_y = _apply_operator(torch, family, context_u, support_grid, context_q, params)

        if mode == "same_function_field":
            target_u = context_u[:, 0]
        else:
            target_u = _sample_functions(torch, batch_size, 1, support_grid, generator, device)[:, 0]
        target_y = _apply_operator(torch, family, target_u.unsqueeze(1), support_grid, target_q.unsqueeze(1), params)[:, 0]

        context_mask = torch.ones(batch_size, num_demos, context_points, dtype=torch.bool, device=device)
        target_mask = torch.ones(batch_size, target_q.shape[1], dtype=torch.bool, device=device)
        if split == "confusable_context":
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
            latent_params={key: _to_python(value) for key, value in params.items() if key != "mixture"},
            mixture_weights=_to_python(params.get("mixture")),
            oracle_hints={"family": family},
        )
        return batch, hidden


def _sample_context_q(torch: Any, batch_size: int, num_demos: int, context_points: int, split: str, generator: Any, device: str):
    if split == "confusable_context":
        base = torch.linspace(0.2, 0.4, context_points, device=device)
    else:
        base = torch.linspace(0.0, 1.0, context_points, device=device)
    jitter = 0.01 * torch.randn(batch_size, num_demos, context_points, 1, generator=generator, device=device)
    return (base.view(1, 1, context_points, 1) + jitter).clamp(0.0, 1.0)


def _sample_functions(torch: Any, batch_size: int, count: int, support_grid: Any, generator: Any, device: str):
    grid = support_grid.view(1, 1, -1, 1)
    freq = torch.randint(1, 4, (batch_size, count, 1, 1), generator=generator, device=device).float()
    phase = 2.0 * math.pi * torch.rand(batch_size, count, 1, 1, generator=generator, device=device)
    amp = 0.7 + 0.6 * torch.rand(batch_size, count, 1, 1, generator=generator, device=device)
    return amp * torch.sin(2.0 * math.pi * freq * grid + phase) + 0.35 * torch.cos(math.pi * (freq + 1.0) * grid)


def _sample_params(torch: Any, batch_size: int, family: str, split: str, generator: Any, device: str) -> dict[str, Any]:
    holdout = 0.7 if split == "parameter_holdout" else 0.0
    gain = 0.8 + holdout + 0.4 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device)
    params = {
        "gain": gain,
        "frequency": 1.0 + holdout + 2.0 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "phase": 2.0 * math.pi * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "lengthscale": 0.08 + 0.25 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "shift": -0.15 + 0.3 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
        "rank_weight": 0.4 + 0.8 * torch.rand(batch_size, 1, 1, 1, generator=generator, device=device),
    }
    if family == "compositional_mixed_family":
        mixture = torch.rand(batch_size, 4, generator=generator, device=device)
        params["mixture"] = mixture / mixture.sum(dim=-1, keepdim=True)
    return params


def _apply_operator(torch: Any, family: str, u: Any, support_grid: Any, query: Any, params: dict[str, Any]):
    spectral = _spectral(torch, u, support_grid, query, params)
    local = _local(torch, u, support_grid, query, params)
    separable = _separable(torch, u, support_grid, query, params)
    nonlinear = torch.tanh(local + 0.35 * spectral)
    if family == "spectral_family":
        return spectral
    if family == "local_green_family":
        return local
    if family == "separable_lowrank_family":
        return separable
    if family == "nonlinear_family":
        return nonlinear
    mixture = params["mixture"].view(u.shape[0], 1, 1, 4)
    stacked = torch.cat([spectral, local, separable, nonlinear], dim=-1)
    return (stacked * mixture).sum(dim=-1, keepdim=True)


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
    q = query
    return params["gain"].view(u.shape[0], 1, 1, 1) * (
        mean + params["rank_weight"] * moment * q + 0.5 * sine * torch.sin(math.pi * q)
    )


def _stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % (2**31)


def _to_python(value: Optional[Any]) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().reshape(-1)[:16].tolist()
    return value
