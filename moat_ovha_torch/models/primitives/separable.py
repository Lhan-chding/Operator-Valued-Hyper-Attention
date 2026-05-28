from __future__ import annotations

import math

import torch
from torch import nn

from moat_ovha_torch.models.primitives.base import PrimitiveParams, apply_film, expand_grid


class SeparableBasisPrimitive(nn.Module):
    name = "separable"

    def __init__(self, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.coefficients = nn.Parameter(torch.ones(rank) / rank)

    def forward(
        self,
        target_u: torch.Tensor,
        support_grid: torch.Tensor,
        target_q: torch.Tensor,
        params: PrimitiveParams | None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = expand_grid(support_grid, target_u.shape[0])
        branches = [
            target_u.mean(dim=1),
            (target_u * grid).mean(dim=1),
            (target_u * torch.sin(math.pi * grid)).mean(dim=1),
            (target_u * torch.cos(2.0 * math.pi * grid)).mean(dim=1),
        ]
        next_frequency = 3.0
        while len(branches) < self.rank:
            branches.append((target_u * torch.cos(math.pi * next_frequency * grid)).mean(dim=1))
            next_frequency += 1.0
        branch = torch.stack(branches[: self.rank], dim=-1)
        trunks = [
            torch.ones_like(target_q),
            target_q,
            torch.sin(math.pi * target_q),
            torch.cos(2.0 * math.pi * target_q),
        ]
        next_frequency = 3.0
        while len(trunks) < self.rank:
            trunks.append(torch.cos(math.pi * next_frequency * target_q))
            next_frequency += 1.0
        trunk = torch.stack(trunks[: self.rank], dim=-1)
        if params is not None and params.separable_rank_logits is not None:
            weights = torch.softmax(params.separable_rank_logits[..., : self.rank], dim=-1).unsqueeze(-2)
        elif params is not None and params.kernel_params and "rank_logits" in params.kernel_params:
            weights = torch.softmax(params.kernel_params["rank_logits"][..., : self.rank], dim=-1).unsqueeze(-2)
        else:
            weights = torch.softmax(self.coefficients, dim=0)
        value = (branch.unsqueeze(1) * trunk * weights).sum(dim=-1)
        return apply_film(value, params)
