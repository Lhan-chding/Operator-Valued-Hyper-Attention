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
        branches = [target_u.mean(dim=1)]
        if self.rank >= 2:
            branches.append((target_u * grid).mean(dim=1))
        if self.rank >= 3:
            branches.append((target_u * torch.sin(math.pi * grid)).mean(dim=1))
        while len(branches) < self.rank:
            freq = len(branches)
            branches.append((target_u * torch.cos(math.pi * freq * grid)).mean(dim=1))
        branch = torch.stack(branches[: self.rank], dim=-1)
        trunks = [torch.ones_like(target_q)]
        if self.rank >= 2:
            trunks.append(target_q)
        if self.rank >= 3:
            trunks.append(torch.sin(math.pi * target_q))
        while len(trunks) < self.rank:
            freq = len(trunks)
            trunks.append(torch.cos(math.pi * freq * target_q))
        trunk = torch.stack(trunks[: self.rank], dim=-1)
        if params is not None and params.separable_rank_logits is not None:
            weights = torch.softmax(params.separable_rank_logits[..., : self.rank], dim=-1).unsqueeze(-2)
        elif params is not None and params.kernel_params and "rank_logits" in params.kernel_params:
            weights = torch.softmax(params.kernel_params["rank_logits"][..., : self.rank], dim=-1).unsqueeze(-2)
        else:
            weights = torch.softmax(self.coefficients, dim=0)
        value = (branch.unsqueeze(1) * trunk * weights).sum(dim=-1)
        return apply_film(value, params)
