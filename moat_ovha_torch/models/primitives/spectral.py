from __future__ import annotations

import math

import torch
from torch import nn

from moat_ovha_torch.models.primitives.base import PrimitiveParams, apply_film, expand_grid


class SpectralIntegralPrimitive(nn.Module):
    name = "spectral"

    def __init__(self, modes: int = 4):
        super().__init__()
        self.modes = modes
        self.mode_weights = nn.Parameter(torch.ones(modes) / modes)

    def forward(
        self,
        target_u: torch.Tensor,
        support_grid: torch.Tensor,
        target_q: torch.Tensor,
        params: PrimitiveParams | None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = expand_grid(support_grid, target_u.shape[0])
        q = target_q.unsqueeze(-2)
        s = grid.unsqueeze(1)
        outputs = []
        for index in range(1, self.modes + 1):
            kernel = torch.cos(2.0 * math.pi * index * (q - s))
            outputs.append((kernel * target_u.unsqueeze(1)).mean(dim=-2))
        stacked = torch.stack(outputs, dim=-1)
        value = (stacked * torch.softmax(self.mode_weights, dim=0)).sum(dim=-1)
        return apply_film(value, params)
