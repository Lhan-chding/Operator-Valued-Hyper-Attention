from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.primitives.base import PrimitiveParams, apply_film, expand_grid


class VectorValuePrimitive(nn.Module):
    name = "vector_value"

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        target_u: torch.Tensor,
        support_grid: torch.Tensor,
        target_q: torch.Tensor,
        params: PrimitiveParams | None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = expand_grid(support_grid, target_u.shape[0])
        temperature = torch.nn.functional.softplus(self.log_temperature) + 1e-3
        logits = -torch.abs(target_q.unsqueeze(-2) - grid.unsqueeze(1)).sum(dim=-1) / temperature
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        value = (weights * target_u.unsqueeze(1)).sum(dim=-2)
        return apply_film(value, params)
