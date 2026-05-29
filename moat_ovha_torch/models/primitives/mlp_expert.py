from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.coordinate_features import coordinate_scalar
from moat_ovha_torch.models.primitives.base import PrimitiveParams, apply_film


class MLPExpertPrimitive(nn.Module):
    name = "mlp_expert"

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(
        self,
        target_u: torch.Tensor,
        support_grid: torch.Tensor,
        target_q: torch.Tensor,
        params: PrimitiveParams | None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean = target_u.mean(dim=1, keepdim=True).expand(-1, target_q.shape[1], -1)
        std = target_u.std(dim=1, keepdim=True, unbiased=False).expand(-1, target_q.shape[1], -1)
        energy = (target_u**2).mean(dim=1, keepdim=True).expand(-1, target_q.shape[1], -1)
        features = torch.cat([coordinate_scalar(target_q), mean, std, energy], dim=-1)
        return apply_film(self.net(features), params)
