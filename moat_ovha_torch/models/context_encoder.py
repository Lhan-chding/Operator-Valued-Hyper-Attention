from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.data.episodes import MetaOperatorBatch


class ContextTokenEncoder(nn.Module):
    """Build metadata-free context tokens from observed u, q, y triples."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, batch: MetaOperatorBatch) -> torch.Tensor:
        context_u = batch.context_u
        context_q = batch.context_q
        context_y = batch.context_y
        support_grid = batch.support_grid
        if support_grid.shape[0] == 1:
            support_grid = support_grid.expand(context_u.shape[0], -1, -1)
        grid = support_grid[:, None, :, :]
        u_mean = context_u.mean(dim=2, keepdim=True)
        u_std = context_u.std(dim=2, keepdim=True, unbiased=False)
        u_energy = (context_u**2).mean(dim=2, keepdim=True)
        u_grid_moment = (context_u * grid).mean(dim=2, keepdim=True)
        summary = torch.cat([u_mean, u_std, u_energy, u_grid_moment], dim=-1)
        summary = summary.expand(-1, -1, context_q.shape[2], -1)
        features = torch.cat([summary, context_q, context_y], dim=-1)
        return self.net(features)
