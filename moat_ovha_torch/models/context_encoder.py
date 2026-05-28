from __future__ import annotations

import math

import torch
from torch import nn

from moat_ovha_torch.data.episodes import MetaOperatorBatch


class ContextTokenEncoder(nn.Module):
    """Build metadata-free context tokens from observed u, q, y triples."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.candidate_feature_count = 12
        self.input_dim = 6 + 2 * self.candidate_feature_count
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
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
        candidates = _primitive_candidate_outputs(context_u, support_grid, context_q)
        residuals = context_y - candidates
        features = torch.cat([summary, context_q, context_y, candidates, residuals], dim=-1)
        return self.net(features)


def _primitive_candidate_outputs(context_u: torch.Tensor, support_grid: torch.Tensor, context_q: torch.Tensor) -> torch.Tensor:
    if support_grid.shape[0] == 1:
        support_grid = support_grid.expand(context_u.shape[0], -1, -1)
    u = context_u.unsqueeze(2)
    q = context_q.unsqueeze(3)
    grid = support_grid[:, None, None, :, :]

    spectral = []
    for mode in range(1, 5):
        kernel = torch.cos(2.0 * math.pi * mode * (q - grid))
        spectral.append((kernel * u).mean(dim=-2))

    local = []
    for lengthscale in (0.08, 0.12, 0.16, 0.20):
        kernel = torch.exp(-((q - grid) ** 2) / (2.0 * lengthscale**2))
        kernel = kernel / kernel.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        local.append((kernel * u).sum(dim=-2))

    branch_grid = support_grid[:, None, :, :]
    separable_branches = [
        context_u.mean(dim=2, keepdim=True),
        (context_u * branch_grid).mean(dim=2, keepdim=True),
        (context_u * torch.sin(math.pi * branch_grid)).mean(dim=2, keepdim=True),
        (context_u * torch.cos(2.0 * math.pi * branch_grid)).mean(dim=2, keepdim=True),
    ]
    separable_trunks = [
        torch.ones_like(context_q),
        context_q,
        torch.sin(math.pi * context_q),
        torch.cos(2.0 * math.pi * context_q),
    ]
    separable = [branch.expand(-1, -1, context_q.shape[2], -1) * trunk for branch, trunk in zip(separable_branches, separable_trunks)]

    return torch.cat(spectral + local + separable, dim=-1)
