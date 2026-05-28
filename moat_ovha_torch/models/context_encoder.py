from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.data.episodes import MetaOperatorBatch
from moat_ovha_torch.models.evidence import PrimitiveEvidenceEncoder, primitive_candidate_outputs


class ContextTokenEncoder(nn.Module):
    """Build metadata-free context tokens from observed u, q, y triples."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.evidence_encoder = PrimitiveEvidenceEncoder()
        self.candidate_feature_count = 12
        self.evidence_feature_count = self.evidence_encoder.point_feature_count
        self.input_dim = 6 + 2 * self.candidate_feature_count + self.evidence_feature_count
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
        evidence = self.evidence_encoder(batch)
        candidates = torch.cat(
            [
                evidence.basis_outputs["spectral"],
                evidence.basis_outputs["local"],
                evidence.basis_outputs["separable"],
            ],
            dim=-1,
        )
        residuals = context_y - candidates
        features = torch.cat([summary, context_q, context_y, candidates, residuals, evidence.point_features], dim=-1)
        return self.net(features)


def _primitive_candidate_outputs(context_u: torch.Tensor, support_grid: torch.Tensor, context_q: torch.Tensor) -> torch.Tensor:
    return primitive_candidate_outputs(context_u, support_grid, context_q)
