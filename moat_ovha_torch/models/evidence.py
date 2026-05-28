from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from moat_ovha_torch.data.episodes import MetaOperatorBatch


PRIMITIVE_EVIDENCE_ORDER = ("spectral", "local", "separable")


@dataclass(frozen=True)
class EvidenceBank:
    basis_outputs: dict[str, torch.Tensor]
    residuals: dict[str, torch.Tensor]
    gram: dict[str, torch.Tensor]
    corr: dict[str, torch.Tensor]
    ls_coeff: dict[str, torch.Tensor]
    residual_energy: dict[str, torch.Tensor]
    uncertainty: dict[str, torch.Tensor]
    point_features: torch.Tensor


class PrimitiveEvidenceEncoder(nn.Module):
    """Compute primitive-aligned sufficient statistics from public context triples."""

    def __init__(self, ridge: float = 1e-3):
        super().__init__()
        self.ridge = ridge
        self.candidate_count_by_primitive = {"spectral": 4, "local": 4, "separable": 4}
        self.point_feature_count = 36

    def forward(self, batch: MetaOperatorBatch) -> EvidenceBank:
        candidates = primitive_candidate_outputs(batch.context_u, batch.support_grid, batch.context_q)
        groups = {
            "spectral": candidates[..., 0:4],
            "local": candidates[..., 4:8],
            "separable": candidates[..., 8:12],
        }
        mask = batch.context_mask
        if mask is None:
            mask = torch.ones(
                batch.context_y.shape[:-1],
                dtype=torch.bool,
                device=batch.context_y.device,
            )
        weights = mask.to(dtype=batch.context_y.dtype).flatten(1).unsqueeze(-1)
        target = batch.context_y.flatten(1, 2)

        residuals: dict[str, torch.Tensor] = {}
        gram: dict[str, torch.Tensor] = {}
        corr: dict[str, torch.Tensor] = {}
        ls_coeff: dict[str, torch.Tensor] = {}
        residual_energy: dict[str, torch.Tensor] = {}
        uncertainty: dict[str, torch.Tensor] = {}
        point_feature_parts = []
        for name in PRIMITIVE_EVIDENCE_ORDER:
            basis = groups[name]
            residual = batch.context_y - basis
            residuals[name] = residual
            flat_basis = basis.flatten(1, 2)
            weighted_basis = flat_basis * weights
            normalizer = weights.sum(dim=1).clamp_min(1.0)
            group_gram = torch.matmul(flat_basis.transpose(1, 2), weighted_basis) / normalizer.unsqueeze(-1)
            group_corr = torch.matmul(flat_basis.transpose(1, 2), target * weights).squeeze(-1) / normalizer
            eye = torch.eye(group_gram.shape[-1], dtype=group_gram.dtype, device=group_gram.device).unsqueeze(0)
            coeff = torch.linalg.solve(group_gram + self.ridge * eye, group_corr.unsqueeze(-1)).squeeze(-1)
            flat_residual = residual.flatten(1, 2)
            energy = (flat_residual.square() * weights).sum(dim=1) / normalizer
            centered = flat_residual - (flat_residual * weights).sum(dim=1, keepdim=True) / normalizer.view(-1, 1, 1)
            spread = (centered.square() * weights).sum(dim=1) / normalizer

            gram[name] = group_gram
            corr[name] = group_corr
            ls_coeff[name] = coeff
            residual_energy[name] = energy
            uncertainty[name] = spread.sqrt()
            point_feature_parts.extend([coeff, energy, uncertainty[name]])

        global_stats = torch.cat(point_feature_parts, dim=-1)
        point_features = global_stats[:, None, None, :].expand(
            -1,
            batch.context_q.shape[1],
            batch.context_q.shape[2],
            -1,
        )
        return EvidenceBank(
            basis_outputs=groups,
            residuals=residuals,
            gram=gram,
            corr=corr,
            ls_coeff=ls_coeff,
            residual_energy=residual_energy,
            uncertainty=uncertainty,
            point_features=point_features,
        )


def primitive_candidate_outputs(context_u: torch.Tensor, support_grid: torch.Tensor, context_q: torch.Tensor) -> torch.Tensor:
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
