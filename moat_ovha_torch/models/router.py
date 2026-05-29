from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from moat_ovha_torch.models.memory import primitive_memory


@dataclass(frozen=True)
class RouterOutput:
    weights: torch.Tensor
    logits: torch.Tensor
    context_prior_logits: torch.Tensor | None = None
    query_residual_logits: torch.Tensor | None = None


class PrimitiveRouter(nn.Module):
    def __init__(
        self,
        primitive_names: tuple[str, ...],
        d_model: int = 64,
        top_k: int | None = None,
        query_conditioned: bool = True,
        random_router: bool = False,
    ):
        super().__init__()
        self.primitive_names = primitive_names
        self.top_k = top_k
        self.query_conditioned = query_conditioned
        self.random_router = random_router
        self.context_prior = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.query_residual = nn.Sequential(nn.Linear(d_model + 1, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, memory: torch.Tensor | dict[str, torch.Tensor], target_q: torch.Tensor) -> RouterOutput:
        if self.random_router:
            logits = torch.randn(target_q.shape[0], target_q.shape[1], len(self.primitive_names), device=target_q.device)
            weights = _normalize_logits(logits, self.top_k)
            return RouterOutput(weights=weights, logits=logits, context_prior_logits=None, query_residual_logits=None)

        primitive_features = torch.stack(
            [primitive_memory(memory, name).mean(dim=1) for name in self.primitive_names],
            dim=1,
        )
        context_prior_logits = self.context_prior(primitive_features).squeeze(-1)
        if self.query_conditioned:
            repeated_features = primitive_features[:, None, :, :].expand(-1, target_q.shape[1], -1, -1)
            repeated_q = target_q[:, :, None, :].expand(-1, -1, len(self.primitive_names), -1)
            query_features = torch.cat([repeated_features, repeated_q], dim=-1)
            raw_query_residual_logits = self.query_residual(query_features).squeeze(-1)
            gate = _context_uncertainty_gate(context_prior_logits).view(target_q.shape[0], 1, 1)
            query_residual_logits = raw_query_residual_logits * gate
        else:
            query_residual_logits = torch.zeros(
                target_q.shape[0],
                target_q.shape[1],
                len(self.primitive_names),
                dtype=target_q.dtype,
                device=target_q.device,
            )
        logits = context_prior_logits.unsqueeze(1) + query_residual_logits
        weights = _normalize_logits(logits, self.top_k)
        return RouterOutput(
            weights=weights,
            logits=logits,
            context_prior_logits=context_prior_logits,
            query_residual_logits=query_residual_logits,
        )


def primitive_entropy(weights: torch.Tensor) -> torch.Tensor:
    return -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()


def _context_uncertainty_gate(context_prior_logits: torch.Tensor) -> torch.Tensor:
    if context_prior_logits.shape[-1] <= 1:
        return torch.zeros(context_prior_logits.shape[0], dtype=context_prior_logits.dtype, device=context_prior_logits.device)
    prior = torch.softmax(context_prior_logits, dim=-1)
    entropy = -(prior * prior.clamp_min(1e-12).log()).sum(dim=-1)
    max_entropy = math.log(context_prior_logits.shape[-1])
    return (entropy / max(max_entropy, 1e-6)).clamp(min=0.0, max=1.0)


def _normalize_logits(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    weights = torch.softmax(logits, dim=-1)
    if top_k is not None and top_k < weights.shape[-1]:
        values, indices = torch.topk(weights, top_k, dim=-1)
        masked = torch.zeros_like(weights).scatter_(-1, indices, values)
        weights = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return weights
