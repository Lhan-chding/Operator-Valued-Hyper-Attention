from __future__ import annotations

import torch
from torch import nn


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
        input_dim = d_model + (1 if query_conditioned else 0)
        self.net = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Linear(d_model, len(primitive_names)))

    def forward(self, memory: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        if self.random_router:
            logits = torch.randn(target_q.shape[0], target_q.shape[1], len(self.primitive_names), device=target_q.device)
        else:
            pooled = memory.mean(dim=1)
            if self.query_conditioned:
                features = torch.cat([pooled.unsqueeze(1).expand(-1, target_q.shape[1], -1), target_q], dim=-1)
            else:
                features = pooled.unsqueeze(1).expand(-1, target_q.shape[1], -1)
            logits = self.net(features)
        weights = torch.softmax(logits, dim=-1)
        if self.top_k is not None and self.top_k < len(self.primitive_names):
            values, indices = torch.topk(weights, self.top_k, dim=-1)
            masked = torch.zeros_like(weights).scatter_(-1, indices, values)
            weights = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights


def primitive_entropy(weights: torch.Tensor) -> torch.Tensor:
    return -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
