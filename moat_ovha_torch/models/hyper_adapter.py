from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.primitives.base import PrimitiveParams


class HyperAdapter(nn.Module):
    """Primitive-specific FiLM and kernel parameters from memory + query."""

    def __init__(self, primitive_names: tuple[str, ...], d_model: int = 64, query_conditioned: bool = True):
        super().__init__()
        self.primitive_names = primitive_names
        self.query_conditioned = query_conditioned
        input_dim = d_model + (1 if query_conditioned else 0)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Linear(d_model, 3))
                for name in primitive_names
            }
        )

    def forward(self, memory: torch.Tensor, target_q: torch.Tensor) -> dict[str, PrimitiveParams]:
        pooled = memory.mean(dim=1)
        if self.query_conditioned:
            features = torch.cat([pooled.unsqueeze(1).expand(-1, target_q.shape[1], -1), target_q], dim=-1)
        else:
            features = pooled.unsqueeze(1).expand(-1, target_q.shape[1], -1)
        params = {}
        for name, head in self.heads.items():
            raw = head(features)
            scale = 1.0 + 0.1 * torch.tanh(raw[..., 0:1])
            bias = 0.1 * torch.tanh(raw[..., 1:2])
            kernel = {"lengthscale": raw[..., 2:3]}
            params[name] = PrimitiveParams(scale=scale, bias=bias, kernel_params=kernel)
        return params
