from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import SeedResult, memory_valid_mask


class GenericDenseSeedPredictor(nn.Module):
    """Untyped capacity control sharing the RQGO seed-loss contract."""

    def __init__(self, d_model: int = 256, num_levels: int = 4,
                 hidden_dim: int | None = None, seed_bias_cap: float = 2.0) -> None:
        super().__init__()
        hidden = hidden_dim or d_model
        self.seed_bias_cap = float(seed_bias_cap)
        self.level_embedding = nn.Embedding(num_levels, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3 + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, memory: Tensor, boxes: Tensor, pooled_text: Tensor,
                level_ids: Tensor, memory_mask: Tensor | None = None) -> SeedResult:
        if memory.ndim != 3 or boxes.shape != (*memory.shape[:2], 4):
            raise ValueError("expected memory [B,N,D] and boxes [B,N,4]")
        if pooled_text.shape != (memory.shape[0], memory.shape[2]):
            raise ValueError("pooled_text must have shape [B,D]")
        if level_ids.shape != (memory.shape[1],):
            raise ValueError("level_ids must have shape [N]")
        valid = memory_valid_mask(memory, memory_mask)
        text = pooled_text[:, None].expand(-1, memory.shape[1], -1)
        levels = self.level_embedding(level_ids)[None].expand(memory.shape[0], -1, -1)
        raw = self.mlp(torch.cat((memory, text, boxes, levels), dim=-1)).squeeze(-1)
        bias = self.seed_bias_cap * raw.tanh()
        raw = raw.masked_fill(~valid, 0.0)
        bias = bias.masked_fill(~valid, 0.0)
        return SeedResult(bias, raw, valid, {"raw_abs_mean": raw.abs().mean()})


def matched_generic_hidden_dim(d_model: int, num_levels: int,
                               relation_count: int = 8,
                               scale_count: int = 3) -> int:
    """Width matching GenericDenseSeed to LatentRoleEncoder + RQGO.

    The closed-form parameter counts avoid instantiating a throw-away typed
    branch and keep the control within 1% when the relation bank changes size.
    """
    if min(d_model, num_levels, relation_count, scale_count) <= 0:
        raise ValueError("capacity dimensions must be positive")
    relation_outputs = relation_count * (1 + scale_count)
    role_parameters = 4 * d_model * d_model + 10 * d_model
    rqgo_parameters = (
        7 * d_model * d_model
        + (11 + relation_outputs) * d_model
        + 4 + relation_outputs
    )
    target = role_parameters + rqgo_parameters
    fixed = num_levels * d_model + 1
    per_hidden = 3 * d_model + 6
    return max(int(round((target - fixed) / per_hidden)), 1)
