from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROLE_NAMES = ("entity", "attribute", "context", "relation")


@dataclass(frozen=True)
class RoleState:
    vectors: Tensor
    attention: Tensor

    def __post_init__(self) -> None:
        if self.vectors.ndim != 3 or self.attention.ndim != 3:
            raise ValueError("role vectors and attention must both be rank-3")
        if self.vectors.shape[:2] != self.attention.shape[:2]:
            raise ValueError("role vector and attention batch/role axes differ")


class LatentRoleEncoder(nn.Module):
    """Four learned latent slots over valid language tokens."""

    def __init__(self, d_model: int = 256, num_heads: int = 8,
                 role_count: int = 4) -> None:
        super().__init__()
        if role_count != len(ROLE_NAMES):
            raise ValueError(f"Phase 1 requires exactly {len(ROLE_NAMES)} roles")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.role_queries = nn.Parameter(torch.empty(role_count, d_model))
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.role_queries, std=0.02)

    def forward(self, text: Tensor, text_valid_mask: Tensor) -> RoleState:
        if text.ndim != 3 or text_valid_mask.shape != text.shape[:2]:
            raise ValueError("expected text [B,T,D] and mask [B,T]")
        valid = text_valid_mask.to(dtype=torch.bool)
        if (~valid.any(dim=1)).any():
            raise ValueError("each sample must contain at least one valid text token")
        queries = self.role_queries.unsqueeze(0).expand(text.shape[0], -1, -1)
        vectors, attention = self.cross_attention(
            queries, text, text, key_padding_mask=~valid,
            need_weights=True, average_attn_weights=True)
        attention = attention.masked_fill(~valid[:, None, :], 0.0)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        return RoleState(self.output_norm(vectors), attention)


def role_diversity_loss(attention: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean pairwise cosine similarity between distinct role attentions."""
    if attention.ndim != 3 or attention.shape[1] < 2:
        raise ValueError("attention must have shape [B,R,T] with R >= 2")
    normalized = F.normalize(attention, dim=-1, eps=eps)
    similarity = normalized @ normalized.transpose(-1, -2)
    roles = attention.shape[1]
    off_diagonal = ~torch.eye(roles, dtype=torch.bool, device=attention.device)
    return similarity[:, off_diagonal].mean()
