from __future__ import annotations

import torch
from torch import nn


class PoolingMemoryEncoder(nn.Module):
    """DeepSets-style context memory, invariant to context token order."""

    def __init__(self, d_model: int = 64, memory_tokens: int = 4):
        super().__init__()
        self.memory_tokens = memory_tokens
        self.project = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, memory_tokens * d_model))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        flat = tokens.flatten(1, 2)
        if mask is not None:
            weights = mask.flatten(1).float().unsqueeze(-1)
            pooled = (flat * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            pooled = flat.mean(dim=1)
        memory = self.project(pooled)
        return memory.view(tokens.shape[0], self.memory_tokens, tokens.shape[-1])


class PerceiverMemoryEncoder(nn.Module):
    """Small latent cross-attention memory encoder for the main torch path."""

    def __init__(self, d_model: int = 64, memory_tokens: int = 4, num_heads: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(memory_tokens, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        flat = tokens.flatten(1, 2)
        latents = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.flatten(1).bool()
        attended, _ = self.attn(latents, flat, flat, key_padding_mask=key_padding_mask)
        return attended + self.ff(attended)
