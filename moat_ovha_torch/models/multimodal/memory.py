from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES


class MultimodalOperatorMemory(nn.Module):
    def __init__(self, d_model: int, memory_tokens: int = 4, candidate_names: tuple[str, ...] = MULTIMODAL_CANDIDATE_NAMES):
        super().__init__()
        self.candidate_names = candidate_names
        self.memory_tokens = memory_tokens
        self.project = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, memory_tokens * d_model))
        self.slot_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(memory_tokens, d_model) * 0.02) for name in candidate_names}
        )

    def forward(self, global_features: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.project(global_features).view(global_features.shape[0], self.memory_tokens, global_features.shape[-1])
        return {name: base + self.slot_embeddings[name].unsqueeze(0) for name in self.candidate_names}
