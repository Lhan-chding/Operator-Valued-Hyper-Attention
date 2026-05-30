from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class CATOPrimitive(MultimodalCandidatePrimitive):
    name = "CATO"

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        feature = evidence.alignment_features + memory_slot.mean(dim=1).unsqueeze(1)
        value = apply_scale_bias(self.head(feature), params)
        diagnostics = {
            "alignment_entropy": evidence.alignment_entropy,
            "alignment_temperature": params.get("alignment_temperature"),
            "transport_scale": params.get("transport_scale"),
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)
