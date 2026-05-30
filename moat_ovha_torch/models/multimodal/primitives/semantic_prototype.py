from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class SPOPrimitive(MultimodalCandidatePrimitive):
    name = "SPO"

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        feature = evidence.prototype_features + memory_slot.mean(dim=1).unsqueeze(1)
        value = apply_scale_bias(self.head(feature), params)
        logits = params.get("prototype_logits_shift")
        diagnostics = {
            "prototype_entropy": _entropy(logits) if logits is not None else torch.zeros((), device=value.device),
            "prototype_temperature": params.get("prototype_temperature"),
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
