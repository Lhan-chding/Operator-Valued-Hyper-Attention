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
            "top_k_alignment": _top_k_alignment(feature),
            "transport_marginal_error": _transport_marginal_error(params.get("transport_scale"), value.device),
            "alignment_temperature": params.get("alignment_temperature"),
            "transport_scale": params.get("transport_scale"),
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _top_k_alignment(feature: torch.Tensor) -> torch.Tensor:
    k = min(2, feature.shape[1])
    scores = feature.norm(dim=-1)
    return torch.topk(scores, k=k, dim=-1).indices.to(dtype=torch.float32).mean()


def _transport_marginal_error(transport_scale: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if transport_scale is None:
        return torch.zeros((), device=device)
    return (1.0 - transport_scale).abs().mean()
