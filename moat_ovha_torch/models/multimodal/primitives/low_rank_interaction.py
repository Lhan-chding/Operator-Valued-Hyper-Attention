from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class LRIOPrimitive(MultimodalCandidatePrimitive):
    name = "LRIO"

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        feature = evidence.low_rank_features + memory_slot.mean(dim=1).unsqueeze(1)
        value = apply_scale_bias(self.head(feature), params)
        rank_logits = params.get("rank_logits")
        diagnostics = {
            "rank_entropy": _entropy(rank_logits) if rank_logits is not None else torch.zeros((), device=value.device),
            "rank_top_k": _top_k(rank_logits, value.device),
            "pair_interaction_strength": evidence.low_rank_features.norm(dim=-1).mean(),
            "interaction_temperature": params.get("interaction_temperature"),
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()


def _top_k(logits: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if logits is None:
        return torch.zeros((), device=device)
    k = min(2, logits.shape[-1])
    return torch.topk(logits, k=k, dim=-1).indices.to(dtype=torch.float32).mean()
