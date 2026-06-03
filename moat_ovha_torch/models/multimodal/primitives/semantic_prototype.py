from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class SPOPrimitive(MultimodalCandidatePrimitive):
    name = "SPO"

    def __init__(self, d_model: int, output_dim: int, num_prototypes: int = 4):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model) * 0.02)
        self.context_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        context = self.context_proj(evidence.global_features)
        base_logits = torch.matmul(context, self.prototypes.transpose(0, 1)).unsqueeze(1)
        shift = 0.1 * torch.tanh(params["prototype_logits_shift"])
        temperature = params["prototype_temperature"].clamp_min(0.5)
        logits = (base_logits + shift) / temperature
        weights = torch.softmax(logits, dim=-1)
        prototype_mix = torch.matmul(weights, self.prototypes)
        memory = self.memory_proj(memory_slot.mean(dim=1)).unsqueeze(1)
        feature = self.norm(prototype_mix + self.query_proj(evidence.query_features) + memory)
        value = apply_scale_bias(self.head(feature), params)
        diagnostics = {
            "prototype_entropy": _entropy(logits) if logits is not None else torch.zeros((), device=value.device),
            "top_prototype": _top_index(logits, value.device),
            "prototype_temperature": params.get("prototype_temperature"),
            "prototype_usage": weights.mean(dim=(0, 1)),
            "prototype_diversity": _prototype_diversity(self.prototypes),
            "prototype_collapse_warning": weights.mean(dim=(0, 1)).max() > 0.85,
            "prototype_logits_shift_norm": shift.norm(dim=-1).mean(),
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()


def _top_index(logits: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if logits is None:
        return torch.zeros((), device=device)
    return logits.argmax(dim=-1).to(dtype=torch.float32).mean()


def _prototype_diversity(prototypes: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(prototypes, dim=-1)
    gram = normalized @ normalized.transpose(0, 1)
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return (gram - identity).square().mean()
