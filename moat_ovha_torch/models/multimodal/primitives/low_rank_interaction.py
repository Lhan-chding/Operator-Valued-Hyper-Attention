from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class LRIOPrimitive(MultimodalCandidatePrimitive):
    name = "LRIO"

    def __init__(self, d_model: int, output_dim: int, rank_count: int = 4):
        super().__init__()
        self.rank_count = rank_count
        self.branch = nn.Linear(d_model, rank_count * d_model)
        self.trunk = nn.Linear(d_model * 2, rank_count * d_model)
        self.pair_gate = nn.Linear(d_model * 2, 1)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        pooled = {
            name: _masked_mean(features, batch.fields[name].mask)
            for name, features in evidence.field_features.items()
        }
        pair_features = []
        pair_gates = []
        temperature = params["interaction_temperature"].clamp_min(1e-4)
        rank_weights = torch.softmax(params["rank_logits"] / temperature, dim=-1)
        for left_index, (left_name, left) in enumerate(pooled.items()):
            for right_name, right in tuple(pooled.items())[left_index + 1:]:
                branch = self.branch(left).view(left.shape[0], 1, self.rank_count, -1)
                right_query = torch.cat([right.unsqueeze(1).expand(-1, evidence.query_features.shape[1], -1), evidence.query_features], dim=-1)
                trunk = self.trunk(right_query).view(right.shape[0], evidence.query_features.shape[1], self.rank_count, -1)
                interaction = (rank_weights.unsqueeze(-1) * (branch * trunk)).sum(dim=-2)
                gate_input = torch.cat([left, right], dim=-1)
                pair_gates.append(self.pair_gate(gate_input).unsqueeze(1))
                pair_features.append(interaction)
        if pair_features:
            pair_stack = torch.stack(pair_features, dim=-2)
            gate = torch.softmax(torch.cat(pair_gates, dim=-1), dim=-1).unsqueeze(-1)
            interaction_feature = (gate * pair_stack).sum(dim=-2)
        else:
            interaction_feature = evidence.low_rank_features
        memory = self.memory_proj(memory_slot.mean(dim=1)).unsqueeze(1)
        feature = self.norm(interaction_feature + memory)
        value = apply_scale_bias(self.head(feature), params)
        rank_logits = params.get("rank_logits")
        diagnostics = {
            "rank_entropy": _entropy(rank_logits) if rank_logits is not None else torch.zeros((), device=value.device),
            "rank_top_k": _top_k(rank_logits, value.device),
            "pair_count": torch.as_tensor(float(len(pair_features)), dtype=value.dtype, device=value.device),
            "pair_interaction_strength": interaction_feature.norm(dim=-1).mean(),
            "interaction_temperature": params.get("interaction_temperature"),
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()


def _top_k(logits: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if logits is None:
        return torch.zeros((), device=device)
    k = min(2, logits.shape[-1])
    return torch.topk(logits, k=k, dim=-1).indices.to(dtype=torch.float32).mean()
