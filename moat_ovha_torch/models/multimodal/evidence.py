from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch
from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES


@dataclass(frozen=True)
class MultimodalEvidenceBank:
    query_features: torch.Tensor
    global_features: torch.Tensor
    local_features: torch.Tensor
    prototype_features: torch.Tensor
    low_rank_features: torch.Tensor
    alignment_features: torch.Tensor
    candidate_evidence_logits: torch.Tensor
    local_entropy: torch.Tensor
    alignment_entropy: torch.Tensor
    field_features: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


class MultimodalEvidenceEncoder(nn.Module):
    def __init__(self, field_dims: dict[str, int], query_dim: int, d_model: int):
        super().__init__()
        self.field_projections = nn.ModuleDict({name: nn.Linear(dim, d_model) for name, dim in field_dims.items()})
        self.query_projection = nn.Linear(query_dim, d_model)
        self.local_head = nn.Linear(d_model * 2, d_model)
        self.prototype_head = nn.Linear(d_model * 2, d_model)
        self.low_rank_head = nn.Linear(d_model * 2, d_model)
        self.alignment_head = nn.Linear(d_model * 2, d_model)
        self.evidence_logit_head = nn.Linear(d_model * 2, len(MULTIMODAL_CANDIDATE_NAMES))

    def forward(self, batch: MultimodalEpisodeBatch) -> MultimodalEvidenceBank:
        query_features = self.query_projection(batch.query.x)
        field_features = {
            name: self.field_projections[name](field.x)
            for name, field in batch.fields.items()
            if name in self.field_projections
        }
        if not field_features:
            raise ValueError("MultimodalEvidenceEncoder requires at least one configured field")
        pooled = {name: _masked_mean(features, batch.fields[name].mask) for name, features in field_features.items()}
        global_features = torch.stack(list(pooled.values()), dim=0).mean(dim=0)
        repeated_global = global_features.unsqueeze(1).expand(-1, query_features.shape[1], -1)
        fused = torch.cat([query_features, repeated_global], dim=-1)

        local_source = field_features["text"] if "text" in field_features else next(iter(field_features.values()))
        local_features, local_entropy = _nearest_token_features(query_features, local_source)
        prototype_features = self.prototype_head(fused)
        low_rank_features = self.low_rank_head(torch.cat([query_features, _paired_field_interaction(pooled, query_features)], dim=-1))
        alignment_features, alignment_entropy = _alignment_features(query_features, field_features)
        local_features = self.local_head(torch.cat([local_features, repeated_global], dim=-1))
        alignment_features = self.alignment_head(torch.cat([alignment_features, repeated_global], dim=-1))
        candidate_evidence_logits = self.evidence_logit_head(fused)
        diagnostics = {
            "local_entropy": local_entropy,
            "alignment_entropy": alignment_entropy,
            "field_count": torch.tensor(float(len(field_features)), dtype=query_features.dtype, device=query_features.device),
        }
        return MultimodalEvidenceBank(
            query_features=query_features,
            global_features=global_features,
            local_features=local_features,
            prototype_features=prototype_features,
            low_rank_features=low_rank_features,
            alignment_features=alignment_features,
            candidate_evidence_logits=candidate_evidence_logits,
            local_entropy=local_entropy,
            alignment_entropy=alignment_entropy,
            field_features=field_features,
            diagnostics=diagnostics,
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _nearest_token_features(query: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(query, tokens.transpose(1, 2)) / max(query.shape[-1] ** 0.5, 1.0)
    weights = torch.softmax(scores, dim=-1)
    features = torch.matmul(weights, tokens)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return features, entropy


def _alignment_features(query: torch.Tensor, field_features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if "region" in field_features:
        source = field_features["region"]
    elif "vision" in field_features:
        source = field_features["vision"]
    else:
        source = next(iter(field_features.values()))
    scores = torch.matmul(query, source.transpose(1, 2)) / max(query.shape[-1] ** 0.5, 1.0)
    weights = torch.softmax(scores, dim=-1)
    features = torch.matmul(weights, source)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return features, entropy


def _paired_field_interaction(pooled: dict[str, torch.Tensor], query_features: torch.Tensor) -> torch.Tensor:
    values = list(pooled.values())
    if len(values) == 1:
        interaction = values[0]
    else:
        interaction = values[0] * values[1]
    return interaction.unsqueeze(1).expand(-1, query_features.shape[1], -1)
