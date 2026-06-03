from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch
from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES


@dataclass(frozen=True)
class PairEvidenceBank:
    pair_names: tuple[str, ...]
    pair_features: dict[str, torch.Tensor]
    pair_gram: dict[str, torch.Tensor]
    pair_correlation: dict[str, torch.Tensor]


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
    pair_features: dict[str, torch.Tensor] | None = None
    pair_evidence: PairEvidenceBank | None = None


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
        paired_interaction, pair_evidence = _paired_field_interaction_bank(pooled, query_features)
        pair_features = {
            name: self.low_rank_head(torch.cat([query_features, feature], dim=-1))
            for name, feature in pair_evidence.pair_features.items()
        }
        low_rank_features = self.low_rank_head(torch.cat([query_features, paired_interaction], dim=-1))
        alignment_features, alignment_entropy = _alignment_features(query_features, field_features)
        local_features = self.local_head(torch.cat([local_features, repeated_global], dim=-1))
        alignment_features = self.alignment_head(torch.cat([alignment_features, repeated_global], dim=-1))
        explicit_relation_logits, explicit_relation_rate = _explicit_query_type_relation_logits(
            batch.query.query_type,
            query_features.shape[:2],
            len(MULTIMODAL_CANDIDATE_NAMES),
            dtype=query_features.dtype,
            device=query_features.device,
        )
        candidate_evidence_logits = self.evidence_logit_head(fused) + explicit_relation_logits
        diagnostics = {
            "local_entropy": local_entropy,
            "alignment_entropy": alignment_entropy,
            "field_count": torch.tensor(float(len(field_features)), dtype=query_features.dtype, device=query_features.device),
            "explicit_query_relation_prior_rate": explicit_relation_rate,
            "controlled_family_relation_prior_rate": torch.zeros((), dtype=query_features.dtype, device=query_features.device),
            "pair_feature_count": torch.tensor(float(len(pair_features)), dtype=query_features.dtype, device=query_features.device),
            "pair_feature_norm": (
                torch.stack([feature.norm(dim=-1).mean() for feature in pair_features.values()]).mean()
                if pair_features
                else torch.zeros((), dtype=query_features.dtype, device=query_features.device)
            ),
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
            pair_features=pair_features,
            pair_evidence=PairEvidenceBank(
                pair_names=pair_evidence.pair_names,
                pair_features=pair_features,
                pair_gram=pair_evidence.pair_gram,
                pair_correlation=pair_evidence.pair_correlation,
            ),
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


def _paired_field_interaction_bank(
    pooled: dict[str, torch.Tensor],
    query_features: torch.Tensor,
) -> tuple[torch.Tensor, PairEvidenceBank]:
    names = list(pooled)
    if len(names) == 1:
        interaction = pooled[names[0]].unsqueeze(1).expand(-1, query_features.shape[1], -1)
        return interaction, PairEvidenceBank(pair_names=(), pair_features={}, pair_gram={}, pair_correlation={})
    pair_features = {}
    pair_gram = {}
    pair_correlation = {}
    for left_index, left_name in enumerate(names):
        left = pooled[left_name]
        for right_name in names[left_index + 1:]:
            right = pooled[right_name]
            key = f"{left_name}__{right_name}"
            pair = left * right
            pair_features[key] = pair.unsqueeze(1).expand(-1, query_features.shape[1], -1)
            pair_gram[key] = (left * right).sum(dim=-1)
            pair_correlation[key] = torch.nn.functional.cosine_similarity(left, right, dim=-1)
    interaction = torch.stack(list(pair_features.values()), dim=0).mean(dim=0)
    return interaction, PairEvidenceBank(
        pair_names=tuple(pair_features),
        pair_features=pair_features,
        pair_gram=pair_gram,
        pair_correlation=pair_correlation,
    )


def _explicit_query_type_relation_logits(
    query_type: torch.Tensor,
    batch_query_shape: torch.Size,
    candidate_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros(*batch_query_shape, candidate_count, dtype=dtype, device=device)
    shape = getattr(query_type, "shape", None)
    if shape is None or len(shape) not in (2, 3):
        return zeros, torch.zeros((), dtype=dtype, device=device)
    values = query_type.to(dtype=dtype, device=device)
    if values.ndim == 3 and int(values.shape[-1]) == candidate_count:
        # Query type may encode public task/query family, but never controlled hidden active operator.
        # Keep the prior deliberately weak so an untrained router cannot become one-hot.
        return 0.25 * values, torch.ones((), dtype=dtype, device=device)
    return zeros, torch.zeros((), dtype=dtype, device=device)
