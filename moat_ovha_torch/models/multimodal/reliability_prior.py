from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalModelInputs
from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank
from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES


@dataclass(frozen=True)
class ReliabilityPrior:
    operator_logit_bias: torch.Tensor
    modality_reliability: torch.Tensor
    modality_names: tuple[str, ...]
    pair_reliability: torch.Tensor | None
    pair_names: tuple[str, ...]
    features: torch.Tensor
    diagnostics: dict[str, Any]


class RCEOReliabilityPrior(nn.Module):
    def __init__(
        self,
        d_model: int,
        candidate_names: tuple[str, ...] = MULTIMODAL_CANDIDATE_NAMES,
        lrio_pairs: tuple[tuple[str, str], ...] | None = None,
    ):
        super().__init__()
        self.candidate_names = candidate_names
        self.lrio_pairs = tuple(_normalize_pair(pair) for pair in (lrio_pairs or ()))
        self.bias_head = nn.Linear(d_model + 1, len(candidate_names))
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    def forward(self, batch: MultimodalModelInputs, evidence: MultimodalEvidenceBank) -> ReliabilityPrior:
        reliabilities = []
        modality_names = tuple(batch.fields)
        for field in batch.fields.values():
            if field.quality is None:
                quality = field.mask.to(dtype=evidence.query_features.dtype, device=evidence.query_features.device).mean(dim=1, keepdim=True)
            else:
                quality = field.quality.to(dtype=evidence.query_features.dtype, device=evidence.query_features.device).reshape(field.quality.shape[0], -1).mean(dim=1, keepdim=True)
            reliabilities.append(quality.clamp(0.0, 1.0))
        modality_reliability = torch.cat(reliabilities, dim=-1)
        pair_names, pair_reliability = _pair_reliability(modality_names, modality_reliability, self.lrio_pairs)
        reliability_gap_mean = (1.0 - modality_reliability).clamp_min(0.0).mean(dim=-1, keepdim=True)
        reliability_query = reliability_gap_mean.unsqueeze(1).expand(-1, evidence.query_features.shape[1], -1)
        reliability_only_context = torch.zeros_like(evidence.query_features)
        features = torch.cat([reliability_only_context, reliability_query], dim=-1)
        bias = reliability_query * self.bias_head(features)
        diagnostics = {
            "modality_reliability": modality_reliability.mean(dim=0),
            "modality_reliability_mean": modality_reliability.mean(),
            "sample_modality_reliability_mean": modality_reliability.mean(dim=-1),
            "sample_modality_reliability_gap_mean": reliability_gap_mean.squeeze(-1),
            "modality_names": modality_names,
            "pair_reliability": _diagnostic_pair_map(pair_names, pair_reliability),
            "reliability_bias_norm": bias.norm(dim=-1).mean(),
            "operator_logit_bias_norm": bias.norm(dim=-1).mean(),
            "evidence_conditioned_bias_norm": torch.zeros((), dtype=bias.dtype, device=bias.device),
            "query_semantic_conditioned": False,
            "rceo_is_pure_reliability_prior": True,
            "corruption_response": (1.0 - modality_reliability).clamp_min(0.0).mean(),
        }
        return ReliabilityPrior(
            operator_logit_bias=bias,
            modality_reliability=modality_reliability,
            modality_names=modality_names,
            pair_reliability=pair_reliability,
            pair_names=pair_names,
            features=features,
            diagnostics=diagnostics,
        )


def _normalize_pair(pair: tuple[str, str]) -> tuple[str, str]:
    left, right = str(pair[0]), str(pair[1])
    if left == right:
        raise ValueError("LRIO reliability pairs must contain two distinct modalities")
    return left, right


def _pair_key(pair: tuple[str, str]) -> str:
    left, right = _normalize_pair(pair)
    return f"{left}__{right}"


def _pair_reliability(
    modality_names: tuple[str, ...],
    modality_reliability: torch.Tensor,
    pairs: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, ...], torch.Tensor | None]:
    if not pairs:
        return (), None
    index_by_modality = {name: index for index, name in enumerate(modality_names)}
    values = []
    names = []
    for pair in pairs:
        left, right = _normalize_pair(pair)
        if left not in index_by_modality or right not in index_by_modality:
            continue
        values.append(modality_reliability[:, index_by_modality[left]] * modality_reliability[:, index_by_modality[right]])
        names.append(_pair_key(pair))
    if not values:
        return (), None
    return tuple(names), torch.stack(values, dim=-1)


def _diagnostic_pair_map(pair_names: tuple[str, ...], pair_reliability: torch.Tensor | None) -> dict[str, torch.Tensor]:
    if pair_reliability is None:
        return {}
    return {
        name: pair_reliability[:, index].mean()
        for index, name in enumerate(pair_names)
    }
