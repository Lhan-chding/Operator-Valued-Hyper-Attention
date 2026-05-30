from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch
from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank
from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES


@dataclass(frozen=True)
class ReliabilityPrior:
    operator_logit_bias: torch.Tensor
    modality_reliability: torch.Tensor
    features: torch.Tensor
    diagnostics: dict[str, Any]


class RCEOReliabilityPrior(nn.Module):
    def __init__(self, d_model: int, candidate_names: tuple[str, ...] = MULTIMODAL_CANDIDATE_NAMES):
        super().__init__()
        self.candidate_names = candidate_names
        self.bias_head = nn.Linear(d_model + 1, len(candidate_names))

    def forward(self, batch: MultimodalEpisodeBatch, evidence: MultimodalEvidenceBank) -> ReliabilityPrior:
        reliabilities = []
        for field in batch.fields.values():
            if field.quality is None:
                quality = field.mask.to(dtype=evidence.query_features.dtype, device=evidence.query_features.device).mean(dim=1, keepdim=True)
            else:
                quality = field.quality.to(dtype=evidence.query_features.dtype, device=evidence.query_features.device).reshape(field.quality.shape[0], -1).mean(dim=1, keepdim=True)
            reliabilities.append(quality.clamp(0.0, 1.0))
        modality_reliability = torch.cat(reliabilities, dim=-1)
        reliability_mean = modality_reliability.mean(dim=-1, keepdim=True)
        reliability_query = reliability_mean.unsqueeze(1).expand(-1, evidence.query_features.shape[1], -1)
        features = torch.cat([evidence.query_features, reliability_query], dim=-1)
        bias = self.bias_head(features)
        diagnostics = {
            "modality_reliability_mean": modality_reliability.mean(),
            "operator_logit_bias_norm": bias.norm(dim=-1).mean(),
        }
        return ReliabilityPrior(
            operator_logit_bias=bias,
            modality_reliability=modality_reliability,
            features=features,
            diagnostics=diagnostics,
        )
