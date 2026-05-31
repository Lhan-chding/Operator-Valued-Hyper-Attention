from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank
from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES
from moat_ovha_torch.models.multimodal.reliability_prior import ReliabilityPrior


@dataclass(frozen=True)
class MultimodalRouterOutput:
    weights: torch.Tensor
    logits: torch.Tensor
    logit_parts: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor | dict[str, torch.Tensor]]


class MultimodalRelationRouter(nn.Module):
    def __init__(self, d_model: int, candidate_names: tuple[str, ...] = MULTIMODAL_CANDIDATE_NAMES):
        super().__init__()
        self.candidate_names = candidate_names
        self.memory_head = nn.Linear(d_model * 2, 1)
        self.query_candidate_head = nn.Linear(d_model, len(candidate_names))
        nn.init.zeros_(self.query_candidate_head.weight)
        nn.init.zeros_(self.query_candidate_head.bias)

    def forward(
        self,
        memory_bank: dict[str, torch.Tensor],
        evidence: MultimodalEvidenceBank,
        reliability: ReliabilityPrior | None,
    ) -> MultimodalRouterOutput:
        memory_logits = []
        for name in self.candidate_names:
            memory = memory_bank[name].mean(dim=1).unsqueeze(1).expand(-1, evidence.query_features.shape[1], -1)
            memory_logits.append(self.memory_head(torch.cat([evidence.query_features, memory], dim=-1)))
        memory_logit = torch.cat(memory_logits, dim=-1) + self.query_candidate_head(evidence.query_features)
        evidence_logit = evidence.candidate_evidence_logits
        reliability_logit = (
            reliability.operator_logit_bias
            if reliability is not None
            else torch.zeros_like(evidence_logit)
        )
        logits = memory_logit + evidence_logit + reliability_logit
        weights = torch.softmax(logits, dim=-1)
        diagnostics = {
            "router_memory_logit_norm": memory_logit.norm(dim=-1).mean(),
            "router_query_logit_norm": self.query_candidate_head(evidence.query_features).norm(dim=-1).mean(),
            "router_evidence_logit_norm": evidence_logit.norm(dim=-1).mean(),
            "router_reliability_logit_norm": reliability_logit.norm(dim=-1).mean(),
            "router_entropy": -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean(),
            "router_load_by_candidate": {
                name: weights[..., index].mean()
                for index, name in enumerate(self.candidate_names)
            },
        }
        return MultimodalRouterOutput(
            weights=weights,
            logits=logits,
            logit_parts={"memory": memory_logit, "evidence": evidence_logit, "reliability": reliability_logit},
            diagnostics=diagnostics,
        )
