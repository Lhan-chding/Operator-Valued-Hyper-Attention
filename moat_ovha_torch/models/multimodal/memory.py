from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank
from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES


class MultimodalOperatorMemory(nn.Module):
    def __init__(self, d_model: int, memory_tokens: int = 4, candidate_names: tuple[str, ...] = MULTIMODAL_CANDIDATE_NAMES):
        super().__init__()
        self.candidate_names = candidate_names
        self.memory_tokens = memory_tokens
        self.project = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, memory_tokens * d_model))
        self.candidate_projects = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, memory_tokens * d_model),
                )
                for name in candidate_names
            }
        )
        self.slot_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(memory_tokens, d_model) * 0.02) for name in candidate_names}
        )

    def forward(
        self,
        global_features: torch.Tensor,
        evidence: MultimodalEvidenceBank | None = None,
    ) -> dict[str, torch.Tensor]:
        base = self.project(global_features).view(global_features.shape[0], self.memory_tokens, global_features.shape[-1])
        memory_bank = {}
        for name in self.candidate_names:
            if evidence is None:
                candidate_delta = torch.zeros_like(base)
            else:
                candidate_feature = _candidate_episode_feature(name, evidence)
                candidate_input = torch.cat([global_features, candidate_feature], dim=-1)
                candidate_delta = self.candidate_projects[name](candidate_input).view_as(base)
            memory_bank[name] = base + candidate_delta + self.slot_embeddings[name].unsqueeze(0)
        return memory_bank


def _candidate_episode_feature(name: str, evidence: MultimodalEvidenceBank) -> torch.Tensor:
    if name == "TLEO":
        return evidence.local_features.mean(dim=1)
    if name == "SPO":
        return evidence.prototype_features.mean(dim=1)
    if name == "LRIO":
        if evidence.pair_features:
            return torch.stack([feature.mean(dim=1) for feature in evidence.pair_features.values()], dim=0).mean(dim=0)
        return evidence.low_rank_features.mean(dim=1)
    if name == "CATO":
        return evidence.alignment_features.mean(dim=1)
    if name == "TANSO":
        if evidence.all_pair_features:
            text_pairs = [
                feature.mean(dim=1)
                for key, feature in evidence.all_pair_features.items()
                if _pair_contains_modality(key, "text")
            ]
            if text_pairs:
                return torch.stack(text_pairs, dim=0).mean(dim=0)
        return evidence.low_rank_features.mean(dim=1)
    raise ValueError(f"unknown multimodal candidate: {name}")


def _pair_contains_modality(pair_key: str, modality: str) -> bool:
    return str(modality) in tuple(str(pair_key).split("__"))
