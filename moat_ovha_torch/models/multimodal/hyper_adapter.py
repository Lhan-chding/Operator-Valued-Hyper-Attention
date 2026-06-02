from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank
from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES
from moat_ovha_torch.models.multimodal.reliability_prior import ReliabilityPrior


ALLOWED_V1_ADAPTER_PARAMS = {
    "TLEO": ("lengthscale", "local_temperature", "scale", "bias"),
    "SPO": ("prototype_temperature", "prototype_logits_shift", "scale", "bias"),
    "LRIO": ("rank_logits", "interaction_temperature", "scale", "bias"),
    "CATO": ("alignment_temperature", "transport_scale", "scale", "bias"),
}


class MultimodalHyperAdapter(nn.Module):
    def __init__(self, d_model: int, candidate_names: tuple[str, ...] = MULTIMODAL_CANDIDATE_NAMES):
        super().__init__()
        self.candidate_names = candidate_names
        self.reliability_projection = nn.Linear(d_model + 1, d_model)
        self.router_projection = nn.Linear(len(candidate_names), d_model)
        self.heads = nn.ModuleDict({name: nn.Linear(d_model * 5, 8) for name in candidate_names})
        self._initialize_identity()

    def forward(
        self,
        memory_bank: dict[str, torch.Tensor],
        evidence: MultimodalEvidenceBank,
        reliability: ReliabilityPrior | None,
        router_weights: torch.Tensor | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        params: dict[str, dict[str, torch.Tensor]] = {}
        reliability_features = _reliability_query_features(self.reliability_projection, reliability, evidence)
        router_features = _router_query_features(self.router_projection, router_weights, evidence)
        for name in self.candidate_names:
            memory = memory_bank[name].mean(dim=1).unsqueeze(1).expand(-1, evidence.query_features.shape[1], -1)
            candidate_feature = _candidate_query_feature(name, evidence)
            features = torch.cat(
                [
                    evidence.query_features,
                    memory,
                    candidate_feature,
                    reliability_features,
                    router_features,
                ],
                dim=-1,
            )
            raw = self.heads[name](features)
            params[name] = _params_for_name(name, raw)
        return params

    def _initialize_identity(self) -> None:
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.router_projection.weight)
        nn.init.zeros_(self.router_projection.bias)


def _candidate_query_feature(name: str, evidence: MultimodalEvidenceBank) -> torch.Tensor:
    if name == "TLEO":
        return evidence.local_features
    if name == "SPO":
        return evidence.prototype_features
    if name == "LRIO":
        return evidence.low_rank_features
    if name == "CATO":
        return evidence.alignment_features
    raise ValueError(f"unknown multimodal candidate: {name}")


def _reliability_query_features(
    projection: nn.Linear,
    reliability: ReliabilityPrior | None,
    evidence: MultimodalEvidenceBank,
) -> torch.Tensor:
    if reliability is None:
        return torch.zeros_like(evidence.query_features)
    return projection(reliability.features)


def _router_query_features(
    projection: nn.Linear,
    router_weights: torch.Tensor | None,
    evidence: MultimodalEvidenceBank,
) -> torch.Tensor:
    if router_weights is None:
        return torch.zeros_like(evidence.query_features)
    return projection(router_weights.detach().to(dtype=evidence.query_features.dtype, device=evidence.query_features.device))


def _params_for_name(name: str, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    scale = 1.0 + 0.5 * torch.tanh(raw[..., 0:1])
    bias = 0.25 * torch.tanh(raw[..., 1:2])
    if name == "TLEO":
        return {
            "lengthscale": torch.nn.functional.softplus(raw[..., 2:3]) + 1e-3,
            "local_temperature": torch.nn.functional.softplus(raw[..., 3:4]) + 1e-3,
            "scale": scale,
            "bias": bias,
        }
    if name == "SPO":
        return {
            "prototype_temperature": torch.nn.functional.softplus(raw[..., 2:3]) + 1e-3,
            "prototype_logits_shift": raw[..., 3:7],
            "scale": scale,
            "bias": bias,
        }
    if name == "LRIO":
        return {
            "rank_logits": raw[..., 2:6],
            "interaction_temperature": torch.nn.functional.softplus(raw[..., 6:7]) + 1e-3,
            "scale": scale,
            "bias": bias,
        }
    if name == "CATO":
        return {
            "alignment_temperature": torch.nn.functional.softplus(raw[..., 2:3]) + 1e-3,
            "transport_scale": torch.sigmoid(raw[..., 3:4]),
            "scale": scale,
            "bias": bias,
        }
    raise ValueError(f"unknown multimodal candidate: {name}")
