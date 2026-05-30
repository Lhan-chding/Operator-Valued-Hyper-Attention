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
        self.heads = nn.ModuleDict({name: nn.Linear(d_model * 2, 8) for name in candidate_names})
        self._initialize_identity()

    def forward(
        self,
        memory_bank: dict[str, torch.Tensor],
        evidence: MultimodalEvidenceBank,
        reliability: ReliabilityPrior | None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        params: dict[str, dict[str, torch.Tensor]] = {}
        for name in self.candidate_names:
            memory = memory_bank[name].mean(dim=1).unsqueeze(1).expand(-1, evidence.query_features.shape[1], -1)
            features = torch.cat([evidence.query_features, memory], dim=-1)
            raw = self.heads[name](features)
            params[name] = _params_for_name(name, raw)
        return params

    def _initialize_identity(self) -> None:
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)


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
