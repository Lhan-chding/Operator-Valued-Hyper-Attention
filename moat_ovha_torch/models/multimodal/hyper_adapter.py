from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank
from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES
from moat_ovha_torch.models.multimodal.primitives.low_rank_interaction import LRIOPrimitive
from moat_ovha_torch.models.multimodal.reliability_prior import ReliabilityPrior


ALLOWED_V1_ADAPTER_PARAMS = {
    "TLEO": ("lengthscale", "local_temperature", "scale", "bias"),
    "SPO": ("prototype_temperature", "prototype_logits_shift", "scale", "bias"),
    "LRIO": ("rank_logits", "rank_logits_by_pair", "interaction_temperature", "interaction_temperature_by_pair", "scale", "bias"),
    "CATO": ("alignment_temperature", "transport_scale", "scale", "bias"),
    "TANSO": ("audio_shift_scale", "vision_shift_scale", "shift_temperature", "scale", "bias"),
}


class MultimodalHyperAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        candidate_names: tuple[str, ...] = MULTIMODAL_CANDIDATE_NAMES,
        lrio_pairs: tuple[tuple[str, str], ...] | None = None,
        lrio_rank_count: int = 4,
    ):
        super().__init__()
        self.candidate_names = candidate_names
        self.lrio_pairs = tuple(lrio_pairs or LRIOPrimitive.default_pairs())
        self.lrio_rank_count = int(lrio_rank_count)
        self.reliability_projection = nn.Linear(d_model + 1, d_model)
        self.router_projection = nn.Linear(len(candidate_names), d_model)
        self.heads = nn.ModuleDict({name: nn.Linear(d_model * 5, 8) for name in candidate_names})
        rng_state = torch.random.get_rng_state()
        self.lrio_pair_head = nn.Linear(d_model * 6, self.lrio_rank_count + 1)
        torch.random.set_rng_state(rng_state)
        self.lrio_pair_rank_bias = nn.ParameterDict(
            {
                _pair_key(pair): nn.Parameter(torch.zeros(self.lrio_rank_count))
                for pair in self.lrio_pairs
            }
        )
        self.lrio_pair_temperature_bias = nn.ParameterDict(
            {
                _pair_key(pair): nn.Parameter(torch.zeros(1))
                for pair in self.lrio_pairs
            }
        )
        self.lrio_pair_embedding = nn.ParameterDict(
            {
                _pair_key(pair): nn.Parameter(torch.zeros(d_model))
                for pair in self.lrio_pairs
            }
        )
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
            if name == "LRIO":
                params[name] = {
                    **params[name],
                    **self._lrio_pair_params(
                        evidence=evidence,
                        memory=memory,
                        reliability_features=reliability_features,
                        router_features=router_features,
                    ),
                }
        return params

    def _initialize_identity(self) -> None:
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.lrio_pair_head.weight)
        nn.init.zeros_(self.lrio_pair_head.bias)
        nn.init.zeros_(self.router_projection.weight)
        nn.init.zeros_(self.router_projection.bias)

    def _lrio_pair_params(
        self,
        *,
        evidence: MultimodalEvidenceBank,
        memory: torch.Tensor,
        reliability_features: torch.Tensor,
        router_features: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[str, ...]]:
        pair_features = evidence.pair_features or {}
        rank_logits = []
        temperatures = []
        pair_names = []
        for pair in self.lrio_pairs:
            key = _pair_key(pair)
            pair_feature = pair_features.get(key, evidence.low_rank_features)
            embedding = self.lrio_pair_embedding[key].view(1, 1, -1).expand_as(evidence.query_features)
            raw = self.lrio_pair_head(
                torch.cat(
                    [
                        evidence.query_features,
                        memory,
                        pair_feature,
                        reliability_features,
                        router_features,
                        embedding,
                    ],
                    dim=-1,
                )
            )
            rank_logits.append(raw[..., : self.lrio_rank_count] + self.lrio_pair_rank_bias[key].view(1, 1, -1))
            temperatures.append(
                torch.nn.functional.softplus(raw[..., self.lrio_rank_count : self.lrio_rank_count + 1] + self.lrio_pair_temperature_bias[key].view(1, 1, 1))
                + 0.1
            )
            pair_names.append(key)
        if not rank_logits:
            return {}
        return {
            "rank_logits_by_pair": torch.stack(rank_logits, dim=-2),
            "interaction_temperature_by_pair": torch.stack(temperatures, dim=-2),
            "rank_pair_names": tuple(pair_names),
        }


def _candidate_query_feature(name: str, evidence: MultimodalEvidenceBank) -> torch.Tensor:
    if name == "TLEO":
        return evidence.local_features
    if name == "SPO":
        return evidence.prototype_features
    if name == "LRIO":
        if evidence.pair_features:
            return torch.stack(list(evidence.pair_features.values()), dim=0).mean(dim=0)
        return evidence.low_rank_features
    if name == "CATO":
        return evidence.alignment_features
    if name == "TANSO":
        if evidence.all_pair_features:
            text_pairs = [
                feature
                for key, feature in evidence.all_pair_features.items()
                if key.startswith("text__")
            ]
            if text_pairs:
                return torch.stack(text_pairs, dim=0).mean(dim=0)
        return evidence.low_rank_features
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
            "prototype_temperature": torch.nn.functional.softplus(raw[..., 2:3]) + 0.1,
            "prototype_logits_shift": 0.1 * torch.tanh(raw[..., 3:7]),
            "scale": scale,
            "bias": bias,
        }
    if name == "LRIO":
        return {
            "rank_logits": raw[..., 2:6],
            "interaction_temperature": torch.nn.functional.softplus(raw[..., 6:7]) + 0.1,
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
    if name == "TANSO":
        return {
            "audio_shift_scale": torch.sigmoid(raw[..., 2:3]),
            "vision_shift_scale": torch.sigmoid(raw[..., 3:4]),
            "shift_temperature": torch.nn.functional.softplus(raw[..., 4:5]) + 0.1,
            "scale": scale,
            "bias": bias,
        }
    raise ValueError(f"unknown multimodal candidate: {name}")


def _pair_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"
