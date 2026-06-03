from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class LRIOPrimitive(MultimodalCandidatePrimitive):
    name = "LRIO"

    @staticmethod
    def default_pairs() -> tuple[tuple[str, str], ...]:
        return (
            ("text", "audio"),
            ("text", "vision"),
            ("audio", "vision"),
            ("text", "region"),
            ("audio", "region"),
            ("vision", "region"),
        )

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        rank_count: int = 4,
        pairs: tuple[tuple[str, str], ...] | None = None,
    ):
        super().__init__()
        self.rank_count = rank_count
        self.pairs = tuple(_normalize_pair(pair) for pair in (pairs or self.default_pairs()))
        self.branch = nn.ModuleDict({_pair_key(pair): nn.Linear(d_model, rank_count * d_model) for pair in self.pairs})
        self.trunk = nn.ModuleDict({_pair_key(pair): nn.Linear(d_model * 2, rank_count * d_model) for pair in self.pairs})
        self.pair_gate = nn.ModuleDict({_pair_key(pair): nn.Linear(d_model * 2, 1) for pair in self.pairs})
        self.memory_proj = nn.Linear(d_model, d_model)
        self.pair_evidence_scale = nn.Parameter(torch.zeros(()))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        pooled = {
            name: _masked_mean(features, batch.fields[name].mask)
            for name, features in evidence.field_features.items()
        }
        pair_features = []
        pair_gates = []
        active_pair_names = []
        pair_rank_entropies = {}
        pair_admission_gate = {}
        pair_interaction_strength = {}
        rank_logits_by_pair = _rank_logits_by_pair(params, len(self.pairs))
        temperature_by_pair = _temperature_by_pair(params, len(self.pairs))
        reliability_by_pair = _pair_reliability_by_key(params, self.pairs, pooled, evidence.query_features)
        evidence_pair_features = evidence.pair_features or {}
        for pair in self.pairs:
            pair_index = self.pairs.index(pair)
            left_name, right_name = pair
            if left_name not in pooled or right_name not in pooled:
                continue
            left = pooled[left_name]
            right = pooled[right_name]
            key = _pair_key(pair)
            left_residual = left
            right_residual = right
            branch = self.branch[key](left_residual).view(left.shape[0], 1, self.rank_count, -1)
            right_query = torch.cat(
                [right_residual.unsqueeze(1).expand(-1, evidence.query_features.shape[1], -1), evidence.query_features],
                dim=-1,
            )
            trunk = self.trunk[key](right_query).view(right.shape[0], evidence.query_features.shape[1], self.rank_count, -1)
            rank_logits = rank_logits_by_pair[:, :, pair_index, :]
            temperature = temperature_by_pair[:, :, pair_index, :].clamp_min(0.5)
            rank_weights = torch.softmax(rank_logits / temperature, dim=-1)
            interaction = (rank_weights.unsqueeze(-1) * (branch * trunk)).sum(dim=-2)
            pair_evidence = evidence_pair_features.get(key)
            if pair_evidence is not None:
                interaction = interaction + self.pair_evidence_scale * pair_evidence
            gate_input = torch.cat([left, right], dim=-1)
            learned_gate = self.pair_gate[key](gate_input).unsqueeze(1)
            reliability = reliability_by_pair[key]
            admission = (reliability > 0.0).to(dtype=left.dtype, device=left.device)
            reliability_gate = reliability.clamp_min(1e-6).log().view(left.shape[0], 1, 1)
            pair_gates.append(learned_gate + reliability_gate)
            pair_features.append(interaction)
            active_pair_names.append(key)
            pair_rank_entropies[key] = _entropy(rank_logits)
            pair_admission_gate[key] = admission.mean()
            pair_interaction_strength[key] = interaction.norm(dim=-1).mean()
        if pair_features:
            pair_stack = torch.stack(pair_features, dim=-2)
            gate = torch.softmax(torch.cat(pair_gates, dim=-1), dim=-1).unsqueeze(-1)
            interaction_feature = (gate * pair_stack).sum(dim=-2)
            pair_load = {
                key: gate[..., index, :].mean()
                for index, key in enumerate(active_pair_names)
            }
        else:
            interaction_feature = evidence.low_rank_features
            pair_load = {}
        memory = self.memory_proj(memory_slot.mean(dim=1)).unsqueeze(1)
        feature = self.norm(interaction_feature + memory)
        value = apply_scale_bias(self.head(feature), params)
        rank_logits = params.get("rank_logits")
        diagnostics = {
            "rank_entropy": _entropy(rank_logits) if rank_logits is not None else torch.zeros((), device=value.device),
            "rank_top_k": _top_k(rank_logits, value.device),
            "pair_count": torch.as_tensor(float(len(pair_features)), dtype=value.dtype, device=value.device),
            "configured_pair_count": torch.as_tensor(float(len(self.pairs)), dtype=value.dtype, device=value.device),
            "active_pair_names": tuple(active_pair_names),
            "pair_load": pair_load,
            "pair_admission_gate": pair_admission_gate,
            "pair_reliability": {
                key: reliability_by_pair[key].mean()
                for key in active_pair_names
            },
            "pair_rank_entropy": pair_rank_entropies,
            "pair_interaction_strength": interaction_feature.norm(dim=-1).mean(),
            "pair_interaction_strength_by_pair": pair_interaction_strength,
            "interaction_temperature": params.get("interaction_temperature"),
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _normalize_pair(pair: tuple[str, str]) -> tuple[str, str]:
    left, right = str(pair[0]), str(pair[1])
    if left == right:
        raise ValueError("LRIO pairs must contain two distinct modalities")
    return left, right


def _pair_key(pair: tuple[str, str]) -> str:
    left, right = _normalize_pair(pair)
    return f"{left}__{right}"


def _rank_logits_by_pair(params: dict[str, torch.Tensor], pair_count: int) -> torch.Tensor:
    if "rank_logits_by_pair" in params:
        pair_logits = params["rank_logits_by_pair"]
        global_logits = params["rank_logits"].unsqueeze(-2).expand(*params["rank_logits"].shape[:-1], pair_count, params["rank_logits"].shape[-1])
        if pair_logits.shape[-2] == pair_count:
            return global_logits + pair_logits
        return global_logits
    rank_logits = params["rank_logits"]
    return rank_logits.unsqueeze(-2).expand(*rank_logits.shape[:-1], pair_count, rank_logits.shape[-1])


def _temperature_by_pair(params: dict[str, torch.Tensor], pair_count: int) -> torch.Tensor:
    if "interaction_temperature_by_pair" in params:
        pair_temperature = params["interaction_temperature_by_pair"]
        global_temperature = params["interaction_temperature"].unsqueeze(-2).expand(
            *params["interaction_temperature"].shape[:-1],
            pair_count,
            params["interaction_temperature"].shape[-1],
        )
        if pair_temperature.shape[-2] == pair_count:
            return global_temperature + pair_temperature
        return global_temperature
    temperature = params["interaction_temperature"]
    return temperature.unsqueeze(-2).expand(*temperature.shape[:-1], pair_count, temperature.shape[-1])


def _pair_reliability_by_key(
    params: dict[str, torch.Tensor],
    pairs: tuple[tuple[str, str], ...],
    pooled: dict[str, torch.Tensor],
    query_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size = int(query_features.shape[0])
    fallback = torch.ones(batch_size, dtype=query_features.dtype, device=query_features.device)
    pair_reliability = params.get("pair_reliability")
    pair_names = params.get("pair_names", ())
    if pair_reliability is None or not pair_names:
        return {_pair_key(pair): fallback for pair in pairs}
    reliability_by_name = {
        str(name): pair_reliability[:, index].to(dtype=query_features.dtype, device=query_features.device)
        for index, name in enumerate(pair_names)
    }
    return {
        _pair_key(pair): reliability_by_name.get(_pair_key(pair), fallback)
        for pair in pairs
        if pair[0] in pooled and pair[1] in pooled
    }


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()


def _top_k(logits: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if logits is None:
        return torch.zeros((), device=device)
    k = min(2, logits.shape[-1])
    return torch.topk(logits, k=k, dim=-1).indices.to(dtype=torch.float32).mean()
