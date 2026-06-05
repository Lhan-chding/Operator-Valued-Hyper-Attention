from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias
from moat_ovha_torch.models.multimodal.primitives.phrase_region_similarity import _fit_output_dim, _masked_mean


class CATOPrimitive(MultimodalCandidatePrimitive):
    name = "CATO"

    def __init__(self, d_model: int, output_dim: int, modalities: tuple[str, ...] = ("region", "vision", "audio", "text")):
        super().__init__()
        self.modalities = modalities
        self.query_proj = nn.ModuleDict({name: nn.Linear(d_model, d_model) for name in modalities})
        self.key_proj = nn.ModuleDict({name: nn.Linear(d_model, d_model) for name in modalities})
        self.value_proj = nn.ModuleDict({name: nn.Linear(d_model, d_model) for name in modalities})
        self.source_gate = nn.ModuleDict({name: nn.Linear(d_model + 1, 1) for name in modalities})
        self.shared_query_proj = nn.Linear(d_model, d_model)
        self.shared_key_proj = nn.Linear(d_model, d_model)
        self.shared_value_proj = nn.Linear(d_model, d_model)
        self.shared_source_gate = nn.Linear(d_model + 1, 1)
        self.transport_proj = nn.Linear(d_model * 3, d_model)
        self.null_key = nn.Parameter(torch.zeros(d_model))
        self.null_value = nn.Parameter(torch.zeros(d_model))
        self.null_logit_bias = nn.Parameter(torch.tensor(-8.0))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        if "text" in evidence.field_features and "region" in evidence.field_features and output_dim == int(evidence.field_features["region"].shape[1]):
            return self._forward_region_alignment(batch, evidence, params, output_dim)
        temperature = params["alignment_temperature"].clamp_min(1e-4)
        transported_sources = []
        source_gate_logits = []
        source_entropies = []
        source_marginal_errors = []
        source_valid = []
        source_token_marginals = {}
        query_token_marginals = {}
        null_masses = []
        source_names = []
        for source_name, source in evidence.field_features.items():
            source_field = batch.fields[source_name]
            q_proj, k_proj, v_proj, gate_proj = self._modules_for(source_name)
            query = q_proj(evidence.query_features)
            key = k_proj(source)
            value = v_proj(source)
            null_key = self.null_key.view(1, 1, -1).expand(source.shape[0], 1, -1)
            null_value = self.null_value.view(1, 1, -1).expand(source.shape[0], 1, -1)
            key = torch.cat([key, null_key], dim=1)
            value = torch.cat([value, null_value], dim=1)
            score = torch.matmul(query, key.transpose(1, 2)) / temperature
            score[..., -1] = score[..., -1] + self.null_logit_bias
            null_pos = batch.query.pos[:, :1, :].to(dtype=source_field.pos.dtype, device=source_field.pos.device)
            source_pos = torch.cat([source_field.pos, null_pos], dim=1)
            score = score + _position_bias(batch.query.pos, source_pos, dtype=score.dtype, device=score.device)
            source_mask = source_field.mask.to(device=score.device)
            null_mask = torch.ones(source_mask.shape[0], 1, dtype=torch.bool, device=score.device)
            extended_mask = torch.cat([source_mask, null_mask], dim=1)
            score = score.masked_fill(~extended_mask.unsqueeze(1), -1e9)
            valid_source = source_mask.any(dim=1)
            transport = torch.softmax(score, dim=-1)
            transported = torch.matmul(transport, value)
            quality = _field_quality(source_field, dtype=transported.dtype, device=transported.device)
            gate_input = torch.cat([transported.mean(dim=1), quality], dim=-1)
            source_gate_logits.append(gate_proj(gate_input).unsqueeze(1))
            transported_sources.append(transported)
            source_entropies.append(-(transport * transport.clamp_min(1e-12).log()).sum(dim=-1).mean())
            source_marginal_errors.append(_source_transport_marginal_error(transport, source_mask))
            source_valid.append(valid_source)
            real_transport = transport[..., : source.shape[1]]
            source_token_marginals[source_name] = real_transport.mean(dim=(0, 1))
            query_token_marginals[source_name] = real_transport.sum(dim=-1).mean()
            null_masses.append(transport[..., -1].mean())
            source_names.append(source_name)
        if transported_sources:
            transported_stack = torch.stack(transported_sources, dim=-2)
            valid_stack = torch.stack(source_valid, dim=-1)
            gate_logits = torch.cat(source_gate_logits, dim=-1).masked_fill(~valid_stack.unsqueeze(1), -1e9)
            source_gates = torch.softmax(gate_logits, dim=-1)
            source_gates = source_gates * valid_stack.unsqueeze(1).to(dtype=source_gates.dtype)
            source_gates = source_gates / source_gates.sum(dim=-1, keepdim=True).clamp_min(1.0)
            source_gates = source_gates.unsqueeze(-1)
            transported = (source_gates * transported_stack).sum(dim=-2)
        else:
            source_gates = torch.zeros(1, device=evidence.query_features.device)
            transported = evidence.alignment_features
        scale = params["transport_scale"]
        memory = memory_slot.mean(dim=1).unsqueeze(1).expand_as(transported)
        feature = self.norm(
            self.transport_proj(
                torch.cat(
                    [
                        evidence.query_features,
                        scale * transported + 0.1 * memory,
                        evidence.query_features * transported,
                    ],
                    dim=-1,
                )
            )
        )
        value = apply_scale_bias(self.head(feature), params)
        diagnostics = {
            "alignment_entropy": torch.stack(source_entropies).mean() if source_entropies else evidence.alignment_entropy,
            "token_alignment_entropy": torch.stack(source_entropies).mean() if source_entropies else evidence.alignment_entropy,
            "top_k_alignment": _top_k_alignment(source_token_marginals) if transported_sources else torch.zeros((), device=value.device),
            "source_top_k_gate": _top_k_source_gate(source_gates.squeeze(-1)) if transported_sources else torch.zeros((), device=value.device),
            "transport_marginal_error": (
                torch.stack(source_marginal_errors).mean() + _transport_scale_error(params.get("transport_scale"), value.device)
                if source_marginal_errors
                else _transport_scale_error(params.get("transport_scale"), value.device)
            ),
            "alignment_temperature": params.get("alignment_temperature"),
            "transport_scale": params.get("transport_scale"),
            "source_modality": tuple(source_names),
            "source_gate": {
                name: source_gates[..., index, 0].mean()
                for index, name in enumerate(source_names)
            } if transported_sources else {},
            "valid_source_rate": {
                name: source_valid[index].to(dtype=value.dtype).mean()
                for index, name in enumerate(source_names)
            } if transported_sources else {},
            "source_token_marginal": source_token_marginals,
            "query_token_marginal": query_token_marginals,
            "null_mass": (
                torch.stack(null_masses).mean()
                if null_masses
                else torch.ones((), dtype=value.dtype, device=value.device)
            ),
            "learned_null_mass": (
                torch.stack(null_masses).mean()
                if null_masses
                else torch.ones((), dtype=value.dtype, device=value.device)
            ),
            "invalid_source_rate": (
                torch.stack([(~valid).to(dtype=value.dtype).mean() for valid in source_valid]).mean()
                if source_valid
                else torch.ones((), dtype=value.dtype, device=value.device)
            ),
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)

    def _modules_for(self, name: str):
        if name in self.query_proj:
            return self.query_proj[name], self.key_proj[name], self.value_proj[name], self.source_gate[name]
        return self.shared_query_proj, self.shared_key_proj, self.shared_value_proj, self.shared_source_gate

    def _forward_region_alignment(self, batch, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        text = evidence.field_features["text"]
        region = evidence.field_features["region"]
        region_field = batch.fields["region"]
        phrase = _masked_mean(text, batch.fields["text"].mask).unsqueeze(1).expand(-1, evidence.query_features.shape[1], -1)
        q_proj, k_proj, _v_proj, _gate_proj = self._modules_for("region")
        query = q_proj(phrase)
        key = k_proj(region)
        temperature = params["alignment_temperature"].clamp_min(0.05)
        logits = torch.matmul(query, key.transpose(1, 2)) / temperature
        logits = logits + _region_geometry_bias(region_field.pos, dtype=logits.dtype, device=logits.device).unsqueeze(1)
        region_mask = region_field.mask.to(device=logits.device)
        logits = logits.masked_fill(~region_mask.unsqueeze(1), -1e9)
        value = apply_scale_bias(_fit_output_dim(logits, output_dim), params)
        weights = torch.softmax(logits, dim=-1)
        feature = self.norm(torch.matmul(weights, region))
        diagnostics = {
            "alignment_entropy": -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean(),
            "token_alignment_entropy": -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean(),
            "top_k_alignment": weights.argmax(dim=-1).to(dtype=torch.float32).mean(),
            "source_top_k_gate": torch.zeros((), dtype=value.dtype, device=value.device),
            "transport_marginal_error": (weights.sum(dim=-1) - 1.0).abs().mean(),
            "alignment_temperature": params.get("alignment_temperature"),
            "transport_scale": params.get("transport_scale"),
            "source_modality": ("region",),
            "source_gate": {"region": torch.ones((), dtype=value.dtype, device=value.device)},
            "valid_source_rate": {"region": region_mask.any(dim=1).to(dtype=value.dtype).mean()},
            "source_token_marginal": {"region": weights.mean(dim=(0, 1))},
            "query_token_marginal": {"region": weights.sum(dim=-1).mean()},
            "null_mass": torch.zeros((), dtype=value.dtype, device=value.device),
            "learned_null_mass": torch.zeros((), dtype=value.dtype, device=value.device),
            "invalid_source_rate": (~region_mask.any(dim=1)).to(dtype=value.dtype).mean(),
            "direct_region_logits": True,
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _position_bias(query_pos: torch.Tensor, source_pos: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    pos_dims = min(int(query_pos.shape[-1]), int(source_pos.shape[-1]))
    if pos_dims <= 0:
        return torch.zeros(query_pos.shape[0], query_pos.shape[1], source_pos.shape[1], dtype=dtype, device=device)
    q_pos = query_pos[..., :pos_dims].to(dtype=dtype, device=device)
    s_pos = source_pos[..., :pos_dims].to(dtype=dtype, device=device)
    return -((q_pos.unsqueeze(2) - s_pos.unsqueeze(1)).square().sum(dim=-1))


def _top_k_source_gate(transport: torch.Tensor) -> torch.Tensor:
    k = min(2, transport.shape[-1])
    return torch.topk(transport, k=k, dim=-1).indices.to(dtype=torch.float32).mean()


def _top_k_alignment(source_token_marginals: dict[str, torch.Tensor]) -> torch.Tensor:
    if not source_token_marginals:
        return torch.zeros(())
    values = []
    for marginal in source_token_marginals.values():
        if marginal.numel() == 0:
            continue
        values.append(marginal.argmax(dim=-1).to(dtype=torch.float32).mean())
    if not values:
        return torch.zeros(())
    return torch.stack(values).mean()


def _source_transport_marginal_error(transport: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
    row_error = (transport.sum(dim=-1) - 1.0).abs().mean()
    real_transport = transport[..., : source_mask.shape[-1]]
    valid_count = source_mask.to(dtype=transport.dtype, device=transport.device).sum(dim=-1).clamp_min(1.0)
    target_col = source_mask.to(dtype=transport.dtype, device=transport.device) / valid_count.unsqueeze(-1)
    observed_col = real_transport.mean(dim=1)
    null_mass = transport[..., -1].mean()
    return row_error + (observed_col - target_col).abs().mean() + 0.1 * null_mass


def _transport_scale_error(transport_scale: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if transport_scale is None:
        return torch.zeros((), device=device)
    return (1.0 - transport_scale).abs().mean()


def _field_quality(field, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if field.quality is None:
        return field.mask.to(device=device).any(dim=1, keepdim=True).to(dtype=dtype)
    return field.quality.to(dtype=dtype, device=device).reshape(field.quality.shape[0], -1).mean(dim=1, keepdim=True)


def _region_geometry_bias(pos: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    geometry = pos.to(dtype=dtype, device=device)
    if int(geometry.shape[-1]) < 9:
        return torch.zeros(geometry.shape[0], geometry.shape[1], dtype=dtype, device=device)
    cx = geometry[..., 4]
    cy = geometry[..., 5]
    area = geometry[..., 8].clamp_min(0.0)
    center_distance = (cx - 0.5).square() + (cy - 0.5).square()
    return 0.01 * area - 0.01 * center_distance
