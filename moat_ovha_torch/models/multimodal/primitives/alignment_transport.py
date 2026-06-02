from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


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
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        temperature = params["alignment_temperature"].clamp_min(1e-4)
        transported_sources = []
        source_gate_logits = []
        source_entropies = []
        source_marginal_errors = []
        source_names = []
        for source_name, source in evidence.field_features.items():
            source_field = batch.fields[source_name]
            q_proj, k_proj, v_proj, gate_proj = self._modules_for(source_name)
            query = q_proj(evidence.query_features)
            key = k_proj(source)
            score = torch.matmul(query, key.transpose(1, 2)) / temperature
            score = score + _position_bias(batch.query.pos, source_field.pos, dtype=score.dtype, device=score.device)
            source_mask = source_field.mask.to(device=score.device)
            score = score.masked_fill(~source_mask.unsqueeze(1), -1e9)
            transport = torch.softmax(score, dim=-1)
            transported = torch.matmul(transport, v_proj(source))
            quality = _field_quality(source_field, dtype=transported.dtype, device=transported.device)
            gate_input = torch.cat([transported.mean(dim=1), quality], dim=-1)
            source_gate_logits.append(gate_proj(gate_input).unsqueeze(1))
            transported_sources.append(transported)
            source_entropies.append(-(transport * transport.clamp_min(1e-12).log()).sum(dim=-1).mean())
            source_marginal_errors.append(_source_transport_marginal_error(transport, source_mask))
            source_names.append(source_name)
        if transported_sources:
            transported_stack = torch.stack(transported_sources, dim=-2)
            source_gates = torch.softmax(torch.cat(source_gate_logits, dim=-1), dim=-1).unsqueeze(-1)
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
            "top_k_alignment": _top_k_alignment(source_gates.squeeze(-1)) if transported_sources else torch.zeros((), device=value.device),
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
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)

    def _modules_for(self, name: str):
        if name in self.query_proj:
            return self.query_proj[name], self.key_proj[name], self.value_proj[name], self.source_gate[name]
        return self.shared_query_proj, self.shared_key_proj, self.shared_value_proj, self.shared_source_gate


def _position_bias(query_pos: torch.Tensor, source_pos: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    pos_dims = min(int(query_pos.shape[-1]), int(source_pos.shape[-1]))
    if pos_dims <= 0:
        return torch.zeros(query_pos.shape[0], query_pos.shape[1], source_pos.shape[1], dtype=dtype, device=device)
    q_pos = query_pos[..., :pos_dims].to(dtype=dtype, device=device)
    s_pos = source_pos[..., :pos_dims].to(dtype=dtype, device=device)
    return -((q_pos.unsqueeze(2) - s_pos.unsqueeze(1)).square().sum(dim=-1))


def _top_k_alignment(transport: torch.Tensor) -> torch.Tensor:
    k = min(2, transport.shape[-1])
    return torch.topk(transport, k=k, dim=-1).indices.to(dtype=torch.float32).mean()


def _source_transport_marginal_error(transport: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
    row_error = (transport.sum(dim=-1) - 1.0).abs().mean()
    valid_count = source_mask.to(dtype=transport.dtype, device=transport.device).sum(dim=-1).clamp_min(1.0)
    target_col = source_mask.to(dtype=transport.dtype, device=transport.device) / valid_count.unsqueeze(-1)
    observed_col = transport.mean(dim=1)
    return row_error + (observed_col - target_col).abs().mean()


def _transport_scale_error(transport_scale: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if transport_scale is None:
        return torch.zeros((), device=device)
    return (1.0 - transport_scale).abs().mean()


def _field_quality(field, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if field.quality is None:
        return field.mask.to(dtype=dtype, device=device).mean(dim=1, keepdim=True)
    return field.quality.to(dtype=dtype, device=device).reshape(field.quality.shape[0], -1).mean(dim=1, keepdim=True)
