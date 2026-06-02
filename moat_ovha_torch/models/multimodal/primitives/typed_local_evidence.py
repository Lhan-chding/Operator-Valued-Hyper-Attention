from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class TLEOPrimitive(MultimodalCandidatePrimitive):
    name = "TLEO"

    def __init__(self, d_model: int, output_dim: int, modalities: tuple[str, ...] = ("text", "audio", "vision", "region")):
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
        self.memory_gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        local_values = []
        local_entropies = []
        source_gate_logits = []
        source_names = []
        lengthscale = params["lengthscale"].clamp_min(1e-4)
        temperature = params["local_temperature"].clamp_min(1e-4)
        for name, projected_tokens in evidence.field_features.items():
            field = batch.fields[name]
            q_proj, k_proj, v_proj, gate_proj = self._modules_for(name)
            query = q_proj(evidence.query_features)
            keys = k_proj(projected_tokens)
            logits = _local_kernel_logits(
                query,
                keys,
                batch.query.pos,
                field.pos,
                lengthscale,
                temperature,
            )
            logits = logits.masked_fill(~field.mask.to(device=logits.device).unsqueeze(1), -1e9)
            weights = torch.softmax(logits, dim=-1)
            value = torch.matmul(weights, v_proj(projected_tokens))
            quality = _field_quality(field, dtype=value.dtype, device=value.device)
            gate_input = torch.cat([value.mean(dim=1), quality], dim=-1)
            source_gate_logits.append(gate_proj(gate_input).unsqueeze(1))
            local_values.append(value)
            local_entropies.append(-(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean())
            source_names.append(name)
        if local_values:
            local_stack = torch.stack(local_values, dim=-2)
            source_gates = torch.softmax(torch.cat(source_gate_logits, dim=-1), dim=-1).unsqueeze(-1)
            local_feature = (source_gates * local_stack).sum(dim=-2)
        else:
            source_gates = torch.zeros(1, device=evidence.query_features.device)
            local_feature = evidence.local_features
        memory = memory_slot.mean(dim=1).unsqueeze(1).expand_as(local_feature)
        feature = self.norm(local_feature + self.memory_gate(torch.cat([local_feature, memory], dim=-1)))
        value = apply_scale_bias(self.head(feature), params)
        diagnostics = {
            "lengthscale": params.get("lengthscale"),
            "local_temperature": params.get("local_temperature"),
            "local_entropy": torch.stack(local_entropies).mean() if local_entropies else evidence.local_entropy,
            "local_window_size": torch.as_tensor(feature.shape[1], device=value.device),
            "modality_kernel_count": torch.as_tensor(float(len(source_names)), dtype=value.dtype, device=value.device),
            "modality_kernel_names": tuple(source_names),
            "modality_gate": {
                name: source_gates[..., index, 0].mean()
                for index, name in enumerate(source_names)
            } if local_values else {},
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)

    def _modules_for(self, name: str):
        if name in self.query_proj:
            return self.query_proj[name], self.key_proj[name], self.value_proj[name], self.source_gate[name]
        return self.shared_query_proj, self.shared_key_proj, self.shared_value_proj, self.shared_source_gate


def _local_kernel_logits(
    query: torch.Tensor,
    tokens: torch.Tensor,
    query_pos: torch.Tensor,
    token_pos: torch.Tensor,
    lengthscale: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    semantic = torch.matmul(query, tokens.transpose(1, 2)) / max(query.shape[-1] ** 0.5, 1.0)
    pos_dims = min(int(query_pos.shape[-1]), int(token_pos.shape[-1]))
    if pos_dims <= 0:
        distance_term = torch.zeros_like(semantic)
    else:
        q_pos = query_pos[..., :pos_dims].to(dtype=query.dtype, device=query.device)
        t_pos = token_pos[..., :pos_dims].to(dtype=query.dtype, device=query.device)
        distance_term = (q_pos.unsqueeze(2) - t_pos.unsqueeze(1)).square().sum(dim=-1)
    return -distance_term / lengthscale.square().clamp_min(1e-6) + semantic / temperature


def _field_quality(field, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if field.quality is None:
        return field.mask.to(dtype=dtype, device=device).mean(dim=1, keepdim=True)
    return field.quality.to(dtype=dtype, device=device).reshape(field.quality.shape[0], -1).mean(dim=1, keepdim=True)
