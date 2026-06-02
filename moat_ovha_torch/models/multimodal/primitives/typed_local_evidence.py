from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class TLEOPrimitive(MultimodalCandidatePrimitive):
    name = "TLEO"

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.memory_gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        local_values = []
        local_entropies = []
        lengthscale = params["lengthscale"].clamp_min(1e-4)
        temperature = params["local_temperature"].clamp_min(1e-4)
        for name, projected_tokens in evidence.field_features.items():
            field = batch.fields[name]
            logits = _local_kernel_logits(
                evidence.query_features,
                projected_tokens,
                batch.query.pos,
                field.pos,
                lengthscale,
                temperature,
            )
            logits = logits.masked_fill(~field.mask.to(device=logits.device).unsqueeze(1), -1e9)
            weights = torch.softmax(logits, dim=-1)
            local_values.append(torch.matmul(weights, projected_tokens))
            local_entropies.append(-(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean())
        local_feature = torch.stack(local_values, dim=0).mean(dim=0)
        memory = memory_slot.mean(dim=1).unsqueeze(1).expand_as(local_feature)
        feature = self.norm(local_feature + self.memory_gate(torch.cat([local_feature, memory], dim=-1)))
        value = apply_scale_bias(self.head(feature), params)
        diagnostics = {
            "lengthscale": params.get("lengthscale"),
            "local_temperature": params.get("local_temperature"),
            "local_entropy": torch.stack(local_entropies).mean() if local_entropies else evidence.local_entropy,
            "local_window_size": torch.as_tensor(feature.shape[1], device=value.device),
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


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
