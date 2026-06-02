from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class CATOPrimitive(MultimodalCandidatePrimitive):
    name = "CATO"

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.transport_proj = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        source_name = _alignment_source_name(evidence.field_features)
        source = evidence.field_features[source_name]
        source_field = batch.fields[source_name]
        temperature = params["alignment_temperature"].clamp_min(1e-4)
        query = self.query_proj(evidence.query_features)
        key = self.key_proj(source)
        score = torch.matmul(query, key.transpose(1, 2)) / temperature
        score = score + _position_bias(batch.query.pos, source_field.pos, dtype=score.dtype, device=score.device)
        score = score.masked_fill(~source_field.mask.to(device=score.device).unsqueeze(1), -1e9)
        transport = torch.softmax(score, dim=-1)
        transported = torch.matmul(transport, self.value_proj(source))
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
            "alignment_entropy": -(transport * transport.clamp_min(1e-12).log()).sum(dim=-1).mean(),
            "top_k_alignment": _top_k_alignment(transport),
            "transport_marginal_error": _transport_marginal_error(params.get("transport_scale"), value.device),
            "alignment_temperature": params.get("alignment_temperature"),
            "transport_scale": params.get("transport_scale"),
            "source_modality": source_name,
            "candidate": self.name,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _alignment_source_name(field_features: dict[str, torch.Tensor]) -> str:
    for name in ("region", "vision", "audio", "text"):
        if name in field_features:
            return name
    return next(iter(field_features))


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


def _transport_marginal_error(transport_scale: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if transport_scale is None:
        return torch.zeros((), device=device)
    return (1.0 - transport_scale).abs().mean()
