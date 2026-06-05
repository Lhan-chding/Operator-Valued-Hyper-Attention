from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class PRSOPrimitive(MultimodalCandidatePrimitive):
    name = "PRSO"

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.text_proj = nn.Linear(d_model, d_model)
        self.region_proj = nn.Linear(d_model, d_model)
        self.feature_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        if "text" not in evidence.field_features or "region" not in evidence.field_features:
            value = torch.zeros(
                evidence.query_features.shape[0],
                evidence.query_features.shape[1],
                output_dim,
                dtype=evidence.query_features.dtype,
                device=evidence.query_features.device,
            )
            return CandidateOutput(value=value, feature=evidence.query_features, diagnostics={"candidate": self.name, "direct_region_logits": False})
        text = evidence.field_features["text"]
        region = evidence.field_features["region"]
        text_mask = batch.fields["text"].mask
        region_mask = batch.fields["region"].mask.to(device=region.device)
        phrase = _masked_mean(text, text_mask).unsqueeze(1).expand(-1, evidence.query_features.shape[1], -1)
        query = self.text_proj(phrase)
        keys = self.region_proj(region)
        temperature = params.get("alignment_temperature", torch.ones_like(query[..., :1])).clamp_min(0.05)
        logits = torch.matmul(query, keys.transpose(1, 2)) / temperature
        logits = logits.masked_fill(~region_mask.unsqueeze(1), -1e9)
        value = apply_scale_bias(_fit_output_dim(logits, output_dim), params)
        weights = torch.softmax(logits.masked_fill(~region_mask.unsqueeze(1), -1e9), dim=-1)
        feature = self.norm(self.feature_proj(torch.matmul(weights, region)))
        diagnostics = {
            "candidate": self.name,
            "direct_region_logits": True,
            "phrase_region_similarity": True,
            "alignment_entropy": _masked_entropy(logits, region_mask),
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _fit_output_dim(value: torch.Tensor, output_dim: int) -> torch.Tensor:
    current = int(value.shape[-1])
    if current == output_dim:
        return value
    if current > output_dim:
        return value[..., :output_dim]
    pad = torch.full(
        (*value.shape[:-1], output_dim - current),
        -1e9,
        dtype=value.dtype,
        device=value.device,
    )
    return torch.cat([value, pad], dim=-1)


def _masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.masked_fill(~mask.unsqueeze(1), -1e9), dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
