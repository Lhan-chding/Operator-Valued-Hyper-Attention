from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias
from moat_ovha_torch.models.multimodal.primitives.phrase_region_similarity import _fit_output_dim, _masked_mean


class SROPrimitive(MultimodalCandidatePrimitive):
    name = "SRO"

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.text_proj = nn.Linear(d_model, d_model)
        self.geometry_proj = nn.Linear(9, d_model)
        self.score = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
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
        region_field = batch.fields["region"]
        region_mask = region_field.mask.to(device=region.device)
        geometry = _region_geometry(region_field.pos, dtype=region.dtype, device=region.device)
        phrase = self.text_proj(_masked_mean(text, batch.fields["text"].mask))
        geo = self.geometry_proj(geometry)
        phrase_by_region = phrase.unsqueeze(1).expand_as(geo)
        score_input = torch.cat([phrase_by_region, geo, phrase_by_region * geo], dim=-1)
        logits = self.score(score_input).squeeze(-1).unsqueeze(1)
        logits = logits.masked_fill(~region_mask.unsqueeze(1), -1e9)
        value = apply_scale_bias(_fit_output_dim(logits, output_dim), params)
        weights = torch.softmax(logits, dim=-1)
        feature = self.norm(self.feature_proj(torch.matmul(weights, region)))
        diagnostics = {
            "candidate": self.name,
            "direct_region_logits": True,
            "spatial_relation_geometry": True,
            "geometry_dim": torch.as_tensor(float(geometry.shape[-1]), dtype=value.dtype, device=value.device),
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _region_geometry(pos: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    geometry = pos.to(dtype=dtype, device=device)
    if int(geometry.shape[-1]) >= 9:
        return geometry[..., :9]
    if int(geometry.shape[-1]) >= 4:
        x1, y1, x2, y2 = geometry[..., 0], geometry[..., 1], geometry[..., 2], geometry[..., 3]
        width = (x2 - x1).clamp_min(0.0)
        height = (y2 - y1).clamp_min(0.0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        area = width * height
        return torch.stack([x1, y1, x2, y2, cx, cy, width, height, area], dim=-1)
    pad = torch.zeros(*geometry.shape[:-1], 9 - int(geometry.shape[-1]), dtype=dtype, device=device)
    return torch.cat([geometry, pad], dim=-1)
