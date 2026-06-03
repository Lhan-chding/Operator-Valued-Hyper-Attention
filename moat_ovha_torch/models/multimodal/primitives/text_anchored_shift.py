from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class TANSOPrimitive(MultimodalCandidatePrimitive):
    name = "TANSO"

    def __init__(self, d_model: int, output_dim: int, sources: tuple[str, ...] = ("audio", "vision")):
        super().__init__()
        self.sources = tuple(str(source) for source in sources)
        self.anchor_query_proj = nn.Linear(d_model, d_model)
        self.text_key_proj = nn.Linear(d_model, d_model)
        self.text_value_proj = nn.Linear(d_model, d_model)
        self.source_key_proj = nn.ModuleDict({source: nn.Linear(d_model, d_model) for source in self.sources})
        self.source_value_proj = nn.ModuleDict({source: nn.Linear(d_model, d_model) for source in self.sources})
        self.source_gate = nn.ModuleDict({source: nn.Linear(d_model * 3, 1) for source in self.sources})
        self.source_shift = nn.ModuleDict({source: nn.Linear(d_model * 3, d_model) for source in self.sources})
        self.memory_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        text_tokens = evidence.field_features.get("text")
        if text_tokens is None:
            text_anchor = evidence.query_features
            text_entropy = torch.zeros((), dtype=evidence.query_features.dtype, device=evidence.query_features.device)
        else:
            text_anchor, text_entropy = _masked_attention(
                self.anchor_query_proj(evidence.query_features),
                self.text_key_proj(text_tokens),
                self.text_value_proj(text_tokens),
                batch.fields["text"].mask,
                temperature=params["shift_temperature"],
            )
        source_shifts = []
        source_gate_logits = []
        source_names = []
        source_entropy = {}
        for source in self.sources:
            tokens = evidence.field_features.get(source)
            if tokens is None or source not in batch.fields:
                continue
            attended, entropy = _masked_attention(
                text_anchor,
                self.source_key_proj[source](tokens),
                self.source_value_proj[source](tokens),
                batch.fields[source].mask,
                temperature=params["shift_temperature"],
            )
            source_input = torch.cat([evidence.query_features, text_anchor, attended], dim=-1)
            scale = params.get(f"{source}_shift_scale", torch.ones_like(params["scale"]))
            shift = self.source_shift[source](source_input) * scale
            source_shifts.append(shift)
            source_gate_logits.append(self.source_gate[source](source_input))
            source_names.append(source)
            source_entropy[source] = entropy
        if source_shifts:
            shift_stack = torch.stack(source_shifts, dim=-2)
            gate = torch.softmax(torch.cat(source_gate_logits, dim=-1), dim=-1).unsqueeze(-1)
            shift_feature = (gate * shift_stack).sum(dim=-2)
            source_load = {source: gate[..., index, :].mean() for index, source in enumerate(source_names)}
        else:
            shift_feature = torch.zeros_like(evidence.query_features)
            source_load = {}
        memory = self.memory_proj(memory_slot.mean(dim=1)).unsqueeze(1)
        feature = self.norm(shift_feature + memory)
        value = apply_scale_bias(self.head(feature), params)
        diagnostics = {
            "candidate": self.name,
            "text_anchor_entropy": text_entropy,
            "shift_magnitude": shift_feature.norm(dim=-1).mean(),
            "audio_shift_load": source_load.get("audio", torch.zeros((), dtype=value.dtype, device=value.device)),
            "vision_shift_load": source_load.get("vision", torch.zeros((), dtype=value.dtype, device=value.device)),
            "nonverbal_source_count": torch.as_tensor(float(len(source_names)), dtype=value.dtype, device=value.device),
            "source_attention_entropy": source_entropy,
            "shift_direction_alignment": _shift_direction_alignment(value, batch.target_y, batch.target_mask),
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)


def _masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = max(query.shape[-1] ** 0.5, 1.0)
    temp = temperature.clamp_min(0.25)
    while temp.ndim < query.ndim:
        temp = temp.unsqueeze(-1)
    scores = torch.matmul(query, key.transpose(1, 2)) / (scale * temp)
    valid = mask.to(dtype=torch.bool, device=scores.device).unsqueeze(1)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = weights / denom
    attended = torch.matmul(weights, value)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return attended, entropy


def _shift_direction_alignment(value: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool, device=value.device)
    if not bool(valid.any()):
        return torch.zeros((), dtype=value.dtype, device=value.device)
    pred = value[valid].reshape(-1).to(dtype=torch.float32)
    truth = target.to(device=value.device, dtype=torch.float32)[valid].reshape(-1)
    if pred.numel() < 2 or float(pred.norm().item()) <= 1e-12 or float(truth.norm().item()) <= 1e-12:
        return torch.zeros((), dtype=value.dtype, device=value.device)
    return torch.nn.functional.cosine_similarity(pred.view(1, -1), truth.view(1, -1)).squeeze(0).to(dtype=value.dtype)
