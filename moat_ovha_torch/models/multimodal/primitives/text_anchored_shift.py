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
        self.register_buffer("lag_hypotheses", torch.linspace(-0.4, 0.4, steps=5), persistent=False)
        self._zero_initialize_residual_paths()

    def forward(self, batch, memory_slot: torch.Tensor, evidence, params: dict[str, torch.Tensor], output_dim: int) -> CandidateOutput:
        text_tokens = evidence.field_features.get("text")
        if text_tokens is None or "text" not in batch.fields:
            text_anchor = evidence.query_features
            text_weights = torch.ones(
                evidence.query_features.shape[0],
                evidence.query_features.shape[1],
                1,
                dtype=evidence.query_features.dtype,
                device=evidence.query_features.device,
            )
            text_entropy = torch.zeros((), dtype=evidence.query_features.dtype, device=evidence.query_features.device)
            text_anchor_count = torch.zeros((), dtype=evidence.query_features.dtype, device=evidence.query_features.device)
            text_pos = batch.query.pos
        else:
            text_anchor, text_entropy, text_weights = _masked_attention_with_weights(
                self.anchor_query_proj(evidence.query_features),
                self.text_key_proj(text_tokens),
                self.text_value_proj(text_tokens),
                batch.fields["text"].mask,
                temperature=params["shift_temperature"],
            )
            text_anchor_count = batch.fields["text"].mask.to(dtype=evidence.query_features.dtype).sum(dim=1).mean()
            text_pos = batch.fields["text"].pos

        source_shifts = []
        source_gate_logits = []
        source_names = []
        source_entropy = {}
        source_lag_load = {}
        source_specific_raw_delta = {}
        memory = self.memory_proj(memory_slot.mean(dim=1)).unsqueeze(1)
        for source in self.sources:
            tokens = evidence.field_features.get(source)
            if tokens is None or source not in batch.fields:
                continue
            if text_tokens is None:
                attended, entropy, _ = _masked_attention_with_weights(
                    text_anchor,
                    self.source_key_proj[source](tokens),
                    self.source_value_proj[source](tokens),
                    batch.fields[source].mask,
                    temperature=params["shift_temperature"],
                )
                lag_load = {
                    f"lag_{index}": torch.zeros((), dtype=tokens.dtype, device=tokens.device)
                    for index in range(int(self.lag_hypotheses.numel()))
                }
            else:
                attended_by_text, entropy, lag_load = _temporal_lag_attention(
                    text_tokens=text_tokens,
                    text_pos=text_pos,
                    source_tokens=tokens,
                    source_pos=batch.fields[source].pos,
                    source_mask=batch.fields[source].mask,
                    text_query_proj=self.anchor_query_proj,
                    source_key_proj=self.source_key_proj[source],
                    source_value_proj=self.source_value_proj[source],
                    lag_hypotheses=self.lag_hypotheses.to(dtype=tokens.dtype, device=tokens.device),
                    lag_logits=params.get(f"{source}_lag_logits"),
                    lag_width=params.get("lag_width"),
                    temperature=params.get("temporal_temperature", params["shift_temperature"]),
                )
                attended = torch.matmul(text_weights.to(dtype=attended_by_text.dtype, device=attended_by_text.device), attended_by_text)
            source_input = torch.cat([evidence.query_features, text_anchor, attended], dim=-1)
            scale = params.get(f"{source}_shift_scale", torch.ones_like(params["scale"]))
            shift = self.source_shift[source](source_input) * scale
            source_shifts.append(shift)
            source_gate_logits.append(self.source_gate[source](source_input))
            source_names.append(source)
            source_entropy[source] = entropy
            source_lag_load[source] = lag_load
            source_specific_raw_delta[source] = self.head(self.norm(shift + memory))

        if source_shifts:
            shift_stack = torch.stack(source_shifts, dim=-2)
            gate = torch.softmax(torch.cat(source_gate_logits, dim=-1), dim=-1).unsqueeze(-1)
            shift_feature = (gate * shift_stack).sum(dim=-2)
            source_load = {source: gate[..., index, :].mean() for index, source in enumerate(source_names)}
            source_gate_tensor = {source: gate[..., index, 0] for index, source in enumerate(source_names)}
        else:
            shift_feature = torch.zeros_like(evidence.query_features)
            source_load = {}
            source_gate_tensor = {}

        feature = self.norm(shift_feature + memory)
        value = apply_scale_bias(self.head(feature), params)
        diagnostics = {
            "candidate": self.name,
            "tanso_version": "v2_temporal_lag_residual",
            "text_anchor_entropy": text_entropy,
            "text_token_anchor_count": text_anchor_count,
            "temporal_lag_hypotheses": tuple(float(value) for value in self.lag_hypotheses.detach().cpu().tolist()),
            "temporal_lag_load": source_lag_load,
            "shift_magnitude": shift_feature.norm(dim=-1).mean(),
            "audio_shift_load": source_load.get("audio", torch.zeros((), dtype=value.dtype, device=value.device)),
            "vision_shift_load": source_load.get("vision", torch.zeros((), dtype=value.dtype, device=value.device)),
            "nonverbal_source_count": torch.as_tensor(float(len(source_names)), dtype=value.dtype, device=value.device),
            "source_gate_tensor": source_gate_tensor,
            "source_specific_raw_delta": source_specific_raw_delta,
            "source_specific_temporal_entropy": source_entropy,
            "source_attention_entropy": source_entropy,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)

    def _zero_initialize_residual_paths(self) -> None:
        for source_shift in self.source_shift.values():
            nn.init.zeros_(source_shift.weight)
            nn.init.zeros_(source_shift.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


def _masked_attention_with_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = max(query.shape[-1] ** 0.5, 1.0)
    temp = temperature.clamp_min(0.25)
    while temp.ndim < query.ndim:
        temp = temp.unsqueeze(-1)
    scores = torch.matmul(query, key.transpose(1, 2)) / (scale * temp)
    valid = mask.to(dtype=torch.bool, device=scores.device).unsqueeze(1)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    attended = torch.matmul(weights, value)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return attended, entropy, weights


def _temporal_lag_attention(
    *,
    text_tokens: torch.Tensor,
    text_pos: torch.Tensor,
    source_tokens: torch.Tensor,
    source_pos: torch.Tensor,
    source_mask: torch.Tensor,
    text_query_proj: nn.Linear,
    source_key_proj: nn.Linear,
    source_value_proj: nn.Linear,
    lag_hypotheses: torch.Tensor,
    lag_logits: torch.Tensor | None,
    lag_width: torch.Tensor | None,
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    text_query = text_query_proj(text_tokens)
    source_key = source_key_proj(source_tokens)
    source_value = source_value_proj(source_tokens)
    scale = max(text_query.shape[-1] ** 0.5, 1.0)
    temp = temperature.clamp_min(0.25).mean(dim=1, keepdim=True).unsqueeze(-1)
    similarity = torch.matmul(text_query, source_key.transpose(1, 2)).unsqueeze(-1) / (scale * temp)
    width = torch.ones(1, dtype=text_query.dtype, device=text_query.device) if lag_width is None else lag_width.clamp_min(0.05)
    width = width.mean(dim=1, keepdim=True).unsqueeze(-1)
    text_time = text_pos[..., :1].to(dtype=text_query.dtype, device=text_query.device).unsqueeze(2)
    source_time = source_pos[..., :1].to(dtype=text_query.dtype, device=text_query.device).unsqueeze(1)
    lag = lag_hypotheses.view(1, 1, 1, -1)
    scores = similarity - ((text_time - source_time - lag) / width).square()
    if lag_logits is not None:
        scores = scores + lag_logits.to(dtype=scores.dtype, device=scores.device).mean(dim=1)[:, None, None, :]
    valid = source_mask.to(dtype=torch.bool, device=scores.device)[:, None, :, None]
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores.flatten(start_dim=2), dim=-1).view_as(scores)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    attended = (weights.unsqueeze(-1) * source_value[:, None, :, None, :]).sum(dim=(2, 3))
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=(2, 3)).mean()
    lag_load = weights.sum(dim=(0, 1, 2))
    lag_load = lag_load / lag_load.sum().clamp_min(1e-12)
    return attended, entropy, {f"lag_{index}": lag_load[index] for index in range(int(lag_load.shape[0]))}
