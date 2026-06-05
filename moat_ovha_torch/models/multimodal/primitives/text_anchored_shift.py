from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput, MultimodalCandidatePrimitive, apply_scale_bias


class TANSOPrimitive(MultimodalCandidatePrimitive):
    name = "TANSO"

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        sources: tuple[str, ...] = ("audio", "vision"),
        role: str = "shift",
        candidate_name: str = "TANSO",
    ):
        super().__init__()
        if role not in {"base", "shift"}:
            raise ValueError("TANSO role must be base or shift")
        self.name = str(candidate_name)
        self.role = role
        self.sources = tuple(str(source) for source in sources)
        self.anchor_query_proj = nn.Linear(d_model, d_model)
        self.text_key_proj = nn.Linear(d_model, d_model)
        self.text_value_proj = nn.Linear(d_model, d_model)
        self.source_key_proj = nn.ModuleDict({source: nn.Linear(d_model, d_model) for source in self.sources})
        self.source_value_proj = nn.ModuleDict({source: nn.Linear(d_model, d_model) for source in self.sources})
        self.source_gate = nn.ModuleDict({source: nn.Linear(d_model * 3, 1) for source in self.sources})
        self.source_shift = nn.ModuleDict({source: nn.Linear(d_model * 3, d_model) for source in self.sources})
        self.null_source_gate = nn.Linear(d_model * 3, 1)
        self.memory_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)
        self.register_buffer("lag_hypotheses", torch.linspace(-0.4, 0.4, steps=5), persistent=False)
        self._small_initialize_residual_paths()

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
        source_valid_by_name = {}
        memory = self.memory_proj(memory_slot.mean(dim=1)).unsqueeze(1)
        for source in self.sources:
            tokens = evidence.field_features.get(source)
            if tokens is None or source not in batch.fields:
                continue
            source_valid = batch.fields[source].mask.to(dtype=torch.bool, device=tokens.device).any(dim=1)
            source_valid_by_name[source] = source_valid
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
            source_valid_f = source_valid.to(dtype=attended.dtype, device=attended.device).view(-1, 1, 1)
            attended = attended * source_valid_f
            source_input = torch.cat([evidence.query_features, text_anchor, attended], dim=-1)
            scale = params.get(f"{source}_shift_scale", torch.ones_like(params["scale"]))
            shift = self.source_shift[source](source_input) * scale * source_valid_f
            raw_delta = self.head(self.norm(shift + memory)) * source_valid_f
            source_shifts.append(shift)
            source_gate_logits.append(self.source_gate[source](source_input))
            source_names.append(source)
            source_entropy[source] = entropy
            source_lag_load[source] = lag_load
            source_specific_raw_delta[source] = raw_delta

        null_attended = torch.zeros_like(text_anchor)
        null_input = torch.cat([evidence.query_features, text_anchor, null_attended], dim=-1)
        source_shifts.append(torch.zeros_like(evidence.query_features))
        source_gate_logits.append(self.null_source_gate(null_input))
        source_names.append("null")
        null_valid = torch.ones(
            evidence.query_features.shape[0],
            dtype=torch.bool,
            device=evidence.query_features.device,
        )
        source_valid_by_name["null"] = null_valid

        shift_stack = torch.stack(source_shifts, dim=-2)
        valid_stack = torch.stack(
            [source_valid_by_name[source] for source in source_names],
            dim=-1,
        )[:, None, :]
        gate_logits = torch.cat(source_gate_logits, dim=-1)
        gate_logits = gate_logits.masked_fill(~valid_stack, torch.finfo(gate_logits.dtype).min)
        gate = torch.softmax(gate_logits, dim=-1)
        gate = torch.where(valid_stack, gate, torch.zeros_like(gate))
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        gate = gate.unsqueeze(-1)
        shift_feature = (gate * shift_stack).sum(dim=-2)
        source_load = {source: gate[..., index, :].mean() for index, source in enumerate(source_names)}
        source_gate_tensor = {source: gate[..., index, 0] for index, source in enumerate(source_names)}

        valid_source_count = _valid_source_count_by_sample(source_valid_by_name, value_reference=evidence.query_features)
        nonverbal_source_count = _valid_source_count_by_sample(
            {source: valid for source, valid in source_valid_by_name.items() if source != "null"},
            value_reference=evidence.query_features,
        )
        non_null_gate = (
            1.0 - source_gate_tensor.get("null", torch.zeros_like(evidence.query_features[..., 0]))
        ).unsqueeze(-1)
        if self.role == "base":
            feature = self.norm(text_anchor + shift_feature + memory)
            value = apply_scale_bias(self.head(feature), params)
            semantic_role = "full_predictor"
        else:
            feature = self.norm(shift_feature + memory)
            has_valid_source = (nonverbal_source_count > 0).to(
                dtype=evidence.query_features.dtype,
                device=evidence.query_features.device,
            ).view(-1, 1, 1)
            value = apply_scale_bias(self.head(feature), params) * non_null_gate * has_valid_source
            semantic_role = "residual_delta"
        diagnostics = {
            "candidate": self.name,
            "tanso_version": "v2_temporal_lag_residual",
            "tanso_source_gate_version": "v3_null_source",
            "semantic_role": semantic_role,
            "null_source_enabled": True,
            "text_anchor_entropy": text_entropy,
            "text_token_anchor_count": text_anchor_count,
            "temporal_lag_hypotheses": tuple(float(value) for value in self.lag_hypotheses.detach().cpu().tolist()),
            "temporal_lag_load": source_lag_load,
            "shift_magnitude": shift_feature.norm(dim=-1).mean(),
            "audio_source_load": source_load.get("audio", torch.zeros((), dtype=value.dtype, device=value.device)),
            "vision_source_load": source_load.get("vision", torch.zeros((), dtype=value.dtype, device=value.device)),
            "null_source_load": source_load.get("null", torch.ones((), dtype=value.dtype, device=value.device)),
            "audio_shift_load": source_load.get("audio", torch.zeros((), dtype=value.dtype, device=value.device)),
            "vision_shift_load": source_load.get("vision", torch.zeros((), dtype=value.dtype, device=value.device)),
            "nonverbal_source_count": nonverbal_source_count.mean(),
            "source_count_including_null": valid_source_count.mean(),
            "source_gate_tensor": source_gate_tensor,
            "source_specific_raw_delta": source_specific_raw_delta,
            "source_specific_temporal_entropy": source_entropy,
            "source_attention_entropy": source_entropy,
        }
        return CandidateOutput(value=value, feature=feature, diagnostics=diagnostics)

    def _small_initialize_residual_paths(self) -> None:
        for source_shift in self.source_shift.values():
            nn.init.normal_(source_shift.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(source_shift.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)


def _valid_source_count_by_sample(source_valid_by_name: dict[str, torch.Tensor], value_reference: torch.Tensor) -> torch.Tensor:
    if not source_valid_by_name:
        return torch.zeros(value_reference.shape[0], dtype=value_reference.dtype, device=value_reference.device)
    return torch.stack(
        [valid.to(dtype=value_reference.dtype, device=value_reference.device) for valid in source_valid_by_name.values()],
        dim=-1,
    ).sum(dim=-1)


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
