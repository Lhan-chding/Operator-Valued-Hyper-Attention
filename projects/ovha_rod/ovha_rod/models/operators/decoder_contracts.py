from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DecoderResidualState:
    """Structurally frozen decoder tensors before or after residual fusion."""

    query: Tensor
    box_logits: Tensor
    referent_score: Tensor

    def __post_init__(self) -> None:
        if self.query.ndim != 3:
            raise ValueError("query must have shape [B,Q,D]")
        batch_queries = self.query.shape[:2]
        if self.box_logits.shape != (*batch_queries, 4):
            raise ValueError("box_logits must have shape [B,Q,4]")
        if self.referent_score.shape != batch_queries:
            raise ValueError("referent_score must have shape [B,Q]")
        _validate_float_tensors(
            (self.query, self.box_logits, self.referent_score),
            names=("query", "box_logits", "referent_score"),
        )


@dataclass(frozen=True)
class DecoderOperatorResidual:
    """Structurally frozen three-channel residual proposal from one operator."""

    query_delta: Tensor
    box_delta: Tensor
    score_delta: Tensor
    gate_logits: Tensor
    valid: Tensor
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.query_delta.ndim != 3:
            raise ValueError("query_delta must have shape [B,Q,D]")
        batch_queries = self.query_delta.shape[:2]
        if self.box_delta.shape != (*batch_queries, 4):
            raise ValueError("box_delta must have shape [B,Q,4]")
        if self.score_delta.shape != batch_queries:
            raise ValueError("score_delta must have shape [B,Q]")
        if self.gate_logits.shape != (*batch_queries, 3):
            raise ValueError("gate_logits must have shape [B,Q,3]")
        if self.valid.shape != batch_queries or self.valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,Q] tensor")
        values = (
            self.query_delta,
            self.box_delta,
            self.score_delta,
            self.gate_logits,
        )
        _validate_float_tensors(
            values,
            names=(
                "query_delta",
                "box_delta",
                "score_delta",
                "gate_logits",
            ),
        )
        if self.valid.device != self.query_delta.device:
            raise ValueError("valid and residual tensors must share a device")
        diagnostics = dict(self.diagnostics)
        for name, value in diagnostics.items():
            if not isinstance(value, Tensor) or value.numel() != 1:
                raise ValueError(f"diagnostic {name!r} must be a scalar tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"diagnostic {name!r} must be finite")
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))


class StructuredResidualFusion(nn.Module):
    """Apply masked, zero-at-origin operator gates without mutating inputs."""

    def forward(
        self,
        parent: DecoderResidualState,
        residuals: Sequence[DecoderOperatorResidual],
    ) -> DecoderResidualState:
        query_delta = torch.zeros_like(parent.query)
        box_delta = torch.zeros_like(parent.box_logits)
        score_delta = torch.zeros_like(parent.referent_score)

        for residual in tuple(residuals):
            self._validate_compatible(parent, residual)
            valid = residual.valid.to(dtype=parent.query.dtype)
            gates = residual.gate_logits.tanh()
            query_delta = query_delta + (
                residual.query_delta * gates[..., 0, None] * valid[..., None]
            )
            box_delta = box_delta + (
                residual.box_delta * gates[..., 1, None] * valid[..., None]
            )
            score_delta = score_delta + (
                residual.score_delta * gates[..., 2] * valid
            )

        return DecoderResidualState(
            query=parent.query + query_delta,
            box_logits=parent.box_logits + box_delta,
            referent_score=parent.referent_score + score_delta,
        )

    @staticmethod
    def _validate_compatible(
        parent: DecoderResidualState,
        residual: DecoderOperatorResidual,
    ) -> None:
        if residual.query_delta.shape != parent.query.shape:
            raise ValueError("query_delta must match parent query shape")
        if residual.box_delta.shape != parent.box_logits.shape:
            raise ValueError("box_delta must match parent box_logits shape")
        if residual.score_delta.shape != parent.referent_score.shape:
            raise ValueError("score_delta must match parent referent_score shape")
        if residual.query_delta.device != parent.query.device:
            raise ValueError("parent and residual tensors must share a device")
        if residual.query_delta.dtype != parent.query.dtype:
            raise ValueError("parent and residual tensors must share a dtype")


def _validate_float_tensors(
    values: Sequence[Tensor], *, names: Sequence[str]
) -> None:
    device = values[0].device
    dtype = values[0].dtype
    for name, value in zip(names, values):
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if value.device != device or value.dtype != dtype:
            raise ValueError("structured tensors must share a device and dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
