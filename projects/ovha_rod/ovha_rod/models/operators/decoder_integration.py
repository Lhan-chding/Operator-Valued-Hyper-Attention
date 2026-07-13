from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .decoder_contracts import (
    DecoderOperatorResidual,
    DecoderResidualState,
    StructuredResidualFusion,
)


@dataclass(frozen=True)
class DecoderIntegrationResult:
    """Immutable output of one decoder-layer operator integration step."""

    query: Tensor
    reference_points: Tensor
    next_reference_points: Tensor
    operator_box_delta: Tensor
    operator_referent_score: Tensor

    def __post_init__(self) -> None:
        if self.query.ndim != 3:
            raise ValueError("query must have shape [B,N,D]")
        batch_size, total_queries = self.query.shape[:2]
        if self.reference_points.shape != (batch_size, total_queries, 4):
            raise ValueError("reference_points must have shape [B,N,4]")
        if self.next_reference_points.shape != self.reference_points.shape:
            raise ValueError(
                "next_reference_points must match reference_points shape")
        matching_shape = self.operator_referent_score.shape
        if len(matching_shape) != 2 or matching_shape[0] != batch_size:
            raise ValueError(
                "operator_referent_score must have shape [B,Q]")
        if self.operator_box_delta.shape != (*matching_shape, 4):
            raise ValueError("operator_box_delta must have shape [B,Q,4]")
        values = (
            self.query,
            self.reference_points,
            self.next_reference_points,
            self.operator_box_delta,
            self.operator_referent_score,
        )
        device, dtype = self.query.device, self.query.dtype
        for value in values:
            if (
                not value.is_floating_point()
                or value.device != device
                or value.dtype != dtype
            ):
                raise ValueError(
                    "decoder integration tensors must share float dtype/device")
            if not torch.isfinite(value).all():
                raise ValueError(
                    "decoder integration tensors must contain finite values")
        if self.next_reference_points.requires_grad:
            raise ValueError("next_reference_points must be detached")


ResidualInput = Union[
    DecoderOperatorResidual,
    Sequence[DecoderOperatorResidual],
]


def apply_matching_query_residual(
    *,
    query: Tensor,
    parent_box_logits: Tensor,
    residual: ResidualInput,
    matching_query_count: int,
) -> DecoderIntegrationResult:
    """Fuse operator residuals into the trailing matching-query partition.

    DINO places denoising queries first and matching queries last.  This helper
    therefore never indexes the DN prefix through an operator and constructs
    fresh tensors rather than mutating the parent decoder state.
    """
    if query.ndim != 3:
        raise ValueError("query must have shape [B,N,D]")
    if parent_box_logits.shape != (*query.shape[:2], 4):
        raise ValueError(
            "parent_box_logits must have shape [B,N,4] matching query")
    if (
        not isinstance(matching_query_count, int)
        or isinstance(matching_query_count, bool)
        or matching_query_count <= 0
        or matching_query_count > query.shape[1]
    ):
        raise ValueError(
            "matching_query_count must be a positive count no larger than N")
    if (
        not query.is_floating_point()
        or parent_box_logits.device != query.device
        or parent_box_logits.dtype != query.dtype
    ):
        raise ValueError("query and parent_box_logits must share float dtype/device")

    residuals = (
        (residual,)
        if isinstance(residual, DecoderOperatorResidual)
        else tuple(residual)
    )
    if not residuals:
        raise ValueError("residual must contain at least one operator output")
    for item in residuals:
        if not isinstance(item, DecoderOperatorResidual):
            raise TypeError("residual entries must be DecoderOperatorResidual")
        if item.query_delta.shape[1] != matching_query_count:
            raise ValueError(
                "matching_query_count must match residual query count")

    matching_query = query[:, -matching_query_count:, :]
    matching_box_logits = parent_box_logits[:, -matching_query_count:, :]
    parent = DecoderResidualState(
        query=matching_query,
        box_logits=matching_box_logits,
        referent_score=query.new_zeros(query.shape[0], matching_query_count),
    )
    fused = StructuredResidualFusion()(parent, residuals)
    prefix_count = query.shape[1] - matching_query_count
    fused_query = torch.cat(
        (query[:, :prefix_count, :], fused.query), dim=1)
    fused_box_logits = torch.cat(
        (parent_box_logits[:, :prefix_count, :], fused.box_logits), dim=1)
    reference_points = fused_box_logits.sigmoid()
    return DecoderIntegrationResult(
        query=fused_query,
        reference_points=reference_points,
        next_reference_points=reference_points.detach(),
        operator_box_delta=fused.box_logits - matching_box_logits,
        operator_referent_score=fused.referent_score,
    )


def stack_decoder_operator_outputs(
    outputs: Sequence[DecoderIntegrationResult],
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Stack matching-only outputs as ``[L,B,Q,...]`` or return two Nones."""
    rows = tuple(outputs)
    if not rows:
        return None, None
    if not all(isinstance(row, DecoderIntegrationResult) for row in rows):
        raise TypeError("outputs must contain DecoderIntegrationResult values")
    box_shape = rows[0].operator_box_delta.shape
    score_shape = rows[0].operator_referent_score.shape
    for row in rows[1:]:
        if row.operator_box_delta.shape != box_shape:
            raise ValueError("operator box delta shapes must agree across layers")
        if row.operator_referent_score.shape != score_shape:
            raise ValueError(
                "operator referent score shapes must agree across layers")
    return (
        torch.stack(tuple(row.operator_box_delta for row in rows)),
        torch.stack(tuple(row.operator_referent_score for row in rows)),
    )
