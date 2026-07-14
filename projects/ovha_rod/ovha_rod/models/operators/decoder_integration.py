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
from .tensor_validation import tensor_value_checks_enabled


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
            if (tensor_value_checks_enabled(value)
                    and not torch.isfinite(value).all()):
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
    _validate_parent_inputs(query, parent_box_logits, matching_query_count)

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
    return integrate_matching_query_state(
        query=query,
        parent_box_logits=parent_box_logits,
        fused_matching_state=fused,
        matching_query_count=matching_query_count,
    )


def integrate_matching_query_state(
    *,
    query: Tensor,
    parent_box_logits: Tensor,
    fused_matching_state: DecoderResidualState,
    matching_query_count: int,
    fused_query_parent_box_logits: Optional[Tensor] = None,
) -> DecoderIntegrationResult:
    """Splice a bank-fused matching state into the full DINO query set.

    ``DecoderOperatorBank`` already performs structured residual fusion.  This
    boundary only restores the untouched denoising prefix and computes the
    matching-only head deltas; it must never fuse the bank output a second
    time.
    """
    _validate_parent_inputs(query, parent_box_logits, matching_query_count)
    if not isinstance(fused_matching_state, DecoderResidualState):
        raise TypeError(
            "fused_matching_state must be a DecoderResidualState")
    expected_query = (
        query.shape[0], matching_query_count, query.shape[-1])
    expected_boxes = (query.shape[0], matching_query_count, 4)
    expected_scores = (query.shape[0], matching_query_count)
    if fused_matching_state.query.shape != expected_query:
        raise ValueError(
            "fused_matching_state query must match trailing queries")
    if fused_matching_state.box_logits.shape != expected_boxes:
        raise ValueError(
            "fused_matching_state box logits must match trailing queries")
    if fused_matching_state.referent_score.shape != expected_scores:
        raise ValueError(
            "fused_matching_state referent score must match trailing queries")
    values = (
        fused_matching_state.query,
        fused_matching_state.box_logits,
        fused_matching_state.referent_score,
    )
    if any(
        value.device != query.device or value.dtype != query.dtype
        for value in values
    ):
        raise ValueError(
            "fused_matching_state must share query dtype and device")

    if fused_query_parent_box_logits is None:
        fused_query_parent_box_logits = parent_box_logits
    elif fused_query_parent_box_logits.shape != parent_box_logits.shape:
        raise ValueError(
            "fused_query_parent_box_logits must match parent_box_logits")
    elif (
        fused_query_parent_box_logits.device != query.device
        or fused_query_parent_box_logits.dtype != query.dtype
    ):
        raise ValueError(
            "fused_query_parent_box_logits must share query dtype/device")

    prefix_count = query.shape[1] - matching_query_count
    fused_query = torch.cat(
        (query[:, :prefix_count, :], fused_matching_state.query), dim=1)
    matching_box_logits = parent_box_logits[:, -matching_query_count:, :]
    operator_box_delta = (
        fused_matching_state.box_logits - matching_box_logits)
    fused_box_logits = torch.cat(
        (
            fused_query_parent_box_logits[:, :prefix_count, :],
            fused_query_parent_box_logits[:, -matching_query_count:, :]
            + operator_box_delta,
        ),
        dim=1,
    )
    reference_points = fused_box_logits.sigmoid()
    return DecoderIntegrationResult(
        query=fused_query,
        reference_points=reference_points,
        next_reference_points=reference_points.detach(),
        operator_box_delta=operator_box_delta,
        operator_referent_score=fused_matching_state.referent_score,
    )


def _validate_parent_inputs(
    query: Tensor,
    parent_box_logits: Tensor,
    matching_query_count: int,
) -> None:
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
        raise ValueError(
            "query and parent_box_logits must share float dtype/device")


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


def summarize_decoder_bank_outputs(outputs: Sequence[object]) -> dict[str, Tensor]:
    """Reduce layer bank telemetry to detached device-side scalar tensors."""
    from .decoder_bank import BankOutput, OPERATOR_NAMES

    rows = tuple(outputs)
    if not rows:
        return {}
    if not all(isinstance(row, BankOutput) for row in rows):
        raise TypeError("outputs must contain BankOutput values")

    values: dict[str, list[Tensor]] = {}

    def append(name: str, value: Tensor) -> None:
        values.setdefault(name, []).append(value)

    for row in rows:
        append(
            "decoder_memory_norm",
            row.memory_state.value.norm(dim=-1).mean(),
        )
        for index, operator_name in enumerate(OPERATOR_NAMES):
            router_weight = row.router.weights[..., index]
            append(
                f"decoder_{operator_name}_router_weight_mean",
                router_weight.mean(),
            )
            append(
                f"decoder_{operator_name}_availability_fraction",
                row.availability[..., index].to(
                    dtype=router_weight.dtype).mean(),
            )
            append(
                f"decoder_{operator_name}_reliability_mean",
                row.reliability.reliability[..., index].mean(),
            )
            residual = row.residuals.get(operator_name)
            if residual is None:
                gate_sigmoid_mean = router_weight.new_zeros(())
                gate_abs_mean = router_weight.new_zeros(())
            else:
                gate_sigmoid_mean = residual.gate_logits.sigmoid().mean()
                gate_abs_mean = residual.gate_logits.abs().mean()
            append(
                f"decoder_{operator_name}_gate_sigmoid_mean",
                gate_sigmoid_mean,
            )
            append(
                f"decoder_{operator_name}_gate_abs_mean",
                gate_abs_mean,
            )
        valid_ratio_mean = row.artifacts.get("valid_ratio_mean")
        if valid_ratio_mean is not None:
            append("decoder_valid_ratio_mean", valid_ratio_mean)

    return {
        name: torch.stack(tuple(layer_values)).mean().detach()
        for name, layer_values in values.items()
    }
