"""Multi-scale texture and local-evidence operator (MS-TLEO).

The sampler emits no ``Q x Q`` tensor and never repeats a feature map per
query. With a fixed 25 samples per query, sampling costs
``O(L * B * D * Q * 25)`` time and ``O(B * D * Q * 25)`` peak workspace;
the three output heads add ``O(B * Q * D^2)`` time. This isolated module is
not imported by any formal training config.

Coordinates use normalized continuous image edges: 0 and 1 are the left/top
and right/bottom image edges. They are converted to ``grid_sample`` space as
``2 * coordinate - 1`` and sampled with ``align_corners=False``. Samples
outside the image use zero padding. Degenerate boxes are rejected for valid
queries but tolerated for masked queries, whose outputs are exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoder_contracts import DecoderOperatorResidual


@dataclass(frozen=True)
class MSTLEOEvidence:
    """Mean-aggregated interior, boundary, and context descriptors."""

    interior: Tensor
    boundary: Tensor
    context: Tensor

    def __post_init__(self) -> None:
        if self.interior.ndim != 3:
            raise ValueError("MS-TLEO evidence must have shape [B,Q,D]")
        if self.boundary.shape != self.interior.shape:
            raise ValueError("boundary evidence must match interior evidence")
        if self.context.shape != self.interior.shape:
            raise ValueError("context evidence must match interior evidence")


def normalized_image_to_grid(coordinates: Tensor) -> Tensor:
    """Map normalized image-edge coordinates to align-corners-false space."""

    return coordinates * 2.0 - 1.0


class MSTLEO(nn.Module):
    """Compare local evidence around each valid normalized ``cxcywh`` box."""

    _INTERIOR_POINT_COUNT = 9
    _BOUNDARY_POINT_COUNT = 8
    _CONTEXT_POINT_COUNT = 8

    def __init__(self, d_model: int = 256, context_scale: float = 1.5) -> None:
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(context_scale, (int, float)):
            raise ValueError("context_scale must be a finite number greater than 1")
        if not math.isfinite(float(context_scale)) or context_scale <= 1.0:
            raise ValueError("context_scale must be a finite number greater than 1")

        self.d_model = d_model
        self.context_scale = float(context_scale)
        descriptor_dim = d_model * 3
        self.query_head = nn.Linear(descriptor_dim, d_model, bias=False)
        self.box_head = nn.Linear(descriptor_dim, 4, bias=False)
        self.score_head = nn.Linear(descriptor_dim, 1, bias=False)
        self.gate_head = nn.Linear(descriptor_dim, 3, bias=False)
        nn.init.zeros_(self.gate_head.weight)

        interior, boundary = _base_layouts()
        self.register_buffer("_interior_offsets", interior, persistent=False)
        self.register_buffer("_boundary_offsets", boundary, persistent=False)
        self.register_buffer(
            "_context_offsets",
            boundary * self.context_scale,
            persistent=False,
        )

    def forward(
        self,
        feature_maps: tuple[Tensor, ...],
        boxes: Tensor,
        valid: Tensor,
    ) -> DecoderOperatorResidual:
        evidence = self.extract_evidence(feature_maps, boxes, valid)
        if evidence.interior.shape[-1] != self.d_model:
            raise ValueError("feature map channels must equal d_model")

        descriptor = torch.cat(
            (
                evidence.interior,
                evidence.boundary - evidence.interior,
                evidence.context - evidence.boundary,
            ),
            dim=-1,
        )
        mask = valid[..., None].to(dtype=descriptor.dtype)
        query_delta = self.query_head(descriptor) * mask
        box_delta = self.box_head(descriptor) * mask
        score_delta = self.score_head(descriptor).squeeze(-1) * mask.squeeze(-1)
        gate_logits = self.gate_head(descriptor) * mask

        return DecoderOperatorResidual(
            query_delta=query_delta,
            box_delta=box_delta,
            score_delta=score_delta,
            gate_logits=gate_logits,
            valid=valid,
            diagnostics={
                "interior_energy": evidence.interior.square().mean(),
                "boundary_contrast": (
                    evidence.boundary - evidence.interior
                ).abs().mean(),
                "context_contrast": (
                    evidence.context - evidence.boundary
                ).abs().mean(),
                "valid_fraction": valid.to(dtype=descriptor.dtype).mean(),
            },
        )

    def extract_evidence(
        self,
        feature_maps: tuple[Tensor, ...],
        boxes: Tensor,
        valid: Tensor,
    ) -> MSTLEOEvidence:
        """Sample fixed layouts and equally average descriptors over levels."""

        feature_maps = _validate_inputs(feature_maps, boxes, valid)
        safe_boxes = torch.where(valid[..., None], boxes, torch.zeros_like(boxes))
        grid = self._sampling_grid(safe_boxes)
        point_counts = (
            self._INTERIOR_POINT_COUNT,
            self._BOUNDARY_POINT_COUNT,
            self._CONTEXT_POINT_COUNT,
        )
        per_level = _sample_level(feature_maps[0], grid, point_counts)
        for feature_map in feature_maps[1:]:
            current_level = _sample_level(feature_map, grid, point_counts)
            per_level = tuple(
                accumulated + current
                for accumulated, current in zip(per_level, current_level)
            )
        interior, boundary, context = (
            descriptor / len(feature_maps) for descriptor in per_level
        )
        mask = valid[..., None].to(dtype=boxes.dtype)
        return MSTLEOEvidence(
            interior=interior * mask,
            boundary=boundary * mask,
            context=context * mask,
        )

    def _sampling_grid(self, boxes: Tensor) -> Tensor:
        offsets = torch.cat(
            (
                self._interior_offsets,
                self._boundary_offsets,
                self._context_offsets,
            ),
            dim=0,
        ).to(device=boxes.device, dtype=boxes.dtype)
        coordinates = (
            boxes[..., None, :2] + offsets[None, None] * boxes[..., None, 2:]
        )
        return normalized_image_to_grid(coordinates)


def _base_layouts() -> tuple[Tensor, Tensor]:
    interior_axis = torch.tensor((-0.25, 0.0, 0.25), dtype=torch.float32)
    boundary_axis = torch.tensor((-0.5, 0.0, 0.5), dtype=torch.float32)
    interior_y, interior_x = torch.meshgrid(
        interior_axis, interior_axis, indexing="ij")
    boundary_y, boundary_x = torch.meshgrid(
        boundary_axis, boundary_axis, indexing="ij")
    interior = torch.stack((interior_x.flatten(), interior_y.flatten()), dim=-1)
    boundary_grid = torch.stack(
        (boundary_x.flatten(), boundary_y.flatten()), dim=-1)
    boundary = boundary_grid[boundary_grid.abs().amax(dim=-1) == 0.5]
    return interior, boundary


def _validate_inputs(
    feature_maps: tuple[Tensor, ...],
    boxes: Tensor,
    valid: Tensor,
) -> tuple[Tensor, ...]:
    if not isinstance(feature_maps, tuple) or not feature_maps:
        raise ValueError("feature_maps must be a non-empty tuple")
    if not all(isinstance(feature_map, Tensor) for feature_map in feature_maps):
        raise ValueError("feature_maps must contain only tensors")
    if any(feature_map.ndim != 4 for feature_map in feature_maps):
        raise ValueError("feature_maps must have shape [B,D,H,W]")
    if any(min(feature_map.shape[1:]) <= 0 for feature_map in feature_maps):
        raise ValueError(
            "feature_maps must have positive channel and spatial dimensions")

    reference = feature_maps[0]
    if not reference.is_floating_point():
        raise ValueError("feature_maps must be floating point")
    if any(feature_map.shape[0] != reference.shape[0] for feature_map in feature_maps):
        raise ValueError("feature_maps must share the same batch dimension")
    if any(feature_map.shape[1] != reference.shape[1] for feature_map in feature_maps):
        raise ValueError("feature_maps must share the same channel dimension")
    if any(
        feature_map.device != reference.device or feature_map.dtype != reference.dtype
        for feature_map in feature_maps
    ):
        raise ValueError("feature_maps must share a device and dtype")
    if any(not torch.isfinite(feature_map).all() for feature_map in feature_maps):
        raise ValueError("feature_maps must contain only finite values")

    if (
        boxes.ndim != 3
        or boxes.shape[0] != reference.shape[0]
        or boxes.shape[1] <= 0
        or boxes.shape[-1] != 4
    ):
        raise ValueError("boxes must have shape [B,Q,4]")
    if not boxes.is_floating_point():
        raise ValueError("boxes must be floating point")
    if boxes.device != reference.device or boxes.dtype != reference.dtype:
        raise ValueError("boxes and feature_maps must share a device and dtype")
    if not torch.isfinite(boxes).all():
        raise ValueError("boxes must contain only finite values")
    if valid.shape != boxes.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean [B,Q] tensor")
    if valid.device != boxes.device:
        raise ValueError("valid, boxes, and feature_maps must share a device")
    if valid.any() and not (boxes[..., 2:][valid] > 0).all():
        raise ValueError("valid boxes must have positive width and height")
    return feature_maps


def _sample_level(
    feature_map: Tensor,
    grid: Tensor,
    point_counts: Sequence[int],
) -> tuple[Tensor, Tensor, Tensor]:
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    layouts = sampled.split(tuple(point_counts), dim=-1)
    return tuple(layout.mean(dim=-1).transpose(1, 2) for layout in layouts)
