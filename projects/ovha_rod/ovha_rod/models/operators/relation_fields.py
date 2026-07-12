from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


RELATION_TYPES = (
    "null", "left", "right", "above", "below", "near", "far", "overlap")


class RelationFieldBank(nn.Module):
    """Resolution-relative, full-field spatial relation transforms.

    Directional fields use cumulative mass and therefore cover arbitrary
    distances without an O((HW)^2) token graph.  Gaussian widths are fractions
    of the feature-map extent, so a scale has the same meaning at every FPN
    resolution.
    """

    def __init__(self, scales: tuple[float, ...] = (0.05, 0.15, 0.30)) -> None:
        super().__init__()
        if not scales or any(scale <= 0 or scale > 0.5 for scale in scales):
            raise ValueError("relation scales must be in (0, 0.5]")
        self.scales = tuple(float(scale) for scale in scales)

    def forward(self, context: Tensor,
                valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        if context.ndim != 3:
            raise ValueError("context must have shape [B,H,W]")
        valid = (torch.ones_like(context, dtype=torch.bool)
                 if valid_mask is None else valid_mask.to(dtype=torch.bool))
        if valid.shape != context.shape:
            raise ValueError("valid_mask must match context")
        batch_fields: dict[str, list[Tensor]] = {
            name: [] for name in RELATION_TYPES}
        full_height, full_width = context.shape[-2:]
        for batch_index in range(context.shape[0]):
            row_valid = valid[batch_index].any(dim=-1)
            column_valid = valid[batch_index].any(dim=-2)
            valid_height = int(row_valid.sum())
            valid_width = int(column_valid.sum())
            if valid_height == 0 or valid_width == 0:
                for name in RELATION_TYPES:
                    batch_fields[name].append(context.new_zeros(
                        (1, len(self.scales), full_height, full_width)))
                continue
            local_context = context[
                batch_index:batch_index + 1, :valid_height, :valid_width]
            local_valid = valid[
                batch_index:batch_index + 1, :valid_height, :valid_width]
            local_fields = self._compute_fields(local_context, local_valid)
            for name, field in local_fields.items():
                batch_fields[name].append(F.pad(
                    field, (0, full_width - valid_width,
                            0, full_height - valid_height)))
        return {
            name: torch.cat(values, dim=0).masked_fill(~valid[:, None], 0.0)
            for name, values in batch_fields.items()
        }

    def _compute_fields(self, context: Tensor,
                        valid: Tensor) -> dict[str, Tensor]:
        probability = _masked_spatial_probability(context, valid)
        fields: dict[str, list[Tensor]] = {name: [] for name in RELATION_TYPES}
        overlap = _gaussian_blur(probability, 0.03)
        for scale in self.scales:
            vertical = _gaussian_blur_axis(probability, scale, axis=-2)
            horizontal = _gaussian_blur_axis(probability, scale, axis=-1)
            near = _gaussian_blur(probability, scale)
            broad = _gaussian_blur(probability, min(scale * 2.0, 0.5))
            ring = (broad - near).clamp_min(0.0)
            ring = ring / ring.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            fields["null"].append(torch.zeros_like(probability))
            fields["left"].append(_mass_strictly_after(vertical, dim=-1))
            fields["right"].append(_mass_strictly_before(vertical, dim=-1))
            fields["above"].append(_mass_strictly_after(horizontal, dim=-2))
            fields["below"].append(_mass_strictly_before(horizontal, dim=-2))
            fields["near"].append(near)
            fields["far"].append(ring)
            fields["overlap"].append(overlap)
        return {name: torch.stack(values, dim=1) for name, values in fields.items()}


def _masked_spatial_probability(values: Tensor, valid: Tensor) -> Tensor:
    flat = values.float().flatten(1).masked_fill(~valid.flatten(1), float("-inf"))
    has_valid = valid.flatten(1).any(dim=1)
    safe = torch.where(has_valid[:, None], flat, torch.zeros_like(flat))
    probability = safe.softmax(dim=1)
    probability = probability.masked_fill(~valid.flatten(1), 0.0)
    probability = torch.where(has_valid[:, None], probability,
                              torch.zeros_like(probability))
    return probability.reshape_as(values).to(dtype=values.dtype)


def _gaussian_blur(values: Tensor, scale: float) -> Tensor:
    return _gaussian_blur_axis(
        _gaussian_blur_axis(values, scale, axis=-1), scale, axis=-2)


def _gaussian_blur_axis(values: Tensor, scale: float, axis: int) -> Tensor:
    extent = values.shape[axis]
    sigma = max(scale * extent, 0.5)
    radius = min(max(int(math.ceil(3.0 * sigma)), 1), max(extent - 1, 1))
    coordinates = torch.arange(
        -radius, radius + 1, dtype=torch.float32, device=values.device)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.max().clamp_min(1e-8)
    data = values[:, None].float()
    if axis == -1:
        weight = kernel.view(1, 1, 1, -1)
        padding = (radius, radius, 0, 0)
    elif axis == -2:
        weight = kernel.view(1, 1, -1, 1)
        padding = (0, 0, radius, radius)
    else:
        raise ValueError("axis must be a spatial axis")
    blurred = F.conv2d(F.pad(data, padding, mode="constant", value=0.0), weight)
    return blurred[:, 0].to(dtype=values.dtype)


def _mass_strictly_before(values: Tensor, dim: int) -> Tensor:
    return (values.cumsum(dim=dim) - values).clamp_min(0.0)


def _mass_strictly_after(values: Tensor, dim: int) -> Tensor:
    reversed_values = values.flip((dim,))
    return (reversed_values.cumsum(dim=dim) - reversed_values).flip((dim,)).clamp_min(0.0)
