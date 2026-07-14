from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .tensor_validation import tensor_value_checks_enabled
import torch.nn.functional as F

from .decoder_contracts import DecoderOperatorResidual


_GEOMETRY_DIM = 8


class QuerySpatialRelationOperator(nn.Module):
    """Contrast relation-selected queries with valid distractor queries.

    The operator consumes decoder predictions only. Pairwise evidence is built
    from query features and predicted ``cxcywh`` boxes; annotations and target
    boxes are intentionally absent from the public boundary.
    """

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = int(d_model)
        self.query_projection = nn.Linear(self.d_model, self.d_model)
        self.key_projection = nn.Linear(self.d_model, self.d_model)
        self.value_projection = nn.Linear(self.d_model, self.d_model)
        self.role_projection = nn.Linear(self.d_model, self.d_model)
        self.geometry_parameter_head = nn.Linear(
            self.d_model, _GEOMETRY_DIM)
        self.fusion = nn.Sequential(
            nn.Linear(3 * self.d_model + _GEOMETRY_DIM + 1, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.query_head = nn.Linear(self.d_model, self.d_model)
        self.box_head = nn.Linear(self.d_model, 4)
        self.score_head = nn.Linear(self.d_model, 1)
        self.gate_head = nn.Linear(self.d_model, 3)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(
        self,
        query: Tensor,
        boxes: Tensor,
        relation_role: Tensor,
        valid: Tensor,
    ) -> DecoderOperatorResidual:
        self._validate_inputs(query, boxes, relation_role, valid)
        valid_feature = valid[..., None]
        safe_query = query.masked_fill(~valid_feature, 0.0)
        safe_boxes = boxes.masked_fill(~valid_feature, 0.0)

        projected_role = self.role_projection(relation_role)
        relation_query = F.normalize(
            self.query_projection(safe_query) + projected_role[:, None],
            dim=-1,
        )
        key = F.normalize(self.key_projection(safe_query), dim=-1)
        value = self.value_projection(safe_query)

        geometry = _pairwise_box_geometry(safe_boxes)
        geometry_parameters = self.geometry_parameter_head(
            relation_role).tanh()
        semantic_logits = torch.matmul(relation_query, key.transpose(-1, -2))
        geometry_logits = torch.einsum(
            "bqkg,bg->bqk", geometry, geometry_parameters)
        pair_logits = semantic_logits + geometry_logits / math.sqrt(
            _GEOMETRY_DIM)

        pair_mask = _directed_pair_mask(valid)
        relation_weights = _masked_softmax(pair_logits, pair_mask)
        distractor_weights = _masked_uniform(pair_mask, query.dtype)
        contrast_weights = relation_weights - distractor_weights

        relation_context = torch.matmul(relation_weights, value)
        distractor_context = torch.matmul(distractor_weights, value)
        contrast_context = relation_context - distractor_context
        geometry_contrast = torch.einsum(
            "bqk,bqkg->bqg", contrast_weights, geometry)
        contrast_score = (contrast_weights * pair_logits).sum(-1)

        expanded_role = projected_role[:, None].expand_as(safe_query)
        hidden = self.fusion(torch.cat(
            (
                safe_query,
                contrast_context,
                expanded_role,
                geometry_contrast,
                contrast_score[..., None],
            ),
            dim=-1,
        ))
        has_distractor = pair_mask.any(dim=-1)
        active = valid & has_distractor
        active_feature = active[..., None].to(dtype=query.dtype)
        active_scalar = active.to(dtype=query.dtype)

        query_delta = self.query_head(hidden) * active_feature
        box_delta = self.box_head(hidden) * active_feature
        score_delta = self.score_head(hidden).squeeze(-1) * active_scalar
        gate_logits = self.gate_head(hidden) * active_feature

        diagnostics = {
            "qsro_attention_entropy": _attention_entropy(
                relation_weights, active),
            "qsro_contrast_abs_mean": _masked_mean(
                contrast_score.abs(), active),
            "qsro_pair_density": pair_mask.to(dtype=query.dtype).mean(),
            "qsro_gate_abs_mean": _masked_mean(
                gate_logits.abs().mean(-1), valid),
        }
        return DecoderOperatorResidual(
            query_delta=query_delta,
            box_delta=box_delta,
            score_delta=score_delta,
            gate_logits=gate_logits,
            valid=valid,
            diagnostics=diagnostics,
        )

    def _validate_inputs(
        self,
        query: Tensor,
        boxes: Tensor,
        relation_role: Tensor,
        valid: Tensor,
    ) -> None:
        if query.ndim != 3 or query.shape[-1] != self.d_model:
            raise ValueError(
                f"query must have shape [B,Q,{self.d_model}]")
        if query.shape[0] == 0 or query.shape[1] == 0:
            raise ValueError("query batch and query dimensions must be non-empty")
        if boxes.shape != (*query.shape[:2], 4):
            raise ValueError("boxes must have shape [B,Q,4]")
        if relation_role.shape != (query.shape[0], self.d_model):
            raise ValueError("relation_role must have shape [B,D]")
        if valid.shape != query.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,Q] tensor")
        values = (query, boxes, relation_role)
        if not all(value.is_floating_point() for value in values):
            raise ValueError("query, boxes, and relation_role must be floating point")
        if any(value.device != query.device for value in values[1:]):
            raise ValueError("operator inputs must share a device")
        if valid.device != query.device:
            raise ValueError("valid and query must share a device")
        if any(value.dtype != query.dtype for value in values[1:]):
            raise ValueError("operator inputs must share a dtype")
        if tensor_value_checks_enabled(query):
            if not all(torch.isfinite(value).all() for value in values):
                raise ValueError(
                    "operator inputs must contain only finite values")
            if not bool(((boxes >= 0.0) & (boxes <= 1.0)).all()):
                raise ValueError(
                    "boxes must contain normalized cxcywh values")


QSRO = QuerySpatialRelationOperator


def _directed_pair_mask(valid: Tensor) -> Tensor:
    query_count = valid.shape[1]
    off_diagonal = ~torch.eye(
        query_count, dtype=torch.bool, device=valid.device)
    return valid[:, :, None] & valid[:, None, :] & off_diagonal[None]


def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    minimum = torch.finfo(logits.dtype).min
    safe_logits = logits.masked_fill(~mask, minimum)
    row_max = safe_logits.max(dim=-1, keepdim=True).values
    numerator = (safe_logits - row_max).exp() * mask.to(dtype=logits.dtype)
    return numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(logits.dtype).eps)


def _masked_uniform(mask: Tensor, dtype: torch.dtype) -> Tensor:
    values = mask.to(dtype=dtype)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1.0)


def _pairwise_box_geometry(boxes: Tensor) -> Tensor:
    epsilon = torch.finfo(boxes.dtype).eps
    centers = boxes[..., :2]
    sizes = boxes[..., 2:].clamp_min(epsilon)
    target_centers = centers[:, :, None]
    other_centers = centers[:, None, :]
    target_sizes = sizes[:, :, None]
    other_sizes = sizes[:, None, :]

    center_delta = (other_centers - target_centers) / target_sizes
    absolute_delta = (other_centers - target_centers).abs()
    log_scale = (other_sizes / target_sizes).log()
    log_area_ratio = (
        other_sizes.prod(-1) / target_sizes.prod(-1)).log()[..., None]
    distance = center_delta.square().sum(-1, keepdim=True).clamp_min(
        epsilon).sqrt()
    return torch.cat(
        (center_delta, absolute_delta, log_scale, log_area_ratio, distance),
        dim=-1,
    )


def _attention_entropy(weights: Tensor, active: Tensor) -> Tensor:
    epsilon = torch.finfo(weights.dtype).eps
    entropy = -(weights * weights.clamp_min(epsilon).log()).sum(-1)
    return _masked_mean(entropy, active)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    masked = values.masked_fill(~mask, 0.0)
    return masked.sum() / mask.sum().clamp_min(1).to(dtype=values.dtype)
