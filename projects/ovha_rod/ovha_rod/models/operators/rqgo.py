from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..role_encoder import RoleState
from .base import SeedResult, memory_valid_mask
from .relation_fields import RELATION_TYPES, RelationFieldBank


RQGOResult = SeedResult


class RQGO(nn.Module):
    """Relation-Conditioned Query Genesis Operator for encoder proposals."""

    def __init__(self, d_model: int = 256, seed_bias_cap: float = 2.0,
                 relation_scales: tuple[float, ...] = (0.05, 0.15, 0.30)) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.seed_bias_cap = float(seed_bias_cap)
        self.relation_fields = RelationFieldBank(relation_scales)
        self.visual_projections = nn.ModuleList(
            nn.Linear(d_model, d_model) for _ in range(3))
        self.role_projections = nn.ModuleList(
            nn.Linear(d_model, d_model) for _ in range(3))
        self.raw_temperatures = nn.Parameter(torch.zeros(3))
        relation_count = len(RELATION_TYPES)
        scale_count = len(relation_scales)
        self.relation_parameter_head = nn.Linear(
            d_model, relation_count + relation_count * scale_count)
        self.seed_head = nn.Sequential(
            nn.Linear(d_model + 3, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.zeros_(self.seed_head[-1].weight)
        nn.init.zeros_(self.seed_head[-1].bias)

    def forward(self, memory: Tensor, proposal_boxes: Tensor,
                roles: RoleState, spatial_shapes: Tensor,
                memory_mask: Tensor | None = None) -> RQGOResult:
        self._validate_inputs(memory, proposal_boxes, roles, spatial_shapes)
        valid = memory_valid_mask(memory, memory_mask)
        semantic = [
            self._semantic_field(memory, roles.vectors[:, index], index)
            for index in range(3)
        ]
        relation = self._relation_field(
            semantic[2], roles.vectors[:, 3], spatial_shapes, valid)
        features = torch.cat(
            (memory, semantic[0][..., None], semantic[1][..., None],
             relation[..., None]), dim=-1)
        raw = self.seed_head(features).squeeze(-1)
        bias = self.seed_bias_cap * raw.tanh()
        raw = raw.masked_fill(~valid, 0.0)
        bias = bias.masked_fill(~valid, 0.0)
        diagnostics = {
            "entity_mean": _valid_mean(semantic[0], valid),
            "attribute_mean": _valid_mean(semantic[1], valid),
            "relation_mean": _valid_mean(relation, valid),
            "seed_bias_abs_mean": _valid_mean(bias.abs(), valid),
        }
        return RQGOResult(bias, raw, valid, diagnostics)

    def _semantic_field(self, memory: Tensor, role: Tensor, index: int) -> Tensor:
        visual = F.normalize(self.visual_projections[index](memory), dim=-1)
        language = F.normalize(self.role_projections[index](role), dim=-1)
        temperature = (F.softplus(self.raw_temperatures[index]) + 0.03).clamp(max=2.0)
        return (visual * language[:, None]).sum(-1) / temperature

    def _relation_field(self, context_logits: Tensor, relation_role: Tensor,
                        spatial_shapes: Tensor, valid: Tensor) -> Tensor:
        batch, tokens = context_logits.shape
        relation_count = len(RELATION_TYPES)
        scale_count = len(self.relation_fields.scales)
        parameters = self.relation_parameter_head(relation_role)
        relation_weights = parameters[:, :relation_count].softmax(-1)
        scale_weights = parameters[:, relation_count:].reshape(
            batch, relation_count, scale_count).softmax(-1)
        mixed_levels: list[Tensor] = []
        offset = 0
        for height_tensor, width_tensor in spatial_shapes.tolist():
            height, width = int(height_tensor), int(width_tensor)
            count = height * width
            if offset + count > tokens:
                raise ValueError("spatial_shapes exceeds flattened memory length")
            level_context = context_logits[:, offset:offset + count].reshape(
                batch, height, width)
            level_valid = valid[:, offset:offset + count].reshape(
                batch, height, width)
            fields = self.relation_fields(level_context, level_valid)
            bank = torch.stack([fields[name] for name in RELATION_TYPES], dim=1)
            weights = relation_weights[:, :, None] * scale_weights
            mixed = (bank * weights[:, :, :, None, None]).sum((1, 2))
            mixed_levels.append(mixed.flatten(1))
            offset += count
        if offset != tokens:
            raise ValueError("spatial_shapes must exactly cover flattened memory")
        return torch.cat(mixed_levels, dim=1).masked_fill(~valid, 0.0)

    def _validate_inputs(self, memory: Tensor, boxes: Tensor, roles: RoleState,
                         spatial_shapes: Tensor) -> None:
        if memory.ndim != 3 or memory.shape[-1] != self.d_model:
            raise ValueError(f"memory must have shape [B,N,{self.d_model}]")
        if boxes.shape != (*memory.shape[:2], 4):
            raise ValueError("proposal_boxes must have shape [B,N,4]")
        if roles.vectors.shape != (memory.shape[0], 4, self.d_model):
            raise ValueError("role vectors must have shape [B,4,D]")
        if spatial_shapes.ndim != 2 or spatial_shapes.shape[1] != 2:
            raise ValueError("spatial_shapes must have shape [S,2]")


def _valid_mean(values: Tensor, valid: Tensor) -> Tensor:
    return values.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1)
