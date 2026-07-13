from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OperatorRouterResult:
    """Per-query mixture weights for the enabled decoder operators."""

    weights: Tensor
    logits: Tensor


class OperatorRouter(nn.Module):
    """Route each valid matching query across inference-time operators."""

    def __init__(
        self,
        d_model: int,
        operator_count: int,
        num_layers: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(d_model, operator_count, num_layers, hidden_dim) <= 0:
            raise ValueError("router dimensions must be positive")
        self.d_model = int(d_model)
        self.operator_count = int(operator_count)
        self.num_layers = int(num_layers)
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.layer_embedding = nn.Embedding(num_layers, d_model)
        self.network = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, operator_count),
        )

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        layer_index: int,
        valid: Tensor,
        reliability_log_prior: Optional[Tensor] = None,
        operator_available: Optional[Tensor] = None,
    ) -> OperatorRouterResult:
        self._validate_inputs(
            query,
            memory,
            layer_index,
            valid,
            reliability_log_prior,
            operator_available,
        )
        layer = self.layer_embedding.weight[layer_index].view(1, 1, -1)
        layer = layer.expand(query.shape[0], query.shape[1], -1)
        features = torch.cat(
            (self.query_norm(query), self.memory_norm(memory), layer), dim=-1)
        logits = self.network(features)
        if reliability_log_prior is not None:
            logits = logits + reliability_log_prior

        available = self._expanded_availability(logits, operator_available)
        valid_available = available[valid]
        if valid_available.numel() and not valid_available.any(dim=-1).all():
            raise ValueError("each valid query must have an available operator")

        safe_available = torch.where(
            valid[..., None], available, torch.ones_like(available))
        masked_logits = logits.masked_fill(~safe_available, -torch.inf)
        weights = torch.softmax(masked_logits, dim=-1)
        weights = torch.where(valid[..., None], weights, torch.zeros_like(weights))
        visible_logits = torch.where(
            valid[..., None] & available,
            logits,
            torch.zeros_like(logits),
        )
        return OperatorRouterResult(weights=weights, logits=visible_logits)

    def _validate_inputs(
        self,
        query: Tensor,
        memory: Tensor,
        layer_index: int,
        valid: Tensor,
        reliability_log_prior: Optional[Tensor],
        operator_available: Optional[Tensor],
    ) -> None:
        if query.ndim != 3 or query.shape[-1] != self.d_model:
            raise ValueError("query must have shape [B,Q,D]")
        if memory.shape != query.shape:
            raise ValueError("memory must match query shape")
        if valid.shape != query.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,Q] tensor")
        if not isinstance(layer_index, int) or not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index is outside the configured decoder")
        tensors = (query, memory)
        if any(not value.is_floating_point() for value in tensors):
            raise ValueError("query and memory must be floating point")
        if memory.device != query.device or memory.dtype != query.dtype:
            raise ValueError("query and memory must share device and dtype")
        if valid.device != query.device:
            raise ValueError("valid must share the query device")
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("router inputs must contain only finite values")
        expected = (*query.shape[:2], self.operator_count)
        if reliability_log_prior is not None:
            if reliability_log_prior.shape != expected:
                raise ValueError("reliability_log_prior has the wrong shape")
            if (reliability_log_prior.device != query.device
                    or reliability_log_prior.dtype != query.dtype):
                raise ValueError("reliability_log_prior must match query")
            if not torch.isfinite(reliability_log_prior).all():
                raise ValueError("reliability_log_prior must be finite")
        if operator_available is not None:
            allowed_shapes = ((self.operator_count,), expected)
            if operator_available.shape not in allowed_shapes:
                raise ValueError("operator_available has the wrong shape")
            if operator_available.dtype != torch.bool:
                raise ValueError("operator_available must be boolean")
            if operator_available.device != query.device:
                raise ValueError("operator_available must share the query device")

    @staticmethod
    def _expanded_availability(
        logits: Tensor, operator_available: Optional[Tensor]
    ) -> Tensor:
        if operator_available is None:
            return torch.ones_like(logits, dtype=torch.bool)
        if operator_available.ndim == 1:
            return operator_available.view(1, 1, -1).expand_as(logits)
        return operator_available
