from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .tensor_validation import tensor_value_checks_enabled


@dataclass(frozen=True)
class HyperAdapterResult:
    """Low-rank, per-query modulation for each operator residual."""

    scale: Tensor
    shift: Tensor
    channel_delta: Tensor


class LowRankHyperAdapter(nn.Module):
    """Generate bounded operator modulation from query and memory state."""

    def __init__(self, d_model: int, operator_count: int, rank: int) -> None:
        super().__init__()
        if d_model <= 0 or operator_count <= 0:
            raise ValueError("adapter dimensions must be positive")
        if rank <= 0 or rank > d_model:
            raise ValueError("rank must be in [1, d_model]")
        self.d_model = int(d_model)
        self.operator_count = int(operator_count)
        self.rank = int(rank)
        self.input_norm = nn.LayerNorm(2 * d_model)
        self.coefficients = nn.Linear(
            2 * d_model, operator_count * rank)
        self.scale_basis = nn.Parameter(
            torch.zeros(operator_count, rank, d_model))
        self.shift_basis = nn.Parameter(
            torch.zeros(operator_count, rank, d_model))
        self.channel_basis = nn.Parameter(
            torch.zeros(operator_count, rank, 3))

    def forward(
        self, query: Tensor, memory: Tensor, valid: Tensor
    ) -> HyperAdapterResult:
        self._validate_inputs(query, memory, valid)
        features = self.input_norm(torch.cat((query, memory), dim=-1))
        coefficients = self.coefficients(features).view(
            query.shape[0], query.shape[1], self.operator_count, self.rank)
        coefficients = torch.tanh(coefficients)
        scale = torch.einsum(
            "bqor,ord->bqod", coefficients, self.scale_basis)
        shift = torch.einsum(
            "bqor,ord->bqod", coefficients, self.shift_basis)
        channel_delta = torch.einsum(
            "bqor,orc->bqoc", coefficients, self.channel_basis)
        mask = valid[..., None, None]
        return HyperAdapterResult(
            scale=torch.where(mask, torch.tanh(scale), torch.zeros_like(scale)),
            shift=torch.where(mask, torch.tanh(shift), torch.zeros_like(shift)),
            channel_delta=torch.where(
                mask, torch.tanh(channel_delta), torch.zeros_like(channel_delta)),
        )

    def _validate_inputs(
        self, query: Tensor, memory: Tensor, valid: Tensor
    ) -> None:
        if query.ndim != 3 or query.shape[-1] != self.d_model:
            raise ValueError("query must have shape [B,Q,D]")
        if memory.shape != query.shape:
            raise ValueError("memory must match query shape")
        if valid.shape != query.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,Q] tensor")
        if not query.is_floating_point() or not memory.is_floating_point():
            raise ValueError("query and memory must be floating point")
        if memory.device != query.device or memory.dtype != query.dtype:
            raise ValueError("query and memory must share device and dtype")
        if valid.device != query.device:
            raise ValueError("valid must share the query device")
        if (tensor_value_checks_enabled(query) and (
                not torch.isfinite(query).all()
                or not torch.isfinite(memory).all())):
            raise ValueError("adapter inputs must contain only finite values")
