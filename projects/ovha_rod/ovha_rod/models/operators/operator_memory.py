from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .tensor_validation import tensor_value_checks_enabled


@dataclass(frozen=True)
class OperatorMemoryState:
    """Immutable per-query state carried between decoder layers."""

    value: Tensor
    step: int

    def __post_init__(self) -> None:
        if self.value.ndim != 3:
            raise ValueError("memory value must have shape [B,Q,D]")
        if not self.value.is_floating_point():
            raise ValueError("memory value must be floating point")
        if (tensor_value_checks_enabled(self.value)
                and not torch.isfinite(self.value).all()):
            raise ValueError("memory value must contain only finite values")
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError("memory step must be a non-negative integer")


class OperatorMemory(nn.Module):
    """Masked recurrent state without in-place mutation or implicit detach."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = int(d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.state_norm = nn.LayerNorm(d_model)
        self.candidate = nn.Linear(2 * d_model, d_model)
        self.update_gate = nn.Linear(2 * d_model, d_model)

    def initialize(
        self,
        batch_size: int,
        query_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> OperatorMemoryState:
        if batch_size <= 0 or query_count <= 0:
            raise ValueError("batch_size and query_count must be positive")
        if not dtype.is_floating_point:
            raise ValueError("memory dtype must be floating point")
        return OperatorMemoryState(
            value=torch.zeros(
                batch_size, query_count, self.d_model,
                device=device, dtype=dtype),
            step=0,
        )

    def initialize_like(self, query: Tensor) -> OperatorMemoryState:
        self._validate_query(query)
        return self.initialize(
            batch_size=query.shape[0],
            query_count=query.shape[1],
            device=query.device,
            dtype=query.dtype,
        )

    def forward(
        self,
        state: OperatorMemoryState,
        query: Tensor,
        valid: Tensor,
    ) -> OperatorMemoryState:
        self._validate_query(query)
        if state.value.shape != query.shape:
            raise ValueError("query must match the existing memory state")
        if state.value.device != query.device or state.value.dtype != query.dtype:
            raise ValueError("query and memory state must share device and dtype")
        if valid.shape != query.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,Q] tensor")
        if valid.device != query.device:
            raise ValueError("valid must share the query device")

        features = torch.cat(
            (self.input_norm(query), self.state_norm(state.value)), dim=-1)
        candidate = torch.tanh(self.candidate(features))
        gate = torch.sigmoid(self.update_gate(features))
        proposal = state.value + gate * (candidate - state.value)
        updated = torch.where(valid[..., None], proposal, state.value)
        return OperatorMemoryState(value=updated, step=state.step + 1)

    def _validate_query(self, query: Tensor) -> None:
        if query.ndim != 3 or query.shape[-1] != self.d_model:
            raise ValueError("query must have shape [B,Q,D]")
        if not query.is_floating_point():
            raise ValueError("query must be floating point")
        if (tensor_value_checks_enabled(query)
                and not torch.isfinite(query).all()):
            raise ValueError("query must contain only finite values")
