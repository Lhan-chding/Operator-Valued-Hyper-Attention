from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import Tensor


def memory_valid_mask(
        memory: Tensor, memory_mask: Tensor | None) -> Tensor:
    """Normalize MMDetection's optional padding mask to valid-token form."""
    if memory.ndim < 2:
        raise ValueError("memory must have batch and token dimensions")
    expected = memory.shape[:2]
    if memory_mask is None:
        return torch.ones(expected, dtype=torch.bool, device=memory.device)
    if memory_mask.shape != expected:
        raise ValueError("memory_mask must have shape [B,N]")
    return ~memory_mask.to(device=memory.device, dtype=torch.bool)


@dataclass(frozen=True)
class SeedResult:
    seed_bias: Tensor
    raw_logits: Tensor
    valid: Tensor
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed_bias.ndim != 2:
            raise ValueError("seed_bias must have shape [B,N]")
        if self.raw_logits.shape != self.seed_bias.shape:
            raise ValueError("raw_logits must match seed_bias")
        if self.valid.shape != self.seed_bias.shape or self.valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,N] tensor")
