from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import Tensor


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
