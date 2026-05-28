from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class PrimitiveParams:
    scale: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    spectral_frequency: Optional[torch.Tensor] = None
    spectral_phase: Optional[torch.Tensor] = None
    spectral_mode_logits: Optional[torch.Tensor] = None
    local_lengthscale: Optional[torch.Tensor] = None
    local_shift: Optional[torch.Tensor] = None
    separable_rank_logits: Optional[torch.Tensor] = None
    low_rank_a: Optional[torch.Tensor] = None
    low_rank_b: Optional[torch.Tensor] = None
    kernel_params: Optional[dict[str, torch.Tensor]] = None
    raw: Optional[dict[str, torch.Tensor]] = None
    scope: Optional[dict[str, str]] = None


def apply_film(value: torch.Tensor, params: PrimitiveParams | None) -> torch.Tensor:
    if params is None:
        return value
    if params.scale is not None:
        value = value * params.scale
    if params.bias is not None:
        value = value + params.bias
    return value


def expand_grid(support_grid: torch.Tensor, batch_size: int) -> torch.Tensor:
    if support_grid.shape[0] == 1:
        return support_grid.expand(batch_size, -1, -1)
    return support_grid
