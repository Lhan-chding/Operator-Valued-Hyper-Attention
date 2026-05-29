from __future__ import annotations

import math

import torch
from torch import nn

from moat_ovha_torch.models.coordinate_features import coordinate_scalar
from moat_ovha_torch.models.primitives.base import PrimitiveParams, apply_film, expand_grid


class SpectralIntegralPrimitive(nn.Module):
    name = "spectral"

    def __init__(self, modes: int = 4):
        super().__init__()
        self.modes = modes
        self.mode_weights = nn.Parameter(torch.ones(modes) / modes)

    def forward(
        self,
        target_u: torch.Tensor,
        support_grid: torch.Tensor,
        target_q: torch.Tensor,
        params: PrimitiveParams | None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = expand_grid(support_grid, target_u.shape[0])
        q = coordinate_scalar(target_q).unsqueeze(-2)
        s = coordinate_scalar(grid).unsqueeze(1)
        outputs = []
        for index in range(1, self.modes + 1):
            frequency = index
            phase = 0.0
            if params is not None and params.spectral_frequency is not None:
                frequency = params.spectral_frequency.unsqueeze(-2) * index
            if params is not None and params.spectral_phase is not None:
                phase = params.spectral_phase.unsqueeze(-2)
            kernel = torch.cos(2.0 * math.pi * frequency * (q - s) + phase)
            outputs.append((kernel * target_u.unsqueeze(1)).mean(dim=-2))
        stacked = torch.stack(outputs, dim=-1)
        if params is not None and params.spectral_mode_logits is not None:
            weights = torch.softmax(params.spectral_mode_logits[..., : self.modes], dim=-1).unsqueeze(-2)
        elif params is not None and params.kernel_params and "mode_logits" in params.kernel_params:
            weights = torch.softmax(params.kernel_params["mode_logits"][..., : self.modes], dim=-1).unsqueeze(-2)
        else:
            weights = torch.softmax(self.mode_weights, dim=0)
        value = (stacked * weights).sum(dim=-1)
        return apply_film(value, params)
