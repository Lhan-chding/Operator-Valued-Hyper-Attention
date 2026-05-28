from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.primitives.base import PrimitiveParams, apply_film, expand_grid


class LocalKernelPrimitive(nn.Module):
    name = "local"

    def __init__(self):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        target_u: torch.Tensor,
        support_grid: torch.Tensor,
        target_q: torch.Tensor,
        params: PrimitiveParams | None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = expand_grid(support_grid, target_u.shape[0])
        lengthscale = torch.nn.functional.softplus(self.log_lengthscale) + 1e-3
        if params is not None and params.local_lengthscale is not None:
            lengthscale = torch.nn.functional.softplus(params.local_lengthscale).unsqueeze(-2) + 1e-3
        elif params is not None and params.kernel_params and "lengthscale" in params.kernel_params:
            lengthscale = torch.nn.functional.softplus(params.kernel_params["lengthscale"]).unsqueeze(-2) + 1e-3
        shift = 0.0
        if params is not None and params.local_shift is not None:
            shift = params.local_shift.unsqueeze(-2)
        elif params is not None and params.kernel_params and "shift" in params.kernel_params:
            shift = params.kernel_params["shift"].unsqueeze(-2)
        kernel = torch.exp(-((target_q.unsqueeze(-2) - grid.unsqueeze(1) - shift) ** 2) / (2.0 * lengthscale**2))
        kernel = kernel / kernel.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        value = (kernel * target_u.unsqueeze(1)).sum(dim=-2)
        return apply_film(value, params)
