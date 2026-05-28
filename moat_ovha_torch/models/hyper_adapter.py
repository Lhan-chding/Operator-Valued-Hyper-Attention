from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.primitives.base import PrimitiveParams


class HyperAdapter(nn.Module):
    """Primitive-specific FiLM and operator parameters from memory + query."""

    def __init__(
        self,
        primitive_names: tuple[str, ...],
        d_model: int = 64,
        query_conditioned: bool = True,
        spectral_modes: int = 4,
        separable_rank: int = 4,
    ):
        super().__init__()
        self.primitive_names = primitive_names
        self.query_conditioned = query_conditioned
        self.spectral_modes = spectral_modes
        self.separable_rank = separable_rank
        input_dim = d_model + (1 if query_conditioned else 0)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Linear(d_model, self._output_dim(name)))
                for name in primitive_names
            }
        )
        self._initialize_identity_defaults()

    def forward(self, memory: torch.Tensor, target_q: torch.Tensor) -> dict[str, PrimitiveParams]:
        pooled = memory.mean(dim=1)
        if self.query_conditioned:
            features = torch.cat([pooled.unsqueeze(1).expand(-1, target_q.shape[1], -1), target_q], dim=-1)
        else:
            features = pooled.unsqueeze(1).expand(-1, target_q.shape[1], -1)
        return {name: self._params_for(name, head(features)) for name, head in self.heads.items()}

    def _output_dim(self, primitive_name: str) -> int:
        if primitive_name == "spectral":
            return 4 + self.spectral_modes
        if primitive_name == "local":
            return 4
        if primitive_name == "separable":
            return 2 + self.separable_rank
        return 3

    def _params_for(self, primitive_name: str, raw: torch.Tensor) -> PrimitiveParams:
        scale = 1.0 + 0.1 * torch.tanh(raw[..., 0:1])
        bias = 0.1 * torch.tanh(raw[..., 1:2])
        if primitive_name == "spectral":
            frequency = 1.0 + 0.25 * torch.tanh(raw[..., 2:3])
            phase = torch.pi * torch.tanh(raw[..., 3:4])
            mode_logits = raw[..., 4 : 4 + self.spectral_modes]
            return PrimitiveParams(
                scale=scale,
                bias=bias,
                spectral_frequency=frequency,
                spectral_phase=phase,
                spectral_mode_logits=mode_logits,
                kernel_params={"frequency": frequency, "phase": phase, "mode_logits": mode_logits},
            )
        if primitive_name == "local":
            lengthscale = -2.0 + 0.5 * torch.tanh(raw[..., 2:3])
            shift = 0.25 * torch.tanh(raw[..., 3:4])
            return PrimitiveParams(
                scale=scale,
                bias=bias,
                local_lengthscale=lengthscale,
                local_shift=shift,
                kernel_params={"lengthscale": lengthscale, "shift": shift},
            )
        if primitive_name == "separable":
            rank_logits = raw[..., 2 : 2 + self.separable_rank]
            return PrimitiveParams(
                scale=scale,
                bias=bias,
                separable_rank_logits=rank_logits,
                kernel_params={"rank_logits": rank_logits},
            )
        kernel = {"lengthscale": raw[..., 2:3]} if raw.shape[-1] > 2 else None
        return PrimitiveParams(scale=scale, bias=bias, kernel_params=kernel)

    def _initialize_identity_defaults(self) -> None:
        for head in self.heads.values():
            final = head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
