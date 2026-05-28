from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.hyper_adapter import HyperAdapter
from moat_ovha_torch.models.primitives.base import PrimitiveParams
from moat_ovha_torch.models.router import PrimitiveRouter, RouterOutput


class JointRouterAdapter(nn.Module):
    """Couple routing posterior and adapter parameter inference on shared primitive memory."""

    def __init__(
        self,
        primitive_names: tuple[str, ...],
        d_model: int = 64,
        top_k: int | None = None,
        query_conditioned_router: bool = True,
        query_conditioned_adapter: bool = True,
        random_router: bool = False,
        controlled_generator_variant: str = "model_aligned",
    ):
        super().__init__()
        self.router = PrimitiveRouter(primitive_names, d_model, top_k, query_conditioned_router, random_router)
        self.hyper_adapter = HyperAdapter(
            primitive_names,
            d_model=d_model,
            query_conditioned=query_conditioned_adapter,
            controlled_generator_variant=controlled_generator_variant,
        )

    def forward(
        self,
        memory_bank: torch.Tensor | dict[str, torch.Tensor],
        target_q: torch.Tensor,
    ) -> tuple[RouterOutput, dict[str, PrimitiveParams]]:
        router_out = self.router(memory_bank, target_q)
        params = self.hyper_adapter(memory_bank, target_q, router_out=router_out)
        return router_out, params
