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
        route_override: torch.Tensor | None = None,
    ) -> tuple[RouterOutput, dict[str, PrimitiveParams]]:
        router_out = self.router(memory_bank, target_q)
        adapter_router_out = _adapter_router_output(router_out, route_override)
        params = self.hyper_adapter(memory_bank, target_q, router_out=adapter_router_out)
        return router_out, params


def _adapter_router_output(router_out: RouterOutput, route_override: torch.Tensor | None) -> RouterOutput:
    if route_override is None:
        return router_out
    weights = route_override.to(device=router_out.weights.device, dtype=router_out.weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return RouterOutput(
        weights=weights,
        logits=weights.clamp_min(1e-8).log(),
        context_prior_logits=router_out.context_prior_logits,
        query_residual_logits=router_out.query_residual_logits,
    )
