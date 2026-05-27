from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from moat_ovha_torch.data.episodes import MetaOperatorBatch
from moat_ovha_torch.models.context_encoder import ContextTokenEncoder
from moat_ovha_torch.models.hyper_adapter import HyperAdapter
from moat_ovha_torch.models.memory import PerceiverMemoryEncoder, PoolingMemoryEncoder
from moat_ovha_torch.models.primitives.registry import make_primitive_registry
from moat_ovha_torch.models.router import PrimitiveRouter, primitive_entropy


@dataclass(frozen=True)
class OVHAOutput:
    y_hat: torch.Tensor
    primitive_weights: torch.Tensor
    diagnostics: dict[str, torch.Tensor | dict[str, torch.Tensor]]


class OVHAMetaOperator(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        memory_tokens: int = 4,
        primitive_names: tuple[str, ...] = ("spectral", "separable", "local"),
        memory_kind: str = "perceiver",
        top_k: int | None = None,
        use_memory: bool = True,
        use_hyper_adapter: bool = True,
        query_conditioned_router: bool = True,
        query_conditioned_adapter: bool = True,
        random_router: bool = False,
    ):
        super().__init__()
        self.primitive_names = primitive_names
        self.use_memory = use_memory
        self.use_hyper_adapter = use_hyper_adapter
        self.context_encoder = ContextTokenEncoder(d_model)
        if memory_kind == "pooling":
            self.memory_encoder = PoolingMemoryEncoder(d_model, memory_tokens)
        else:
            self.memory_encoder = PerceiverMemoryEncoder(d_model, memory_tokens)
        self.no_memory = nn.Parameter(torch.zeros(memory_tokens, d_model))
        self.hyper_adapter = HyperAdapter(primitive_names, d_model, query_conditioned_adapter)
        self.router = PrimitiveRouter(primitive_names, d_model, top_k, query_conditioned_router, random_router)
        self.primitives = make_primitive_registry(primitive_names)

    def forward(self, batch: MetaOperatorBatch) -> OVHAOutput:
        tokens = self.context_encoder(batch)
        memory = self.memory_encoder(tokens, batch.context_mask)
        if not self.use_memory:
            memory = self.no_memory.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        params = self.hyper_adapter(memory, batch.target_q) if self.use_hyper_adapter else {name: None for name in self.primitive_names}
        weights = self.router(memory, batch.target_q)
        primitive_outputs = []
        for name in self.primitive_names:
            primitive_outputs.append(self.primitives[name](batch.target_u, batch.support_grid, batch.target_q, params[name], memory))
        stacked = torch.stack(primitive_outputs, dim=-2)
        y_hat = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        adapter_norms = {}
        for name, param in params.items():
            if param is None or param.scale is None:
                adapter_norms[name] = torch.tensor(0.0, device=y_hat.device)
            else:
                adapter_norms[name] = param.scale.norm()
        diagnostics = {
            "primitive_entropy": primitive_entropy(weights),
            "adapter_norms": adapter_norms,
            "memory_norms": memory.norm(dim=-1).mean(),
            "per_primitive_outputs": stacked.detach(),
        }
        return OVHAOutput(y_hat=y_hat, primitive_weights=weights, diagnostics=diagnostics)
