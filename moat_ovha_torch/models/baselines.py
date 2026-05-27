from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.data.episodes import MetaOperatorBatch
from moat_ovha_torch.models.ovha import OVHAMetaOperator, OVHAOutput
from moat_ovha_torch.models.router import primitive_entropy


def build_model(name: str, d_model: int = 64, memory_tokens: int = 4, top_k: int | None = None) -> nn.Module:
    if name == "ovha_full":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, top_k=top_k)
    if name == "transformer_only" or name == "ovha_vector_value_only":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, primitive_names=("vector_value",))
    if name == "perceiver_io_style":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, primitive_names=("vector_value", "mlp_expert"))
    if name == "icon_style":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, primitive_names=("mlp_expert",))
    if name == "spectral_only" or name == "ovha_single_primitive_spectral":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, primitive_names=("spectral",))
    if name == "separable_only" or name == "ovha_single_primitive_separable":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, primitive_names=("separable",))
    if name == "local_only" or name == "ovha_single_primitive_local":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, primitive_names=("local",))
    if name == "simple_stack":
        return SimpleStackModel(d_model=d_model, memory_tokens=memory_tokens)
    if name == "mlp_expert_moe":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, primitive_names=("mlp_expert", "mlp_expert"))
    if name == "ovha_no_memory":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, use_memory=False)
    if name == "ovha_no_hyper_adapter":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, use_hyper_adapter=False)
    if name == "ovha_random_router":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, random_router=True)
    if name == "ovha_no_query_router":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, query_conditioned_router=False)
    if name == "ovha_no_query_adapter":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, query_conditioned_adapter=False)
    if name == "oracle_metadata_upper_bound":
        return OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, top_k=1)
    raise ValueError(f"unknown phase1.5 model: {name}")


class SimpleStackModel(nn.Module):
    """Primitive stack without query/context-conditioned operator-valued aggregation."""

    def __init__(self, d_model: int = 64, memory_tokens: int = 4):
        super().__init__()
        self.core = OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            primitive_names=("spectral", "separable", "local"),
            use_hyper_adapter=False,
            query_conditioned_router=False,
        )

    def forward(self, batch: MetaOperatorBatch) -> OVHAOutput:
        output = self.core(batch)
        weights = torch.full_like(output.primitive_weights, 1.0 / output.primitive_weights.shape[-1])
        y_hat = (weights.unsqueeze(-1) * output.diagnostics["per_primitive_outputs"]).sum(dim=-2)
        diagnostics = dict(output.diagnostics)
        diagnostics["primitive_entropy"] = primitive_entropy(weights)
        return OVHAOutput(y_hat=y_hat, primitive_weights=weights, diagnostics=diagnostics)
