from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.data.episodes import MetaOperatorBatch
from moat_ovha_torch.models.ovha import OVHAMetaOperator, OVHAOutput
from moat_ovha_torch.models.router import primitive_entropy


def build_model(
    name: str,
    d_model: int = 64,
    memory_tokens: int = 4,
    top_k: int | None = None,
    controlled_generator_variant: str = "model_aligned",
) -> nn.Module:
    if name == "ovha_full":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            top_k=top_k,
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "ovha_vector_value_big":
        return OVHAMetaOperator(
            d_model=max(d_model + d_model // 2, d_model + 1),
            memory_tokens=memory_tokens,
            primitive_names=("vector_value",),
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "transformer_only" or name == "ovha_vector_value_only":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            primitive_names=("vector_value",),
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "perceiver_io_style":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            primitive_names=("vector_value", "mlp_expert"),
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "icon_style":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            primitive_names=("mlp_expert",),
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "spectral_only" or name == "ovha_single_primitive_spectral":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            primitive_names=("spectral",),
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "separable_only" or name == "ovha_single_primitive_separable":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            primitive_names=("separable",),
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "local_only" or name == "ovha_single_primitive_local":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            primitive_names=("local",),
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "simple_stack":
        return SimpleStackModel(d_model=d_model, memory_tokens=memory_tokens)
    if name == "mlp_expert_moe":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            primitive_names=("mlp_expert_0", "mlp_expert_1"),
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "mlp_expert_moe_big_optional":
        return OVHAMetaOperator(
            d_model=max(d_model + d_model // 2, d_model + 1),
            memory_tokens=memory_tokens,
            primitive_names=("mlp_expert_0", "mlp_expert_1"),
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "ovha_no_memory":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            use_memory=False,
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "ovha_learned_global_memory":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            use_memory=False,
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "ovha_zero_memory" or name == "target_only":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            use_memory=False,
            trainable_global_memory=False,
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "shuffled_context_memory" or name == "ovha_shuffled_context_memory":
        return ShuffledContextMemoryModel(d_model=d_model, memory_tokens=memory_tokens, top_k=top_k)
    if name == "ovha_no_hyper_adapter":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            use_hyper_adapter=False,
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "ovha_random_router":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            random_router=True,
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "ovha_no_query_router":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            query_conditioned_router=False,
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "ovha_no_query_adapter":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            query_conditioned_adapter=False,
            controlled_generator_variant=controlled_generator_variant,
        )
    if name == "oracle_metadata_upper_bound":
        return OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            top_k=1,
            controlled_generator_variant=controlled_generator_variant,
        )
    raise ValueError(f"unknown phase1.5 model: {name}")


class SimpleStackModel(nn.Module):
    """Primitive stack without query/context-conditioned operator-valued aggregation."""

    def __init__(self, d_model: int = 64, memory_tokens: int = 4):
        super().__init__()
        self.core = OVHAMetaOperator(
            d_model=d_model,
            memory_tokens=memory_tokens,
            primitive_names=("spectral", "local", "separable"),
            use_hyper_adapter=False,
            query_conditioned_router=False,
        )

    def forward(self, batch: MetaOperatorBatch) -> OVHAOutput:
        output = self.core(batch)
        weights = torch.full_like(output.primitive_weights, 1.0 / output.primitive_weights.shape[-1])
        y_hat = (weights.unsqueeze(-1) * output.diagnostics["per_primitive_outputs_train"]).sum(dim=-2)
        diagnostics = dict(output.diagnostics)
        diagnostics["primitive_entropy"] = primitive_entropy(weights)
        return OVHAOutput(y_hat=y_hat, primitive_weights=weights, diagnostics=diagnostics)


class ShuffledContextMemoryModel(nn.Module):
    """Ablation that breaks context-target pairing while preserving tensor shapes."""

    def __init__(self, d_model: int = 64, memory_tokens: int = 4, top_k: int | None = None):
        super().__init__()
        self.core = OVHAMetaOperator(d_model=d_model, memory_tokens=memory_tokens, top_k=top_k)
        self.primitive_names = self.core.primitive_names

    def forward(self, batch: MetaOperatorBatch) -> OVHAOutput:
        shuffled = _shuffled_context_batch(batch)
        return self.core(shuffled)


def _shuffled_context_batch(batch: MetaOperatorBatch) -> MetaOperatorBatch:
    if batch.context_u.shape[0] > 1:
        index = torch.roll(torch.arange(batch.context_u.shape[0], device=batch.context_u.device), shifts=1)
        return MetaOperatorBatch(
            context_u=batch.context_u.index_select(0, index),
            context_q=batch.context_q.index_select(0, index),
            context_y=batch.context_y.index_select(0, index),
            target_u=batch.target_u,
            target_q=batch.target_q,
            target_y=batch.target_y,
            support_grid=batch.support_grid,
            context_mask=batch.context_mask.index_select(0, index) if batch.context_mask is not None else None,
            target_mask=batch.target_mask,
        )
    return MetaOperatorBatch(
        context_u=batch.context_u.flip(1),
        context_q=batch.context_q.flip(1),
        context_y=batch.context_y.flip(1),
        target_u=batch.target_u,
        target_q=batch.target_q,
        target_y=batch.target_y,
        support_grid=batch.support_grid,
        context_mask=batch.context_mask.flip(1) if batch.context_mask is not None else None,
        target_mask=batch.target_mask,
    )
