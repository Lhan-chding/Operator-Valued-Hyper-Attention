from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from moat_ovha_torch.data.episodes import MetaOperatorBatch
from moat_ovha_torch.models.context_encoder import ContextTokenEncoder
from moat_ovha_torch.models.joint_router_adapter import JointRouterAdapter
from moat_ovha_torch.models.memory import PerceiverMemoryEncoder, PoolingMemoryEncoder, PrimitiveSlotMemoryEncoder
from moat_ovha_torch.models.primitives.base import PrimitiveParams
from moat_ovha_torch.models.primitives.registry import make_primitive_registry
from moat_ovha_torch.models.router import RouterOutput, primitive_entropy


@dataclass(frozen=True)
class OVHAOutput:
    y_hat: torch.Tensor
    primitive_weights: torch.Tensor
    diagnostics: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    adapter_params: dict[str, PrimitiveParams] | None = None
    router_logits: torch.Tensor | None = None
    memory_bank: dict[str, torch.Tensor] | torch.Tensor | None = None
    router_output: RouterOutput | None = None


class OVHAMetaOperator(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        memory_tokens: int = 4,
        primitive_names: tuple[str, ...] = ("spectral", "local", "separable"),
        memory_kind: str = "perceiver",
        top_k: int | None = None,
        use_memory: bool = True,
        trainable_global_memory: bool = True,
        use_hyper_adapter: bool = True,
        query_conditioned_router: bool = True,
        query_conditioned_adapter: bool = True,
        random_router: bool = False,
        controlled_generator_variant: str = "model_aligned",
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
        self.primitive_slot_memory = PrimitiveSlotMemoryEncoder(primitive_names, d_model, memory_tokens)
        no_memory = torch.zeros(memory_tokens, d_model)
        if trainable_global_memory:
            self.no_memory = nn.Parameter(no_memory)
        else:
            self.register_buffer("no_memory", no_memory)
        self.joint_router_adapter = JointRouterAdapter(
            primitive_names,
            d_model=d_model,
            top_k=top_k,
            query_conditioned_router=query_conditioned_router,
            query_conditioned_adapter=query_conditioned_adapter,
            random_router=random_router,
            controlled_generator_variant=controlled_generator_variant,
        )
        self.primitives = make_primitive_registry(primitive_names)

    @property
    def router(self):
        return self.joint_router_adapter.router

    @property
    def hyper_adapter(self):
        return self.joint_router_adapter.hyper_adapter

    def forward(
        self,
        batch: MetaOperatorBatch,
        route_override: torch.Tensor | None = None,
        active_primitive_mask: torch.Tensor | None = None,
    ) -> OVHAOutput:
        tokens, evidence_bank = self.context_encoder.encode_with_evidence(batch)
        memory = self.memory_encoder(tokens, batch.context_mask)
        if not self.use_memory:
            memory = self.no_memory.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        memory_bank = self.primitive_slot_memory(memory)
        router_out, params = self.joint_router_adapter(
            memory_bank,
            batch.target_q,
            route_override=route_override,
            evidence_bank=evidence_bank,
        )
        if not self.use_hyper_adapter:
            params = {name: None for name in self.primitive_names}
        weights = _prepare_weights(router_out.weights, route_override, active_primitive_mask)
        primitive_outputs = []
        for index, name in enumerate(self.primitive_names):
            active_mask = _primitive_active_mask(active_primitive_mask, index, batch.target_q)
            if active_mask is not None and not bool(active_mask.any()):
                primitive_outputs.append(torch.zeros_like(batch.target_y))
                continue
            primitive_value = self.primitives[name](batch.target_u, batch.support_grid, batch.target_q, params[name], memory_bank[name])
            if active_mask is not None:
                primitive_value = primitive_value * active_mask.unsqueeze(-1).to(dtype=primitive_value.dtype)
            primitive_outputs.append(primitive_value)
        stacked = torch.stack(primitive_outputs, dim=-2)
        y_hat = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        diagnostics = {
            "primitive_entropy": primitive_entropy(weights),
            "adapter_norms": _adapter_norms(params, y_hat.device),
            "adapter_stats": _adapter_stats(params),
            "memory_norms": memory.norm(dim=-1).mean(),
            "router_context_prior_entropy": _context_prior_entropy(router_out),
            "router_query_residual_norm": _query_residual_norm(router_out),
            "per_primitive_outputs": stacked.detach(),
            "per_primitive_outputs_train": stacked,
        }
        return OVHAOutput(
            y_hat=y_hat,
            primitive_weights=weights,
            diagnostics=diagnostics,
            adapter_params=params,
            router_logits=router_out.logits,
            memory_bank=memory_bank,
            router_output=router_out,
        )


def _prepare_weights(
    learned_weights: torch.Tensor,
    route_override: torch.Tensor | None,
    active_primitive_mask: torch.Tensor | None,
) -> torch.Tensor:
    weights = learned_weights if route_override is None else route_override.to(device=learned_weights.device, dtype=learned_weights.dtype)
    if active_primitive_mask is not None:
        mask = active_primitive_mask.to(device=weights.device, dtype=weights.dtype)
        weights = weights * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weights


def _primitive_active_mask(
    active_primitive_mask: torch.Tensor | None,
    primitive_index: int,
    target_q: torch.Tensor,
) -> torch.Tensor | None:
    if active_primitive_mask is None:
        return None
    mask = active_primitive_mask.to(device=target_q.device)
    if mask.ndim == 1:
        return mask[primitive_index].view(1, 1).expand(target_q.shape[0], target_q.shape[1])
    return mask[..., primitive_index].bool()


def _adapter_norms(params: dict[str, PrimitiveParams | None], device: torch.device) -> dict[str, torch.Tensor]:
    norms = {}
    for name, param in params.items():
        if param is None or param.scale is None:
            norms[name] = torch.tensor(0.0, device=device)
        else:
            norms[name] = param.scale.norm()
    return norms


def _adapter_stats(params: dict[str, PrimitiveParams | None]) -> dict[str, dict[str, torch.Tensor]]:
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for name, param in params.items():
        if param is None:
            stats[name] = {}
            continue
        item: dict[str, torch.Tensor] = {}
        if param.scale is not None:
            item["scale_mean"] = param.scale.mean()
            item["scale_std"] = param.scale.std(unbiased=False)
            item["q_variance_scale"] = param.scale.var(dim=1, unbiased=False).mean()
        if param.bias is not None:
            item["bias_mean"] = param.bias.mean()
            item["bias_std"] = param.bias.std(unbiased=False)
            item["q_variance_bias"] = param.bias.var(dim=1, unbiased=False).mean()
        if param.spectral_mode_logits is not None:
            probs = torch.softmax(param.spectral_mode_logits, dim=-1)
            item["mode_entropy"] = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
            item["q_variance_mode_logits"] = param.spectral_mode_logits.var(dim=1, unbiased=False).mean()
        if param.spectral_frequency is not None:
            item["frequency_mean"] = param.spectral_frequency.mean()
        if param.spectral_phase is not None:
            item["phase_mean"] = param.spectral_phase.mean()
        if param.local_lengthscale is not None:
            lengthscale = torch.nn.functional.softplus(param.local_lengthscale) + 1e-3
            item["lengthscale_mean"] = lengthscale.mean()
            item["q_variance_lengthscale"] = param.local_lengthscale.var(dim=1, unbiased=False).mean()
        if param.local_shift is not None:
            item["shift_mean"] = param.local_shift.mean()
            item["q_variance_shift"] = param.local_shift.var(dim=1, unbiased=False).mean()
        if param.separable_rank_logits is not None:
            probs = torch.softmax(param.separable_rank_logits, dim=-1)
            item["rank_entropy"] = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
            item["q_variance_rank_logits"] = param.separable_rank_logits.var(dim=1, unbiased=False).mean()
        stats[name] = item
    return stats


def _context_prior_entropy(router_out: RouterOutput) -> torch.Tensor:
    if router_out.context_prior_logits is None:
        return torch.tensor(0.0, device=router_out.weights.device)
    probs = torch.softmax(router_out.context_prior_logits, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()


def _query_residual_norm(router_out: RouterOutput) -> torch.Tensor:
    if router_out.query_residual_logits is None:
        return torch.tensor(0.0, device=router_out.weights.device)
    return router_out.query_residual_logits.norm(dim=-1).mean()
