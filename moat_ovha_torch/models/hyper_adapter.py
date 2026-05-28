from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from moat_ovha_torch.models.memory import primitive_memory, safe_module_name
from moat_ovha_torch.models.primitives.base import PrimitiveParams
from moat_ovha_torch.models.router import RouterOutput


DEFAULT_PARAM_SCOPE = {
    "scale": "episode",
    "bias": "episode",
    "spectral_mode_logits": "episode",
    "separable_rank_logits": "episode",
    "local_lengthscale": "episode",
    "local_shift": "episode",
    "spectral_frequency": "episode",
    "spectral_phase": "episode",
}


class EpisodeGlobalParamHead(nn.Module):
    def __init__(self, input_dim: int, d_model: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Linear(d_model, output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).unsqueeze(1)


class QueryLocalParamHead(nn.Module):
    def __init__(self, input_dim: int, d_model: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Linear(d_model, output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class HyperAdapter(nn.Module):
    """Primitive-specific operator parameters from primitive memory and router posterior."""

    def __init__(
        self,
        primitive_names: tuple[str, ...],
        d_model: int = 64,
        query_conditioned: bool = True,
        spectral_modes: int = 4,
        separable_rank: int = 4,
        param_scope_config: Mapping[str, str] | None = None,
        controlled_generator_variant: str = "model_aligned",
        return_raw: bool = True,
    ):
        super().__init__()
        self.primitive_names = primitive_names
        self.query_conditioned = query_conditioned
        self.spectral_modes = spectral_modes
        self.separable_rank = separable_rank
        self.controlled_generator_variant = controlled_generator_variant
        self.return_raw = return_raw
        self.param_scope_config = dict(DEFAULT_PARAM_SCOPE)
        if controlled_generator_variant == "model_aligned":
            self.param_scope_config["spectral_frequency"] = "disabled"
            self.param_scope_config["spectral_phase"] = "disabled"
            self.param_scope_config["local_shift"] = "disabled"
        if param_scope_config is not None:
            self.param_scope_config.update(param_scope_config)
        if not query_conditioned:
            self.param_scope_config = {
                key: ("disabled" if value == "disabled" else "episode")
                for key, value in self.param_scope_config.items()
            }

        self.global_heads = nn.ModuleDict()
        self.query_heads = nn.ModuleDict()
        for name in primitive_names:
            key = safe_module_name(name)
            output_dim = self._output_dim(name)
            self.global_heads[key] = EpisodeGlobalParamHead(d_model + 2, d_model, output_dim)
            self.query_heads[key] = QueryLocalParamHead(d_model + 3, d_model, output_dim)
        self._initialize_identity_defaults()

    def forward(
        self,
        memory: torch.Tensor | dict[str, torch.Tensor],
        target_q: torch.Tensor,
        router_out: RouterOutput | None = None,
    ) -> dict[str, PrimitiveParams]:
        return {name: self._params_for_name(name, memory, target_q, router_out) for name in self.primitive_names}

    def parameters_for_primitive(self, primitive_name: str):
        key = safe_module_name(primitive_name)
        yield from self.global_heads[key].parameters()
        yield from self.query_heads[key].parameters()

    def _params_for_name(
        self,
        primitive_name: str,
        memory: torch.Tensor | dict[str, torch.Tensor],
        target_q: torch.Tensor,
        router_out: RouterOutput | None,
    ) -> PrimitiveParams:
        primitive_index = self.primitive_names.index(primitive_name)
        pooled = primitive_memory(memory, primitive_name).mean(dim=1)
        posterior, entropy = _router_features(router_out, primitive_index, target_q)
        global_features = torch.cat([pooled, posterior.mean(dim=1), entropy.mean(dim=1)], dim=-1)
        repeated = pooled.unsqueeze(1).expand(-1, target_q.shape[1], -1)
        query_features = torch.cat([repeated, target_q, posterior, entropy], dim=-1)
        key = safe_module_name(primitive_name)
        global_raw = self.global_heads[key](global_features).expand(-1, target_q.shape[1], -1)
        query_raw = self.query_heads[key](query_features)
        selected_raw = self._select_raw_by_scope(primitive_name, global_raw, query_raw)
        return self._params_for(primitive_name, selected_raw, global_raw, query_raw)

    def _select_raw_by_scope(self, primitive_name: str, global_raw: torch.Tensor, query_raw: torch.Tensor) -> torch.Tensor:
        if not self.query_conditioned:
            return global_raw
        kind = _primitive_kind(primitive_name)
        selected = global_raw.clone()
        for param_name, value_slice in self._slices(kind).items():
            if self._scope(param_name) == "query":
                selected[..., value_slice] = query_raw[..., value_slice]
        return selected

    def _output_dim(self, primitive_name: str) -> int:
        kind = _primitive_kind(primitive_name)
        if kind == "spectral":
            return 4 + self.spectral_modes if self.controlled_generator_variant == "full" else 2 + self.spectral_modes
        if kind == "local":
            return 4
        if kind == "separable":
            return 2 + self.separable_rank
        return 3

    def _slices(self, primitive_kind: str) -> dict[str, slice]:
        if primitive_kind == "spectral":
            if self.controlled_generator_variant == "full":
                return {
                    "scale": slice(0, 1),
                    "bias": slice(1, 2),
                    "spectral_frequency": slice(2, 3),
                    "spectral_phase": slice(3, 4),
                    "spectral_mode_logits": slice(4, 4 + self.spectral_modes),
                }
            return {
                "scale": slice(0, 1),
                "bias": slice(1, 2),
                "spectral_mode_logits": slice(2, 2 + self.spectral_modes),
            }
        if primitive_kind == "local":
            return {
                "scale": slice(0, 1),
                "bias": slice(1, 2),
                "local_lengthscale": slice(2, 3),
                "local_shift": slice(3, 4),
            }
        if primitive_kind == "separable":
            return {
                "scale": slice(0, 1),
                "bias": slice(1, 2),
                "separable_rank_logits": slice(2, 2 + self.separable_rank),
            }
        return {"scale": slice(0, 1), "bias": slice(1, 2)}

    def _params_for(
        self,
        primitive_name: str,
        raw: torch.Tensor,
        global_raw: torch.Tensor,
        query_raw: torch.Tensor,
    ) -> PrimitiveParams:
        kind = _primitive_kind(primitive_name)
        scale = 1.0 + 0.1 * torch.tanh(raw[..., 0:1])
        bias = 0.1 * torch.tanh(raw[..., 1:2])
        raw_payload = {"selected": raw, "global": global_raw, "query": query_raw} if self.return_raw else None
        scope = dict(self.param_scope_config)
        if kind == "spectral":
            if self.controlled_generator_variant == "full":
                frequency = 1.0 + 0.25 * torch.tanh(raw[..., 2:3]) if self._scope("spectral_frequency") != "disabled" else None
                phase = torch.pi * torch.tanh(raw[..., 3:4]) if self._scope("spectral_phase") != "disabled" else None
                mode_logits = raw[..., 4 : 4 + self.spectral_modes]
            else:
                frequency = None
                phase = None
                mode_logits = raw[..., 2 : 2 + self.spectral_modes]
            kernel_params = {"mode_logits": mode_logits}
            if frequency is not None:
                kernel_params["frequency"] = frequency
            if phase is not None:
                kernel_params["phase"] = phase
            return PrimitiveParams(
                scale=scale,
                bias=bias,
                spectral_frequency=frequency,
                spectral_phase=phase,
                spectral_mode_logits=mode_logits,
                kernel_params=kernel_params,
                raw=raw_payload,
                scope=scope,
            )
        if kind == "local":
            lengthscale = -2.0 + 0.5 * torch.tanh(raw[..., 2:3])
            shift = None if self._scope("local_shift") == "disabled" else 0.25 * torch.tanh(raw[..., 3:4])
            kernel_params = {"lengthscale": lengthscale}
            if shift is not None:
                kernel_params["shift"] = shift
            return PrimitiveParams(
                scale=scale,
                bias=bias,
                local_lengthscale=lengthscale,
                local_shift=shift,
                kernel_params=kernel_params,
                raw=raw_payload,
                scope=scope,
            )
        if kind == "separable":
            rank_logits = raw[..., 2 : 2 + self.separable_rank]
            return PrimitiveParams(
                scale=scale,
                bias=bias,
                separable_rank_logits=rank_logits,
                kernel_params={"rank_logits": rank_logits},
                raw=raw_payload,
                scope=scope,
            )
        kernel = {"lengthscale": raw[..., 2:3]} if raw.shape[-1] > 2 else None
        return PrimitiveParams(scale=scale, bias=bias, kernel_params=kernel, raw=raw_payload, scope=scope)

    def _scope(self, param_name: str) -> str:
        return self.param_scope_config.get(param_name, "episode")

    def _initialize_identity_defaults(self) -> None:
        for head in list(self.global_heads.values()) + list(self.query_heads.values()):
            final = head.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)


def _router_features(
    router_out: RouterOutput | None,
    primitive_index: int,
    target_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if router_out is None:
        posterior = torch.zeros(target_q.shape[0], target_q.shape[1], 1, dtype=target_q.dtype, device=target_q.device)
        entropy = torch.zeros_like(posterior)
        return posterior, entropy
    posterior = router_out.weights[..., primitive_index : primitive_index + 1].detach()
    entropy = -(router_out.weights.detach() * router_out.weights.detach().clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
    return posterior, entropy


def _primitive_kind(primitive_name: str) -> str:
    if primitive_name.startswith("mlp_expert"):
        return "mlp_expert"
    return primitive_name
