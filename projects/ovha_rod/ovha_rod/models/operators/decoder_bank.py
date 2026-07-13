"""Inference-only composition boundary for post-RQGO decoder operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from .decoder_contracts import (
    DecoderOperatorResidual,
    DecoderResidualState,
    StructuredResidualFusion,
)
from .hyper_adapter import HyperAdapterResult, LowRankHyperAdapter
from .ms_tleo import MSTLEO
from .operator_memory import OperatorMemory, OperatorMemoryState
from .operator_router import OperatorRouter, OperatorRouterResult
from .qsro import QuerySpatialRelationOperator
from .rceo import RCEO, RCEOResult
from .tq_cato import TQCATO


OPERATOR_NAMES = ("qsro", "tq_cato", "ms_tleo")


@dataclass(frozen=True)
class DecoderOperatorContext:
    """Frozen inference-time inputs for one decoder-layer bank call."""

    parent: DecoderResidualState
    boxes: Tensor
    valid: Tensor
    layer_index: int
    relation_role: Optional[Tensor] = None
    text: Optional[Tensor] = None
    text_valid: Optional[Tensor] = None
    feature_maps: Sequence[Tensor] = field(default_factory=tuple)
    valid_ratios: Optional[Tensor] = None
    memory_state: Optional[OperatorMemoryState] = None
    operator_available: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.parent, DecoderResidualState):
            raise ValueError("parent must be a DecoderResidualState")
        query = self.parent.query
        batch_queries = query.shape[:2]
        if self.boxes.shape != (*batch_queries, 4):
            raise ValueError("boxes must have shape [B,Q,4]")
        if self.valid.shape != batch_queries or self.valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,Q] tensor")
        if not isinstance(self.layer_index, int) or isinstance(
                self.layer_index, bool) or self.layer_index < 0:
            raise ValueError("layer_index must be a non-negative integer")
        _validate_float_like(self.boxes, query, "boxes")
        if self.valid.device != query.device:
            raise ValueError("valid and parent query must share a device")
        if not bool(((self.boxes >= 0.0) & (self.boxes <= 1.0)).all()):
            raise ValueError("boxes must contain normalized cxcywh values")
        if self.valid.any() and not (self.boxes[..., 2:][self.valid] > 0).all():
            raise ValueError("valid boxes must have positive width and height")

        self._validate_relation_role(query)
        self._validate_text(query)
        feature_maps = tuple(self.feature_maps)
        object.__setattr__(self, "feature_maps", feature_maps)
        self._validate_feature_maps(query, feature_maps)
        self._validate_memory(query)
        self._validate_availability(query)

    def _validate_relation_role(self, query: Tensor) -> None:
        if self.relation_role is None:
            return
        if self.relation_role.shape != (query.shape[0], query.shape[-1]):
            raise ValueError("relation_role must have shape [B,D]")
        _validate_float_like(self.relation_role, query, "relation_role")

    def _validate_text(self, query: Tensor) -> None:
        if (self.text is None) != (self.text_valid is None):
            raise ValueError("text and text_valid must be provided together")
        if self.text is None:
            return
        if (self.text.ndim != 3 or self.text.shape[0] != query.shape[0]
                or self.text.shape[-1] != query.shape[-1]
                or self.text.shape[1] <= 0):
            raise ValueError("text must have shape [B,T,D] with T > 0")
        _validate_float_like(self.text, query, "text")
        if (self.text_valid.shape != self.text.shape[:2]
                or self.text_valid.dtype != torch.bool):
            raise ValueError("text_valid must be a boolean [B,T] tensor")
        if self.text_valid.device != query.device:
            raise ValueError("text_valid and text must share a device")

    def _validate_feature_maps(
        self, query: Tensor, feature_maps: tuple[Tensor, ...]
    ) -> None:
        if not feature_maps:
            if self.valid_ratios is not None:
                raise ValueError("valid_ratios require feature_maps")
            return
        for feature_map in feature_maps:
            if not isinstance(feature_map, Tensor) or feature_map.ndim != 4:
                raise ValueError("feature_maps must contain [B,D,H,W] tensors")
            if (feature_map.shape[:2] != (query.shape[0], query.shape[-1])
                    or min(feature_map.shape[2:]) <= 0):
                raise ValueError("feature_maps must match [B,D] and be non-empty")
            _validate_float_like(feature_map, query, "feature_maps")
        if self.valid_ratios is None:
            raise ValueError("valid_ratios are required with feature_maps")
        expected = (query.shape[0], len(feature_maps), 2)
        if self.valid_ratios.shape != expected:
            raise ValueError("valid_ratios levels must match feature_maps")
        _validate_float_like(self.valid_ratios, query, "valid_ratios")
        in_range = (self.valid_ratios > 0.0) & (self.valid_ratios <= 1.0)
        if not bool(in_range.all()):
            raise ValueError("valid_ratios must be in the range (0, 1]")

    def _validate_memory(self, query: Tensor) -> None:
        if self.memory_state is None:
            return
        if not isinstance(self.memory_state, OperatorMemoryState):
            raise ValueError("memory_state must be an OperatorMemoryState")
        if self.memory_state.value.shape != query.shape:
            raise ValueError("memory_state must match parent query")
        if (self.memory_state.value.device != query.device
                or self.memory_state.value.dtype != query.dtype):
            raise ValueError("memory_state and parent query must match")

    def _validate_availability(self, query: Tensor) -> None:
        if self.operator_available is None:
            return
        allowed = ((len(OPERATOR_NAMES),), (*query.shape[:2], len(OPERATOR_NAMES)))
        if self.operator_available.shape not in allowed:
            raise ValueError("operator_available has the wrong shape")
        if self.operator_available.dtype != torch.bool:
            raise ValueError("operator_available must be boolean")
        if self.operator_available.device != query.device:
            raise ValueError("operator_available must share the query device")


@dataclass(frozen=True)
class BankOutput:
    """Frozen decoder state, recurrent state, and bank diagnostics."""

    fused: DecoderResidualState
    residuals: Mapping[str, DecoderOperatorResidual]
    memory_state: OperatorMemoryState
    router: OperatorRouterResult
    reliability: RCEOResult
    adaptation: HyperAdapterResult
    availability: Tensor
    artifacts: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fused, DecoderResidualState):
            raise ValueError("fused must be a DecoderResidualState")
        residuals = dict(self.residuals)
        if any(name not in OPERATOR_NAMES for name in residuals):
            raise ValueError("residual mapping contains an unknown operator")
        if any(not isinstance(value, DecoderOperatorResidual)
               for value in residuals.values()):
            raise ValueError("residual values must be DecoderOperatorResidual objects")
        object.__setattr__(self, "residuals", MappingProxyType(residuals))
        if not isinstance(self.memory_state, OperatorMemoryState):
            raise ValueError("memory_state must be an OperatorMemoryState")
        if not isinstance(self.router, OperatorRouterResult):
            raise ValueError("router must be an OperatorRouterResult")
        if not isinstance(self.reliability, RCEOResult):
            raise ValueError("reliability must be an RCEOResult")
        if not isinstance(self.adaptation, HyperAdapterResult):
            raise ValueError("adaptation must be a HyperAdapterResult")
        expected = (*self.fused.query.shape[:2], len(OPERATOR_NAMES))
        if self.availability.shape != expected or self.availability.dtype != torch.bool:
            raise ValueError("availability must be a boolean [B,Q,O] tensor")
        if self.availability.device != self.fused.query.device:
            raise ValueError("availability must share the fused state device")
        if self.router.weights.shape != expected:
            raise ValueError("router output must match availability")
        artifacts = dict(self.artifacts)
        for name, value in artifacts.items():
            if not isinstance(value, Tensor):
                raise ValueError(f"artifact {name!r} must be a tensor")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"artifact {name!r} must be finite floating point")
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))


class DecoderOperatorBank(nn.Module):
    """Compose isolated decoder primitives without integrating with MMDet."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        router_hidden_dim: int,
        adapter_rank: int,
        enabled_operators: Sequence[str] = OPERATOR_NAMES,
        use_router: bool = True,
        use_memory: bool = True,
        use_hyper_adapter: bool = True,
        use_rceo: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        enabled = tuple(enabled_operators)
        if not enabled or any(name not in OPERATOR_NAMES for name in enabled):
            raise ValueError("enabled_operators must name at least one known operator")
        if len(set(enabled)) != len(enabled):
            raise ValueError("enabled_operators must not contain duplicate names")
        flags = {
            "use_router": use_router,
            "use_memory": use_memory,
            "use_hyper_adapter": use_hyper_adapter,
            "use_rceo": use_rceo,
        }
        for name, value in flags.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        self.d_model = d_model
        self.num_layers = int(num_layers)
        self.enabled_operators = enabled
        self.use_router = use_router
        self.use_memory = use_memory
        self.use_hyper_adapter = use_hyper_adapter
        self.use_rceo = use_rceo
        self.qsro = QuerySpatialRelationOperator(d_model=d_model)
        self.tq_cato = TQCATO(d_model=d_model)
        self.ms_tleo = MSTLEO(d_model=d_model)
        self.memory = OperatorMemory(d_model=d_model)
        self.router = OperatorRouter(
            d_model=d_model,
            operator_count=len(OPERATOR_NAMES),
            num_layers=num_layers,
            hidden_dim=router_hidden_dim,
        )
        self.adapter = LowRankHyperAdapter(
            d_model=d_model,
            operator_count=len(OPERATOR_NAMES),
            rank=adapter_rank,
        )
        self.rceo = RCEO(d_model=d_model, operator_count=len(OPERATOR_NAMES))
        self.fusion = StructuredResidualFusion()

    def forward(self, context: DecoderOperatorContext) -> BankOutput:
        self._validate_context(context)
        availability = self._effective_availability(context)
        active = tuple(
            name for index, name in enumerate(OPERATOR_NAMES)
            if bool(availability[..., index].any())
        )
        self._require_active_inputs(context, active)

        previous = context.memory_state
        if previous is None or not self.use_memory:
            previous = self.memory.initialize_like(context.parent.query)
        memory_state = (
            self.memory(previous, context.parent.query, context.valid)
            if self.use_memory else previous
        )
        reliability = (
            self.rceo(
                context.parent.query,
                context.boxes,
                context.parent.referent_score,
                context.valid,
            )
            if self.use_rceo else _neutral_reliability(
                context.parent.query, context.valid)
        )
        router = (
            self.router(
                context.parent.query,
                memory_state.value,
                context.layer_index,
                context.valid,
                reliability_log_prior=reliability.log_prior,
                operator_available=availability,
            )
            if self.use_router else _uniform_router(
                context.parent.query, availability)
        )
        adapter = (
            self.adapter(context.parent.query, memory_state.value, context.valid)
            if self.use_hyper_adapter else _neutral_adapter(
                context.parent.query)
        )

        residuals: dict[str, DecoderOperatorResidual] = {}
        artifacts: dict[str, Tensor] = {}
        if "qsro" in active:
            operator_valid = availability[..., 0]
            raw = self.qsro(
                context.parent.query,
                context.boxes,
                context.relation_role,
                operator_valid,
            )
            residuals["qsro"] = self._adapt(raw, 0, availability, router, adapter)
        if "tq_cato" in active:
            operator_valid = availability[..., 1]
            safe_query_valid, safe_text_valid, active_samples = _safe_tq_masks(
                operator_valid, context.text_valid)
            result = self.tq_cato(
                context.parent.query,
                context.text,
                safe_query_valid,
                safe_text_valid,
            )
            residuals["tq_cato"] = self._adapt(
                result.residual, 1, availability, router, adapter)
            artifacts["tq_cato_transport"] = result.transport * active_samples[
                :, None, None].to(dtype=result.transport.dtype)
        if "ms_tleo" in active:
            operator_valid = availability[..., 2]
            raw = self.ms_tleo(
                context.feature_maps,
                context.boxes,
                operator_valid,
                valid_ratios=context.valid_ratios,
            )
            residuals["ms_tleo"] = self._adapt(
                raw, 2, availability, router, adapter)
            artifacts["valid_ratio_mean"] = context.valid_ratios.mean()

        fused = self.fusion(context.parent, tuple(residuals.values()))
        return BankOutput(
            fused=fused,
            residuals=residuals,
            memory_state=memory_state,
            router=router,
            reliability=reliability,
            adaptation=adapter,
            availability=availability,
            artifacts=artifacts,
        )

    def _validate_context(self, context: DecoderOperatorContext) -> None:
        if not isinstance(context, DecoderOperatorContext):
            raise ValueError("context must be a DecoderOperatorContext")
        if context.parent.query.shape[-1] != self.d_model:
            raise ValueError("context feature dimension must equal d_model")
        if context.layer_index >= self.num_layers:
            raise ValueError("layer_index is outside the configured decoder")

    def _effective_availability(self, context: DecoderOperatorContext) -> Tensor:
        query = context.parent.query
        dynamic = context.operator_available
        if dynamic is None:
            dynamic = torch.ones(
                (*query.shape[:2], len(OPERATOR_NAMES)),
                dtype=torch.bool,
                device=query.device,
            )
        elif dynamic.ndim == 1:
            dynamic = dynamic.view(1, 1, -1).expand(
                *query.shape[:2], len(OPERATOR_NAMES))
        enabled = torch.tensor(
            [name in self.enabled_operators for name in OPERATOR_NAMES],
            dtype=torch.bool,
            device=query.device,
        )
        available = dynamic & enabled.view(1, 1, -1)
        available = available & context.valid[..., None]
        if context.valid.any() and not available[context.valid].any(dim=-1).all():
            raise ValueError("each valid query must have an available operator")
        return available

    @staticmethod
    def _require_active_inputs(
        context: DecoderOperatorContext, active: tuple[str, ...]
    ) -> None:
        if "qsro" in active and context.relation_role is None:
            raise ValueError("relation_role is required when qsro is available")
        if "tq_cato" in active and context.text is None:
            raise ValueError("text and text_valid are required when tq_cato is available")
        if "ms_tleo" in active and not context.feature_maps:
            raise ValueError(
                "feature_maps and valid_ratios are required when ms_tleo is available")

    @staticmethod
    def _adapt(
        residual: DecoderOperatorResidual,
        index: int,
        availability: Tensor,
        router: OperatorRouterResult,
        adapter: HyperAdapterResult,
    ) -> DecoderOperatorResidual:
        valid = residual.valid & availability[..., index]
        mask = valid[..., None]
        weight = router.weights[..., index]
        query_delta = (
            residual.query_delta * (1.0 + adapter.scale[..., index, :])
            + adapter.shift[..., index, :]
        ) * weight[..., None]
        box_delta = residual.box_delta * weight[..., None]
        score_delta = residual.score_delta * weight
        gate_logits = residual.gate_logits * (
            1.0 + adapter.channel_delta[..., index, :])
        diagnostics = dict(residual.diagnostics)
        diagnostics["router_weight_mean"] = _masked_mean(weight, valid)
        return DecoderOperatorResidual(
            query_delta=torch.where(mask, query_delta, torch.zeros_like(query_delta)),
            box_delta=torch.where(mask, box_delta, torch.zeros_like(box_delta)),
            score_delta=torch.where(valid, score_delta, torch.zeros_like(score_delta)),
            gate_logits=torch.where(mask, gate_logits, torch.zeros_like(gate_logits)),
            valid=valid,
            diagnostics=diagnostics,
        )


def _validate_float_like(value: Tensor, reference: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name} must match parent query device and dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _neutral_reliability(query: Tensor, valid: Tensor) -> RCEOResult:
    shape = (*query.shape[:2], len(OPERATOR_NAMES))
    reliability = query.new_full(shape, 0.5)
    reliability = torch.where(
        valid[..., None], reliability, torch.zeros_like(reliability))
    return RCEOResult(
        reliability=reliability,
        log_prior=query.new_zeros(shape),
    )


def _uniform_router(
    query: Tensor, availability: Tensor
) -> OperatorRouterResult:
    available = availability.to(dtype=query.dtype)
    denominator = available.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return OperatorRouterResult(
        weights=available / denominator,
        logits=torch.zeros_like(available),
    )


def _neutral_adapter(query: Tensor) -> HyperAdapterResult:
    base = (*query.shape[:2], len(OPERATOR_NAMES))
    return HyperAdapterResult(
        scale=query.new_zeros((*base, query.shape[-1])),
        shift=query.new_zeros((*base, query.shape[-1])),
        channel_delta=query.new_zeros((*base, 3)),
    )


def _safe_tq_masks(
    operator_valid: Tensor, text_valid: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    active_samples = operator_valid.any(dim=-1)
    query_fallback = (
        torch.arange(operator_valid.shape[1], device=operator_valid.device)
        == 0
    ).view(1, -1).expand_as(operator_valid)
    text_fallback = (
        torch.arange(text_valid.shape[1], device=text_valid.device) == 0
    ).view(1, -1).expand_as(text_valid)
    return (
        torch.where(active_samples[:, None], operator_valid, query_fallback),
        torch.where(active_samples[:, None], text_valid, text_fallback),
        active_samples,
    )


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    masked = torch.where(valid, values, torch.zeros_like(values))
    denominator = valid.sum().clamp_min(1).to(dtype=values.dtype)
    return masked.sum() / denominator
