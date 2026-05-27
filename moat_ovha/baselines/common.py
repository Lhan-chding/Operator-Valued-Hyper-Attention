from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from moat_ovha.data.operator_zoo import MetaTask
from moat_ovha.models.hyper_adapter import LowRankHyperAdapter
from moat_ovha.models.memory import MemoryState, PoolingMemoryEncoder
from moat_ovha.models.ovha_layer import OVHALayer
from moat_ovha.models.primitives import (
    FourierPrimitive,
    IdentityValuePrimitive,
    LocalKernelPrimitive,
    SeparablePrimitive,
)
from moat_ovha.models.router import PrimitiveRouter


@dataclass(frozen=True)
class PredictionResult:
    predictions: list[float]
    diagnostics: dict[str, object]
    parameter_count: int


class Predictor(Protocol):
    name: str

    def predict(self, task: MetaTask) -> PredictionResult:
        ...


def make_predictor(name: str) -> Predictor:
    mapping = {
        "ovha_full": lambda: OVHAPredictor(name="ovha_full", use_memory=True, use_hyper=True, top_k=None),
        "ovha_sparse": lambda: OVHAPredictor(name="ovha_sparse", use_memory=True, use_hyper=True, top_k=2),
        "vector_value": lambda: PrimitiveOnlyPredictor("vector_value", IdentityValuePrimitive()),
        "no_hyper_adapter": lambda: OVHAPredictor(name="no_hyper_adapter", use_memory=True, use_hyper=False, top_k=None),
        "no_memory": lambda: OVHAPredictor(name="no_memory", use_memory=False, use_hyper=True, top_k=None),
        "mlp_expert": lambda: PrimitiveOnlyPredictor("mlp_expert", LocalKernelPrimitive(bandwidth=0.45)),
        "random_router": lambda: OVHAPredictor(
            name="random_router",
            use_memory=True,
            use_hyper=True,
            top_k=None,
            random_router=True,
        ),
        "transformer_only": lambda: PrimitiveOnlyPredictor("transformer_only", IdentityValuePrimitive(temperature=0.25)),
        "perceiver_io_style": lambda: PerceiverStylePredictor(),
        "icon_style": lambda: ICONStylePredictor(),
        "deeponet": lambda: PrimitiveOnlyPredictor("deeponet", SeparablePrimitive(rank=4)),
        "fno": lambda: PrimitiveOnlyPredictor("fno", FourierPrimitive(modes=(1, 2, 3, 4))),
        "simple_stack": lambda: SimpleStackPredictor(),
    }
    if name not in mapping:
        raise ValueError(f"unknown predictor: {name}")
    return mapping[name]()


class OVHAPredictor:
    def __init__(
        self,
        name: str,
        use_memory: bool,
        use_hyper: bool,
        top_k: Optional[int],
        random_router: bool = False,
    ):
        self.name = name
        self.use_memory = use_memory
        self.layer = OVHALayer(
            primitives=(
                FourierPrimitive(modes=(1, 2, 3)),
                SeparablePrimitive(rank=3),
                LocalKernelPrimitive(bandwidth=0.25),
            ),
            router=PrimitiveRouter(top_k=top_k, randomize=random_router),
            hyper_adapter=LowRankHyperAdapter(enabled=use_hyper),
        )

    def predict(self, task: MetaTask) -> PredictionResult:
        memory = PoolingMemoryEncoder(token_count=3).encode(task.context) if self.use_memory else _empty_memory()
        result = self.layer.forward(task.input_samples, task.query_points, memory)
        return PredictionResult(
            predictions=result.predictions,
            diagnostics=result.diagnostics,
            parameter_count=128 if self.use_memory else 96,
        )


class PrimitiveOnlyPredictor:
    def __init__(self, name: str, primitive):
        self.name = name
        self.primitive = primitive

    def predict(self, task: MetaTask) -> PredictionResult:
        predictions = self.primitive.apply(task.input_samples, task.query_points)
        return PredictionResult(
            predictions=predictions,
            diagnostics={"primitive_names": [self.primitive.name], "entropy": 0.0},
            parameter_count=32,
        )


class PerceiverStylePredictor:
    name = "perceiver_io_style"

    def predict(self, task: MetaTask) -> PredictionResult:
        memory = PoolingMemoryEncoder(token_count=2).encode(task.context)
        gain = float(memory.summary.get("gain", 1.0))
        base = LocalKernelPrimitive(bandwidth=0.4, scale=gain)
        return PredictionResult(
            predictions=base.apply(task.input_samples, task.query_points),
            diagnostics={"primitive_names": ["latent_attention"], "entropy": 0.0},
            parameter_count=80,
        )


class ICONStylePredictor:
    name = "icon_style"

    def predict(self, task: MetaTask) -> PredictionResult:
        memory = PoolingMemoryEncoder(token_count=2).encode(task.context)
        output_shift = float(memory.summary.get("output_mean", 0.0))
        base = IdentityValuePrimitive(temperature=0.35, bias=0.15 * output_shift)
        return PredictionResult(
            predictions=base.apply(task.input_samples, task.query_points),
            diagnostics={"primitive_names": ["context_prompt"], "entropy": 0.0},
            parameter_count=72,
        )


class SimpleStackPredictor:
    name = "simple_stack"

    def predict(self, task: MetaTask) -> PredictionResult:
        fourier = FourierPrimitive(modes=(1, 2, 3)).apply(task.input_samples, task.query_points)
        separable = SeparablePrimitive(rank=3).apply(task.input_samples, task.query_points)
        local = LocalKernelPrimitive(bandwidth=0.3).apply(task.input_samples, task.query_points)
        predictions = [(a + b + c) / 3.0 for a, b, c in zip(fourier, separable, local)]
        return PredictionResult(
            predictions=predictions,
            diagnostics={"primitive_names": ["fourier", "separable", "local_kernel"], "entropy": 0.0},
            parameter_count=112,
        )


def _empty_memory() -> MemoryState:
    return MemoryState(
        tokens=((0.0, 0.0, 0.0, 0.0),),
        summary={
            "context_count": 0,
            "input_mean": 0.0,
            "output_mean": 0.0,
            "gain": 1.0,
            "spectral_hint": 0.0,
            "separable_hint": 0.0,
            "local_hint": 0.0,
        },
    )
