from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from moat_ovha.models.hyper_adapter import LowRankHyperAdapter
from moat_ovha.models.memory import MemoryState
from moat_ovha.models.primitives import OperatorPrimitive, Sample
from moat_ovha.models.router import PrimitiveRouter, entropy


@dataclass(frozen=True)
class OVHAResult:
    predictions: list[float]
    primitive_weights: list[list[float]]
    diagnostics: dict[str, object]


class OVHALayer:
    """Operator-valued attention layer with structured primitive values."""

    def __init__(
        self,
        primitives: Sequence[OperatorPrimitive],
        router: Optional[PrimitiveRouter] = None,
        hyper_adapter: Optional[LowRankHyperAdapter] = None,
    ):
        if not primitives:
            raise ValueError("OVHALayer requires at least one primitive")
        self.primitives = tuple(primitives)
        self.router = router or PrimitiveRouter()
        self.hyper_adapter = hyper_adapter

    def forward(
        self,
        input_samples: Sequence[Sample],
        query_points: Sequence[float],
        memory: MemoryState,
    ) -> OVHAResult:
        primitive_names = [primitive.name for primitive in self.primitives]
        predictions: list[float] = []
        all_weights: list[list[float]] = []
        adapters: list[list[dict[str, float]]] = []
        entropies: list[float] = []

        for query in query_points:
            weights = self.router.weights(query, memory, primitive_names)
            primitive_values: list[float] = []
            query_adapters: list[dict[str, float]] = []
            for primitive in self.primitives:
                adapter = (
                    self.hyper_adapter.adapt(primitive.name, memory, query)
                    if self.hyper_adapter is not None
                    else {"scale": 1.0, "bias": 0.0}
                )
                conditioned = primitive.context_condition(adapter)
                primitive_values.append(conditioned.apply(input_samples, [query])[0])
                query_adapters.append(adapter)
            prediction = math.fsum(weight * value for weight, value in zip(weights, primitive_values))
            predictions.append(prediction)
            all_weights.append(weights)
            adapters.append(query_adapters)
            entropies.append(entropy(weights))

        diagnostics = {
            "adapters": adapters,
            "entropy": math.fsum(entropies) / len(entropies) if entropies else 0.0,
            "primitive_names": primitive_names,
        }
        return OVHAResult(predictions=predictions, primitive_weights=all_weights, diagnostics=diagnostics)
