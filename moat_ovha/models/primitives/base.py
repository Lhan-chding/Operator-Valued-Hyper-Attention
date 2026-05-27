from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Sample:
    """A scalar function sample at one query/location coordinate."""

    point: float
    value: float


class OperatorPrimitive:
    """Base interface for structured operator-valued attention values."""

    name = "operator"

    def apply(self, samples: Sequence[Sample], queries: Sequence[float]) -> list[float]:
        raise NotImplementedError

    def apply_batch(
        self,
        batch_samples: Sequence[Sequence[Sample]],
        queries: Sequence[float],
    ) -> list[list[float]]:
        return [self.apply(samples, queries) for samples in batch_samples]

    def context_condition(self, adapter: dict[str, float]) -> "OperatorPrimitive":
        updates: dict[str, float] = {}
        if hasattr(self, "scale"):
            updates["scale"] = getattr(self, "scale", 1.0) * adapter.get("scale", 1.0)
        if hasattr(self, "bias"):
            updates["bias"] = getattr(self, "bias", 0.0) + adapter.get("bias", 0.0)
        for key, value in adapter.items():
            if key not in {"scale", "bias"} and hasattr(self, key):
                updates[key] = value
        return replace(self, **updates)

    def regularization(self) -> float:
        return abs(float(getattr(self, "scale", 1.0)) - 1.0) + abs(float(getattr(self, "bias", 0.0)))

    def finite_difference(
        self,
        samples: Sequence[Sample],
        queries: Sequence[float],
        parameter: str,
        epsilon: float = 1e-4,
    ) -> list[float]:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not hasattr(self, parameter):
            raise ValueError(f"unknown primitive parameter: {parameter}")

        current = float(getattr(self, parameter))
        plus = replace(self, **{parameter: current + epsilon})
        minus = replace(self, **{parameter: current - epsilon})
        plus_values = plus.apply(samples, queries)
        minus_values = minus.apply(samples, queries)
        return [(a - b) / (2.0 * epsilon) for a, b in zip(plus_values, minus_values)]


def mean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
