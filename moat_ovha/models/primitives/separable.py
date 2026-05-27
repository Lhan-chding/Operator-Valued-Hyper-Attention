from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from moat_ovha.models.primitives.base import OperatorPrimitive, Sample


@dataclass(frozen=True)
class SeparablePrimitive(OperatorPrimitive):
    """Low-rank separable kernel primitive, analogous to DeepONet basis values."""

    rank: int = 3
    scale: float = 1.0
    bias: float = 0.0
    rank_weight: float = 0.8

    @property
    def name(self) -> str:
        return "separable"

    def apply(self, samples: Sequence[Sample], queries: Sequence[float]) -> list[float]:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if not samples:
            return [self.bias for _ in queries]

        norm = 1.0 / len(samples)
        mean_value = sum(sample.value for sample in samples) * norm
        first_moment = sum(sample.value * sample.point for sample in samples) * norm
        sin_moment = sum(sample.value * math.sin(math.pi * sample.point) for sample in samples) * norm

        values: list[float] = []
        for query in queries:
            kernel_value = mean_value
            if self.rank >= 2:
                kernel_value += self.rank_weight * first_moment * query
            if self.rank >= 3:
                kernel_value += 0.5 * sin_moment * math.sin(math.pi * query)
            for index in range(4, self.rank + 1):
                coefficient = sum(sample.value * _basis(index, sample.point) for sample in samples) * norm
                kernel_value += coefficient * _basis(index, query) / index
            values.append(self.bias + self.scale * kernel_value)
        return values


def _basis(index: int, point: float) -> float:
    if index == 1:
        return 1.0
    if index % 2 == 0:
        return math.sin(math.pi * (index // 2) * point)
    return math.cos(math.pi * (index // 2) * point)
