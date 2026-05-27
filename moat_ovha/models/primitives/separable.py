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

    @property
    def name(self) -> str:
        return "separable"

    def apply(self, samples: Sequence[Sample], queries: Sequence[float]) -> list[float]:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if not samples:
            return [self.bias for _ in queries]

        norm = 1.0 / len(samples)
        coefficients = []
        for index in range(1, self.rank + 1):
            branch = sum(sample.value * _basis(index, sample.point) for sample in samples) * norm
            coefficients.append(branch)

        values: list[float] = []
        for query in queries:
            kernel_value = sum(coeff * _basis(index + 1, query) for index, coeff in enumerate(coefficients))
            values.append(self.bias + self.scale * kernel_value)
        return values


def _basis(index: int, point: float) -> float:
    if index == 1:
        return 1.0
    if index % 2 == 0:
        return math.sin(math.pi * (index // 2) * point)
    return math.cos(math.pi * (index // 2) * point)
