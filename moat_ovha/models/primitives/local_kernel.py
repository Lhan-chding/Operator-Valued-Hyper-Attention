from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from moat_ovha.models.primitives.base import OperatorPrimitive, Sample


@dataclass(frozen=True)
class LocalKernelPrimitive(OperatorPrimitive):
    """Compact local kernel primitive for graph/GNO-style neighborhoods."""

    bandwidth: float = 0.25
    scale: float = 1.0
    bias: float = 0.0
    decay: Optional[float] = None

    @property
    def name(self) -> str:
        return "local_kernel"

    def apply(self, samples: Sequence[Sample], queries: Sequence[float]) -> list[float]:
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if not samples:
            return [self.bias for _ in queries]

        values: list[float] = []
        for query in queries:
            weighted_sum = 0.0
            weight_total = 0.0
            for sample in samples:
                distance = query - sample.point
                if self.decay is None:
                    weight = math.exp(-(distance * distance) / (2.0 * self.bandwidth * self.bandwidth))
                else:
                    weight = math.exp(-self.decay * abs(distance))
                weighted_sum += weight * sample.value
                weight_total += weight
            local_value = weighted_sum / weight_total if weight_total else 0.0
            values.append(self.bias + self.scale * local_value)
        return values
