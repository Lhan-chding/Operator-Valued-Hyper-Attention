from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from moat_ovha.models.primitives.base import OperatorPrimitive, Sample


@dataclass(frozen=True)
class FourierPrimitive(OperatorPrimitive):
    """Finite spectral-kernel primitive, analogous to an FNO value object."""

    modes: tuple[int, ...] = (1, 2, 3)
    scale: float = 1.0
    bias: float = 0.0

    @property
    def name(self) -> str:
        return "fourier"

    def apply(self, samples: Sequence[Sample], queries: Sequence[float]) -> list[float]:
        if not samples:
            return [self.bias for _ in queries]

        norm = 1.0 / len(samples)
        coefficients: list[tuple[int, float, float]] = []
        for mode in self.modes:
            cos_coeff = sum(sample.value * math.cos(2.0 * math.pi * mode * sample.point) for sample in samples)
            sin_coeff = sum(sample.value * math.sin(2.0 * math.pi * mode * sample.point) for sample in samples)
            coefficients.append((mode, cos_coeff * norm, sin_coeff * norm))

        values: list[float] = []
        for query in queries:
            spectral_value = 0.0
            for mode, cos_coeff, sin_coeff in coefficients:
                angle = 2.0 * math.pi * mode * query
                spectral_value += cos_coeff * math.cos(angle) + sin_coeff * math.sin(angle)
            values.append(self.bias + self.scale * spectral_value)
        return values
