from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from moat_ovha.models.primitives.base import OperatorPrimitive, Sample
from moat_ovha.theory.special_cases import standard_attention


@dataclass(frozen=True)
class IdentityValuePrimitive(OperatorPrimitive):
    """Degenerate primitive that recovers scalar/vector-valued attention."""

    temperature: float = 1.0
    scale: float = 1.0
    bias: float = 0.0

    @property
    def name(self) -> str:
        return "identity_value"

    def apply(self, samples: Sequence[Sample], queries: Sequence[float]) -> list[float]:
        return [
            self.bias + self.scale * standard_attention(query, samples, temperature=self.temperature)
            for query in queries
        ]
