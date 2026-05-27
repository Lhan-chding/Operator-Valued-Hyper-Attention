from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from moat_ovha.models.memory import MemoryState


@dataclass(frozen=True)
class PrimitiveRouter:
    """Query-conditioned primitive router with optional sparse top-k masking."""

    top_k: Optional[int] = None
    temperature: float = 1.0
    randomize: bool = False

    def weights(self, query: float, memory: MemoryState, primitive_names: Sequence[str]) -> list[float]:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not primitive_names:
            raise ValueError("at least one primitive is required")

        logits = [_score(name, query, memory, self.randomize) / self.temperature for name in primitive_names]
        dense = _softmax(logits)
        if self.top_k is None or self.top_k >= len(dense):
            return dense
        if self.top_k <= 0:
            raise ValueError("top_k must be positive when provided")

        keep = set(sorted(range(len(dense)), key=lambda index: dense[index], reverse=True)[: self.top_k])
        masked = [weight if index in keep else 0.0 for index, weight in enumerate(dense)]
        total = math.fsum(masked)
        return [weight / total if total else 0.0 for weight in masked]


def entropy(weights: Sequence[float]) -> float:
    return -math.fsum(weight * math.log(max(weight, 1e-12)) for weight in weights)


def _score(name: str, query: float, memory: MemoryState, randomize: bool) -> float:
    if randomize:
        return math.sin(97.0 * query + len(name))
    summary = memory.summary
    if name == "fourier":
        return 1.4 * float(summary.get("spectral_hint", 0.0)) + 0.15 * math.cos(2.0 * math.pi * query)
    if name == "separable":
        return 1.4 * float(summary.get("separable_hint", 0.0)) + 0.10 * math.sin(math.pi * query)
    if name == "local_kernel":
        return 1.4 * float(summary.get("local_hint", 0.0)) + 0.08 * (1.0 - abs(0.5 - query))
    return 0.0


def _softmax(logits: Sequence[float]) -> list[float]:
    max_logit = max(logits)
    values = [math.exp(logit - max_logit) for logit in logits]
    total = math.fsum(values)
    return [value / total for value in values]
