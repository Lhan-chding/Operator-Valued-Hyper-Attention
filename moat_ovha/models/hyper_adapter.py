from __future__ import annotations

import math
from dataclasses import dataclass

from moat_ovha.models.memory import MemoryState


@dataclass(frozen=True)
class LowRankHyperAdapter:
    """Low-rank context/query adapter for primitive scale and bias."""

    enabled: bool = True
    rank: int = 2
    strength: float = 0.15

    def adapt(self, primitive_name: str, memory: MemoryState, query: float) -> dict[str, float]:
        if not self.enabled:
            return {"scale": 1.0, "bias": 0.0}
        gain = float(memory.summary.get("gain", 1.0))
        context_count = float(memory.summary.get("context_count", 0.0))
        rank_factor = min(max(self.rank, 1), 8) / 8.0
        primitive_bias = _primitive_bias(primitive_name, memory)
        query_feature = math.sin(math.pi * query)
        scale = 1.0 + self.strength * rank_factor * (gain - 1.0 + primitive_bias)
        bias = self.strength * 0.1 * query_feature / (1.0 + context_count)
        return {"scale": scale, "bias": bias}


def _primitive_bias(primitive_name: str, memory: MemoryState) -> float:
    if primitive_name == "fourier":
        return float(memory.summary.get("spectral_hint", 0.0))
    if primitive_name == "separable":
        return float(memory.summary.get("separable_hint", 0.0))
    if primitive_name == "local_kernel":
        return float(memory.summary.get("local_hint", 0.0))
    return 0.0
