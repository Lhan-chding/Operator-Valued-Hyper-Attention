from __future__ import annotations

import math
from typing import Sequence


def standard_attention(query: float, samples: Sequence[object], temperature: float = 1.0) -> float:
    """Scalar attention over discrete value samples using distance logits."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not samples:
        return 0.0
    logits = [-abs(query - float(sample.point)) / temperature for sample in samples]
    max_logit = max(logits)
    weights = [math.exp(logit - max_logit) for logit in logits]
    total = sum(weights)
    return sum(weight * float(sample.value) for weight, sample in zip(weights, samples)) / total


def fourier_kernel(query: float, source: float, modes: tuple[int, ...] = (1, 2, 3)) -> float:
    if not modes:
        return 0.0
    return sum(math.cos(2.0 * math.pi * mode * (query - source)) for mode in modes) / len(modes)


def deep_operator_kernel(query: float, source: float, rank: int = 3) -> float:
    if rank <= 0:
        raise ValueError("rank must be positive")
    value = 0.0
    for index in range(1, rank + 1):
        value += _basis(index, query) * _basis(index, source)
    return value / rank


def graph_local_kernel(query: float, source: float, bandwidth: float = 0.25) -> float:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    distance = abs(query - source)
    if distance > 3.0 * bandwidth:
        return 0.0
    return math.exp(-(distance * distance) / (2.0 * bandwidth * bandwidth))


def _basis(index: int, point: float) -> float:
    if index == 1:
        return 1.0
    if index % 2 == 0:
        return math.sin(math.pi * (index // 2) * point)
    return math.cos(math.pi * (index // 2) * point)
