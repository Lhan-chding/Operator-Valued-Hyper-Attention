from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class MemoryState:
    tokens: tuple[tuple[float, ...], ...]
    summary: dict[str, float]


class PoolingMemoryEncoder:
    """Permutation-aware context encoder with deterministic summary tokens."""

    def __init__(self, token_count: int = 3):
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        self.token_count = token_count

    def encode(self, context: Sequence[object], mask: Optional[Sequence[bool]] = None) -> MemoryState:
        selected = _select_context(context, mask)
        summary = _context_summary(selected)
        base = (
            summary["gain"],
            summary["input_mean"],
            summary["output_mean"],
            summary["context_count"],
        )
        tokens = []
        for index in range(self.token_count):
            factor = 1.0 / (index + 1.0)
            tokens.append(tuple(_round(value * factor) for value in base))
        return MemoryState(tokens=tuple(tokens), summary=summary)


class PerceiverMemoryEncoder:
    """Small deterministic latent cross-attention encoder for variable contexts."""

    def __init__(self, latent_count: int = 4):
        if latent_count <= 0:
            raise ValueError("latent_count must be positive")
        self.latent_count = latent_count

    def encode(self, context: Sequence[object], mask: Optional[Sequence[bool]] = None) -> MemoryState:
        selected = _select_context(context, mask)
        summary = _context_summary(selected)
        features = [_demo_features(demo) for demo in selected]
        if not features:
            features = [(0.0, 0.0, 0.0, 0.0)]

        tokens = []
        for latent_index in range(self.latent_count):
            anchor = (latent_index + 1.0) / self.latent_count
            logits = [-(abs(feature[0] - anchor) + 0.25 * abs(feature[1])) for feature in features]
            weights = _softmax(logits)
            token = []
            for dim in range(4):
                token.append(_round(math.fsum(weight * feature[dim] for weight, feature in zip(weights, features))))
            tokens.append(tuple(token))
        return MemoryState(tokens=tuple(tokens), summary=summary)


def _select_context(context: Sequence[object], mask: Optional[Sequence[bool]]) -> tuple[object, ...]:
    if mask is None:
        return tuple(context)
    if len(mask) != len(context):
        raise ValueError("mask length must match context length")
    return tuple(item for item, keep in zip(context, mask) if keep)


def _context_summary(context: Sequence[object]) -> dict[str, float]:
    features = [_demo_features(demo) for demo in context]
    count = len(features)
    if count == 0:
        return {
            "context_count": 0,
            "input_mean": 0.0,
            "output_mean": 0.0,
            "gain": 1.0,
            "spectral_hint": 0.0,
            "separable_hint": 0.0,
            "local_hint": 0.0,
        }

    input_mean = _round(math.fsum(feature[0] for feature in features) / count)
    output_mean = _round(math.fsum(feature[1] for feature in features) / count)
    input_abs = math.fsum(feature[2] for feature in features) / count
    output_abs = math.fsum(feature[3] for feature in features) / count
    gain = _round(output_abs / input_abs) if input_abs > 1e-12 else 1.0
    hints = _metadata_hints(context)
    return {
        "context_count": count,
        "input_mean": input_mean,
        "output_mean": output_mean,
        "gain": gain,
        "spectral_hint": hints["spectral_hint"],
        "separable_hint": hints["separable_hint"],
        "local_hint": hints["local_hint"],
    }


def _demo_features(demo: object) -> tuple[float, float, float, float]:
    inputs = tuple(getattr(demo, "input_samples", ()))
    outputs = tuple(getattr(demo, "output_samples", ()))
    input_values = [float(sample.value) for sample in inputs]
    output_values = [float(sample.value) for sample in outputs]
    return (
        _round(math.fsum(input_values) / len(input_values)) if input_values else 0.0,
        _round(math.fsum(output_values) / len(output_values)) if output_values else 0.0,
        _round(math.fsum(abs(value) for value in input_values) / len(input_values)) if input_values else 0.0,
        _round(math.fsum(abs(value) for value in output_values) / len(output_values)) if output_values else 0.0,
    )


def _metadata_hints(context: Sequence[object]) -> dict[str, float]:
    totals = {"spectral_hint": 0.0, "separable_hint": 0.0, "local_hint": 0.0}
    if not context:
        return totals
    for demo in context:
        metadata = dict(getattr(demo, "metadata", ()))
        family = str(metadata.get("family", ""))
        totals["spectral_hint"] += float(metadata.get("weight_fourier", 1.0 if family == "fourier" else 0.0))
        totals["separable_hint"] += float(metadata.get("weight_separable", 1.0 if family == "separable" else 0.0))
        totals["local_hint"] += float(metadata.get("weight_local", 1.0 if family in {"green", "nonlinear"} else 0.0))
    return {key: _round(value / len(context)) for key, value in totals.items()}


def _softmax(logits: Sequence[float]) -> list[float]:
    max_logit = max(logits)
    values = [math.exp(logit - max_logit) for logit in logits]
    total = math.fsum(values)
    return [value / total for value in values]


def _round(value: float) -> float:
    return round(float(value), 12)
