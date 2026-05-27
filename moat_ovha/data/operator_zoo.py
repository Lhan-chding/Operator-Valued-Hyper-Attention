from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence, Union

from moat_ovha.models.primitives import Sample


@dataclass(frozen=True)
class Demonstration:
    input_samples: tuple[Sample, ...]
    output_samples: tuple[Sample, ...]
    metadata: tuple[tuple[str, Union[float, str]], ...]


@dataclass(frozen=True)
class MetaTask:
    family: str
    split: str
    context: tuple[Demonstration, ...]
    input_samples: tuple[Sample, ...]
    query_points: tuple[float, ...]
    targets: tuple[float, ...]
    metadata: tuple[tuple[str, Union[float, str]], ...]


class OperatorZoo:
    """Deterministic 1D operator task generator for phase-1 validation."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def make_task(
        self,
        family: str,
        split: str,
        context_size: int,
        resolution: int,
        parameter_holdout: bool = False,
    ) -> MetaTask:
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if resolution < 4:
            raise ValueError("resolution must be at least 4")
        if family not in {"fourier", "green", "separable", "nonlinear", "mixed"}:
            raise ValueError(f"unknown operator family: {family}")

        rng = random.Random(_stable_seed(self.seed, family, split, context_size, resolution, parameter_holdout))
        params = _sample_parameters(rng, family, split, parameter_holdout)
        points = _grid(resolution)
        operator = _operator_for(family, params)
        metadata = tuple(sorted({"family": family, **params}.items()))

        context = []
        for demo_index in range(context_size):
            demo_input = _input_function(points, rng, demo_index)
            demo_output = tuple(Sample(point=point, value=operator(demo_input, point)) for point in points)
            context.append(Demonstration(input_samples=demo_input, output_samples=demo_output, metadata=metadata))

        input_samples = _input_function(points, rng, context_size + 1)
        query_points = _grid(resolution + 1)
        targets = tuple(operator(input_samples, point) for point in query_points)
        return MetaTask(
            family=family,
            split=split,
            context=tuple(context),
            input_samples=input_samples,
            query_points=query_points,
            targets=targets,
            metadata=metadata,
        )

    def make_suite(
        self,
        families: Sequence[str],
        split: str,
        context_sizes: Sequence[int],
        resolution: int,
        parameter_holdout: bool = False,
    ) -> list[MetaTask]:
        tasks: list[MetaTask] = []
        for family in families:
            for context_size in context_sizes:
                tasks.append(
                    self.make_task(
                        family=family,
                        split=split,
                        context_size=context_size,
                        resolution=resolution,
                        parameter_holdout=parameter_holdout,
                    )
                )
        return tasks


def _stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % (2**31)


def _grid(resolution: int) -> tuple[float, ...]:
    return tuple(index / (resolution - 1) for index in range(resolution))


def _input_function(points: Sequence[float], rng: random.Random, index: int) -> tuple[Sample, ...]:
    freq = 1 + (index % 3)
    phase = rng.uniform(-0.5, 0.5)
    amp = rng.uniform(0.7, 1.3)
    values = []
    for point in points:
        value = amp * math.sin(2.0 * math.pi * freq * point + phase)
        value += 0.35 * math.cos(math.pi * (freq + 1) * point - phase)
        values.append(Sample(point=point, value=value))
    return tuple(values)


def _sample_parameters(
    rng: random.Random,
    family: str,
    split: str,
    parameter_holdout: bool,
) -> dict[str, Union[float, str]]:
    split_shift = {"train": 0.0, "val": 0.25, "test": 0.5}.get(split, 0.0)
    holdout_shift = 0.7 if parameter_holdout else 0.0
    base_gain = 0.8 + split_shift + holdout_shift + rng.uniform(0.0, 0.2)
    params: dict[str, Union[float, str]] = {"gain": base_gain}
    if family == "fourier":
        params.update({"frequency": 1.0 + split_shift + holdout_shift, "weight_fourier": 1.0})
    elif family == "green":
        params.update({"decay": 2.0 + split_shift + holdout_shift, "weight_local": 1.0})
    elif family == "separable":
        params.update({"rank_weight": 0.6 + split_shift, "weight_separable": 1.0})
    elif family == "nonlinear":
        params.update({"decay": 1.5 + split_shift, "weight_local": 1.0})
    else:
        params.update(
            {
                "weight_fourier": 0.40,
                "weight_separable": 0.35,
                "weight_local": 0.25,
                "frequency": 1.0 + split_shift,
                "decay": 1.5 + holdout_shift,
                "rank_weight": 0.8,
            }
        )
    return params


def _operator_for(family: str, params: dict[str, Union[float, str]]) -> Callable[[Sequence[Sample], float], float]:
    if family == "fourier":
        return lambda samples, query: _fourier_operator(samples, query, params)
    if family == "green":
        return lambda samples, query: _green_operator(samples, query, params)
    if family == "separable":
        return lambda samples, query: _separable_operator(samples, query, params)
    if family == "nonlinear":
        return lambda samples, query: math.tanh(_green_operator(samples, query, params))
    return lambda samples, query: (
        float(params["weight_fourier"]) * _fourier_operator(samples, query, params)
        + float(params["weight_separable"]) * _separable_operator(samples, query, params)
        + float(params["weight_local"]) * _green_operator(samples, query, params)
    )


def _fourier_operator(samples: Sequence[Sample], query: float, params: dict[str, Union[float, str]]) -> float:
    gain = float(params.get("gain", 1.0))
    frequency = float(params.get("frequency", 1.0))
    total = 0.0
    for sample in samples:
        total += sample.value * math.cos(2.0 * math.pi * frequency * (query - sample.point))
    return gain * total / len(samples)


def _green_operator(samples: Sequence[Sample], query: float, params: dict[str, Union[float, str]]) -> float:
    gain = float(params.get("gain", 1.0))
    decay = float(params.get("decay", 2.0))
    weighted_sum = 0.0
    weight_total = 0.0
    for sample in samples:
        weight = math.exp(-decay * abs(query - sample.point))
        weighted_sum += weight * sample.value
        weight_total += weight
    return gain * weighted_sum / weight_total if weight_total else 0.0


def _separable_operator(samples: Sequence[Sample], query: float, params: dict[str, Union[float, str]]) -> float:
    gain = float(params.get("gain", 1.0))
    rank_weight = float(params.get("rank_weight", 0.8))
    mean_value = sum(sample.value for sample in samples) / len(samples)
    first_moment = sum(sample.value * sample.point for sample in samples) / len(samples)
    second_moment = sum(sample.value * math.sin(math.pi * sample.point) for sample in samples) / len(samples)
    return gain * (
        mean_value
        + rank_weight * first_moment * query
        + 0.5 * second_moment * math.sin(math.pi * query)
    )
