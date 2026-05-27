from __future__ import annotations

import math
from typing import Sequence


def mse(predictions: Sequence[float], targets: Sequence[float]) -> float:
    _check_lengths(predictions, targets)
    if not predictions:
        return 0.0
    return math.fsum((pred - target) ** 2 for pred, target in zip(predictions, targets)) / len(predictions)


def relative_l2(predictions: Sequence[float], targets: Sequence[float]) -> float:
    _check_lengths(predictions, targets)
    numerator = math.sqrt(math.fsum((pred - target) ** 2 for pred, target in zip(predictions, targets)))
    denominator = math.sqrt(math.fsum(target * target for target in targets))
    return numerator / max(denominator, 1e-12)


def _check_lengths(predictions: Sequence[float], targets: Sequence[float]) -> None:
    if len(predictions) != len(targets):
        raise ValueError("prediction and target lengths must match")
