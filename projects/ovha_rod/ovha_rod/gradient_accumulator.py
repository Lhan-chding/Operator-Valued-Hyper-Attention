from __future__ import annotations

import torch
from torch import Tensor


def accumulate_gradient_square(
    accumulated: Tensor | None, gradient: Tensor
) -> Tensor:
    """Return an on-device detached sum of squared FP32 gradients."""
    if not isinstance(gradient, Tensor):
        raise TypeError("gradient must be a tensor")
    squared = gradient.detach().float().square().sum()
    if accumulated is None:
        return squared
    if not isinstance(accumulated, Tensor) or accumulated.numel() != 1:
        raise ValueError("accumulated gradient square must be a scalar tensor")
    if accumulated.device != squared.device:
        raise ValueError("accumulator and gradient must share a device")
    return accumulated + squared


def finalize_gradient_norm(accumulated: Tensor | None) -> float:
    """Synchronize one scalar and return its L2 norm."""
    if accumulated is None:
        return 0.0
    if not isinstance(accumulated, Tensor) or accumulated.numel() != 1:
        raise ValueError("accumulated gradient square must be a scalar tensor")
    return float(accumulated.detach().float().clamp_min(0.0).sqrt().cpu())
