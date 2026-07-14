"""Device-aware tensor value checks for operator contract boundaries."""

from __future__ import annotations

import os

from torch import Tensor


_DEBUG_CUDA_VALUES = os.getenv(
    "OVHA_DEBUG_CUDA_TENSOR_VALUES", "0"
).strip().lower() in {"1", "true", "yes", "on"}


def tensor_value_checks_enabled(reference: Tensor) -> bool:
    """Keep CPU contract tests strict without synchronizing production CUDA.

    Shape, dtype, and device checks are always cheap host metadata checks.  Full
    tensor reductions are retained on CPU and become opt-in on CUDA because
    converting their result to Python ``bool`` synchronizes every decoder
    layer.
    """
    return reference.device.type != "cuda" or _DEBUG_CUDA_VALUES
