from __future__ import annotations

import torch


def primitive_load(weights: torch.Tensor, primitive_names: tuple[str, ...]) -> dict[str, float]:
    loads = weights.mean(dim=(0, 1)).detach().cpu()
    return {name: float(loads[index]) for index, name in enumerate(primitive_names)}
