from __future__ import annotations

import torch


def coordinate_scalar(coordinates: torch.Tensor) -> torch.Tensor:
    """Project arbitrary coordinate channels to the scalar channel used by fixed-width heads."""

    if coordinates.shape[-1] == 1:
        return coordinates
    return coordinates.mean(dim=-1, keepdim=True)


def squared_coordinate_distance(query: torch.Tensor, support: torch.Tensor, shift: torch.Tensor | float = 0.0) -> torch.Tensor:
    delta = query.unsqueeze(-2) - support.unsqueeze(1) - shift
    return delta.square().sum(dim=-1, keepdim=True)
