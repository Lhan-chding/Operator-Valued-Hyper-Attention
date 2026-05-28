from __future__ import annotations

import torch


def prediction_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return torch.nn.functional.mse_loss(prediction, target)
    expanded_mask = mask.unsqueeze(-1).to(dtype=prediction.dtype, device=prediction.device)
    squared_error = (prediction - target) ** 2 * expanded_mask
    return squared_error.sum() / expanded_mask.sum().clamp_min(1.0)
