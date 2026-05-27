from __future__ import annotations

import torch


def prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(prediction, target)
