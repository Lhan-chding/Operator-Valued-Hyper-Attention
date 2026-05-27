from __future__ import annotations

import torch


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.norm((prediction - target).reshape(prediction.shape[0], -1), dim=-1)
    denominator = torch.linalg.norm(target.reshape(target.shape[0], -1), dim=-1).clamp_min(1e-8)
    return numerator / denominator


def summarize_relative_l2(values: torch.Tensor) -> dict[str, float]:
    sorted_values = torch.sort(values.detach().cpu()).values
    index_90 = min(len(sorted_values) - 1, int(0.9 * max(len(sorted_values) - 1, 0)))
    return {
        "relative_l2": float(values.mean().detach().cpu()),
        "median_relative_l2": float(sorted_values.median()),
        "p90_relative_l2": float(sorted_values[index_90]),
    }
