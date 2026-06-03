from __future__ import annotations

from typing import Any

import torch


def mosei_standard_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    pred, truth = _masked_flat_pair(prediction, target, mask)
    if pred.numel() == 0:
        return _empty_metrics()
    mae = (pred - truth).abs().mean()
    mse = (pred - truth).square().mean()
    acc2_excl0, f1_excl0 = _binary_exclude_zero_metrics(pred, truth)
    acc2_nonneg, f1_nonneg = _binary_nonnegative_metrics(pred, truth)
    return {
        "mae": _as_float(mae),
        "l1_loss": _as_float(mae),
        "mse_loss": _as_float(mse),
        "pearson_correlation": _pearson(pred, truth),
        "acc7": _multiclass_accuracy(pred, truth, lower=-3.0, upper=3.0),
        "acc5": _multiclass_accuracy(pred, truth, lower=-2.0, upper=2.0),
        "acc2_excl0": acc2_excl0,
        "f1_excl0": f1_excl0,
        "acc2_nonneg": acc2_nonneg,
        "f1_nonneg": f1_nonneg,
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "mae": 0.0,
        "l1_loss": 0.0,
        "mse_loss": 0.0,
        "pearson_correlation": 0.0,
        "acc7": 0.0,
        "acc5": 0.0,
        "acc2_excl0": 0.0,
        "f1_excl0": 0.0,
        "acc2_nonneg": 0.0,
        "f1_nonneg": 0.0,
    }


def _masked_flat_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_grid = _scalar_grid(prediction).to(dtype=torch.float32)
    truth_grid = _scalar_grid(target).to(device=prediction.device, dtype=torch.float32)
    valid = _scalar_grid(mask).to(dtype=torch.bool, device=prediction.device)
    pred_grid, truth_grid, valid = torch.broadcast_tensors(pred_grid, truth_grid, valid)
    pred = pred_grid[valid].reshape(-1)
    truth = truth_grid[valid].reshape(-1)
    return pred, truth


def _scalar_grid(value: torch.Tensor) -> torch.Tensor:
    if value.ndim > 1 and int(value.shape[-1]) == 1:
        return value.squeeze(-1)
    return value


def _pearson(pred: torch.Tensor, truth: torch.Tensor) -> float:
    if pred.numel() < 2:
        return 0.0
    pred_centered = pred - pred.mean()
    truth_centered = truth - truth.mean()
    denom = pred_centered.norm() * truth_centered.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, _as_float((pred_centered * truth_centered).sum() / denom)))


def _multiclass_accuracy(pred: torch.Tensor, truth: torch.Tensor, *, lower: float, upper: float) -> float:
    pred_class = pred.clamp(lower, upper).round()
    truth_class = truth.clamp(lower, upper).round()
    return _as_float((pred_class == truth_class).to(dtype=torch.float32).mean())


def _binary_exclude_zero_metrics(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    non_zero = truth != 0
    if not bool(non_zero.any()):
        return 0.0, 0.0
    return _binary_metrics(pred[non_zero] > 0, truth[non_zero] > 0)


def _binary_nonnegative_metrics(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    return _binary_metrics(pred >= 0, truth >= 0)


def _binary_metrics(pred_positive: torch.Tensor, truth_positive: torch.Tensor) -> tuple[float, float]:
    if pred_positive.numel() == 0:
        return 0.0, 0.0
    accuracy = (pred_positive == truth_positive).to(dtype=torch.float32).mean()
    return _as_float(accuracy), _weighted_binary_f1(pred_positive, truth_positive)


def _weighted_binary_f1(pred_positive: torch.Tensor, truth_positive: torch.Tensor) -> float:
    f1_by_class = []
    weights = []
    for positive_class in (False, True):
        pred_class = pred_positive == positive_class
        truth_class = truth_positive == positive_class
        true_positive = (pred_class & truth_class).to(dtype=torch.float32).sum()
        false_positive = (pred_class & ~truth_class).to(dtype=torch.float32).sum()
        false_negative = (~pred_class & truth_class).to(dtype=torch.float32).sum()
        precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
        recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
        support = truth_class.to(dtype=torch.float32).sum()
        f1_by_class.append(f1)
        weights.append(support)
    total_weight = torch.stack(weights).sum().clamp_min(1.0)
    weighted = sum(f1 * weight for f1, weight in zip(f1_by_class, weights)) / total_weight
    return _as_float(weighted)


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)
