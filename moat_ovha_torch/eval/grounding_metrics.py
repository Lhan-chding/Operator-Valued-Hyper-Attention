from __future__ import annotations

from typing import Any

import torch


METRICS_SOURCE = "canonical_grounding_metrics_v1"


def grounding_candidate_metrics(
    logits: torch.Tensor,
    region_targets: Any,
    candidate_boxes: Any,
    bbox_targets: Any,
    mask: Any,
) -> dict[str, float | str]:
    if candidate_boxes is None:
        raise ValueError("candidate_region_boxes are required for canonical grounding metrics")
    if bbox_targets is None:
        raise ValueError("bbox_targets are required for canonical grounding metrics")
    if region_targets is None:
        raise ValueError("region_targets are required for canonical grounding metrics")
    if logits.ndim != 3 or logits.shape[-1] <= 1:
        raise ValueError("grounding logits must have shape [B,Q,N] with N > 1")

    labels = _labels(region_targets, logits)
    valid = _valid_mask(mask, logits, labels)
    if not bool(valid.any()):
        return {
            "metrics_source": METRICS_SOURCE,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "acc_at_0_5": 0.0,
            "mean_iou": 0.0,
            "cross_entropy": 0.0,
            "mrr": 0.0,
        }

    scoped_logits = logits[:, : labels.shape[1], :]
    top1 = scoped_logits.argmax(dim=-1)
    top5 = torch.topk(scoped_logits, k=min(5, scoped_logits.shape[-1]), dim=-1).indices
    iou = _selected_candidate_iou(scoped_logits, candidate_boxes, bbox_targets, valid)
    ce = torch.nn.functional.cross_entropy(
        scoped_logits.reshape(-1, scoped_logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)
    ranks = _target_ranks(scoped_logits, labels)
    return {
        "metrics_source": METRICS_SOURCE,
        "recall_at_1": _as_float((top1[valid] == labels[valid]).to(dtype=torch.float32).mean()),
        "recall_at_5": _as_float((top5[valid] == labels[valid].unsqueeze(-1)).any(dim=-1).to(dtype=torch.float32).mean()),
        "acc_at_0_5": _as_float((iou >= 0.5).to(dtype=torch.float32).mean()),
        "mean_iou": _as_float(iou.mean()),
        "cross_entropy": _as_float(ce[valid].mean()),
        "mrr": _as_float((1.0 / ranks[valid].to(dtype=torch.float32)).mean()),
    }


def _labels(region_targets: Any, logits: torch.Tensor) -> torch.Tensor:
    labels = region_targets.to(device=logits.device, dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.ndim > 2:
        labels = labels.reshape(labels.shape[0], -1)
    if labels.shape[1] == 1 and logits.shape[1] > 1:
        labels = labels.expand(-1, logits.shape[1])
    labels = labels[:, : logits.shape[1]].contiguous()
    if bool(((labels < 0) | (labels >= logits.shape[-1])).any()):
        raise ValueError("region_targets must point inside candidate logits")
    return labels


def _valid_mask(mask: Any, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool, device=logits.device)
    if valid.ndim != 2:
        valid = valid.reshape(valid.shape[0], -1)
    if valid.shape[1] == 1 and labels.shape[1] > 1:
        valid = valid.expand(-1, labels.shape[1])
    return valid[:, : labels.shape[1]]


def _selected_candidate_iou(
    logits: torch.Tensor,
    candidate_boxes: Any,
    bbox_targets: Any,
    valid: torch.Tensor,
) -> torch.Tensor:
    boxes = candidate_boxes.to(device=logits.device, dtype=torch.float32)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("candidate_region_boxes must have shape [B,N,4]")
    truth = bbox_targets.to(device=logits.device, dtype=torch.float32)
    if truth.ndim == 2:
        truth = truth.unsqueeze(1)
    if truth.shape[1] == 1 and logits.shape[1] > 1:
        truth = truth.expand(-1, logits.shape[1], -1)
    truth = truth[:, : logits.shape[1], :]
    gather_index = logits.argmax(dim=-1).unsqueeze(-1).expand(-1, -1, 4)
    pred_boxes = torch.gather(boxes, dim=1, index=gather_index)
    return _box_iou(pred_boxes[valid], truth[valid])


def _target_ranks(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target_scores = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return (logits > target_scores.unsqueeze(-1)).sum(dim=-1) + 1


def _box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(left[..., 0], right[..., 0])
    y1 = torch.maximum(left[..., 1], right[..., 1])
    x2 = torch.minimum(left[..., 2], right[..., 2])
    y2 = torch.minimum(left[..., 3], right[..., 3])
    inter = (x2 - x1).clamp_min(0.0) * (y2 - y1).clamp_min(0.0)
    left_area = (left[..., 2] - left[..., 0]).clamp_min(0.0) * (left[..., 3] - left[..., 1]).clamp_min(0.0)
    right_area = (right[..., 2] - right[..., 0]).clamp_min(0.0) * (right[..., 3] - right[..., 1]).clamp_min(0.0)
    return inter / (left_area + right_area - inter).clamp_min(1e-8)


def _as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
