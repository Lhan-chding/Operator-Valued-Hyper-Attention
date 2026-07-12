from __future__ import annotations

import torch
from torch import Tensor


def refexp_box_metrics(prediction_cxcywh: Tensor, gt_cxcywh: Tensor,
                       gt_valid_mask: Tensor) -> dict[str, float]:
    """Compute Top-1 Acc@.5/.75 and mIoU for one referent per sample."""
    if prediction_cxcywh.ndim != 2 or prediction_cxcywh.shape[-1] != 4:
        raise ValueError("predictions must have shape [B,4]")
    iou = _max_iou(prediction_cxcywh[:, None], gt_cxcywh, gt_valid_mask)[:, 0]
    return {
        "acc_0.5": float((iou >= 0.5).float().mean()),
        "acc_0.75": float((iou >= 0.75).float().mean()),
        "miou": float(iou.mean()),
    }


def query_oracle_metrics(selected_cxcywh: Tensor, gt_cxcywh: Tensor,
                         gt_valid_mask: Tensor) -> dict[str, float]:
    """Selected-Top-K oracle metrics used by the RQGO scientific gate."""
    if selected_cxcywh.ndim != 3 or selected_cxcywh.shape[-1] != 4:
        raise ValueError("selected queries must have shape [B,K,4]")
    best = _max_iou(selected_cxcywh, gt_cxcywh, gt_valid_mask).amax(dim=1)
    return {
        "oracle_0.5": float((best >= 0.5).float().mean()),
        "oracle_0.75": float((best >= 0.75).float().mean()),
        "max_iou": float(best.mean()),
    }


def _max_iou(proposals: Tensor, targets: Tensor, target_valid: Tensor) -> Tensor:
    if targets.ndim != 3 or targets.shape[-1] != 4:
        raise ValueError("targets must have shape [B,G,4]")
    if target_valid.shape != targets.shape[:2]:
        raise ValueError("target mask must have shape [B,G]")
    if proposals.shape[0] != targets.shape[0]:
        raise ValueError("proposal and target batch sizes differ")
    if targets.shape[1] == 0:
        return proposals.new_zeros(proposals.shape[:2])
    proposal_xyxy = _cxcywh_to_xyxy(proposals)
    target_xyxy = _cxcywh_to_xyxy(targets)
    top_left = torch.maximum(
        proposal_xyxy[:, :, None, :2], target_xyxy[:, None, :, :2])
    bottom_right = torch.minimum(
        proposal_xyxy[:, :, None, 2:], target_xyxy[:, None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(-1)
    proposal_area = (proposal_xyxy[..., 2:] - proposal_xyxy[..., :2]).clamp_min(0.0).prod(-1)
    target_area = (target_xyxy[..., 2:] - target_xyxy[..., :2]).clamp_min(0.0).prod(-1)
    union = proposal_area[:, :, None] + target_area[:, None, :] - intersection
    iou = intersection / union.clamp_min(1e-8)
    iou = iou.masked_fill(~target_valid[:, None].to(dtype=torch.bool), 0.0)
    return iou.amax(dim=-1)


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center, size = boxes[..., :2], boxes[..., 2:].clamp_min(0.0)
    return torch.cat((center - size / 2.0, center + size / 2.0), dim=-1)
