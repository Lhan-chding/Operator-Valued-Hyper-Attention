from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F


def build_seed_quality_targets(proposal_boxes_cxcywh: Tensor,
                               gt_boxes_cxcywh: Tensor,
                               gt_valid_mask: Tensor,
                               gamma: float = 1.0) -> Tensor:
    """Build detached max-IoU soft targets for dense encoder proposals."""
    if proposal_boxes_cxcywh.ndim != 3 or proposal_boxes_cxcywh.shape[-1] != 4:
        raise ValueError("proposal boxes must have shape [B,N,4]")
    if gt_boxes_cxcywh.ndim != 3 or gt_boxes_cxcywh.shape[-1] != 4:
        raise ValueError("GT boxes must have shape [B,G,4]")
    if gt_valid_mask.shape != gt_boxes_cxcywh.shape[:2]:
        raise ValueError("GT valid mask must have shape [B,G]")
    if proposal_boxes_cxcywh.shape[0] != gt_boxes_cxcywh.shape[0]:
        raise ValueError("proposal and GT batch sizes differ")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if gt_boxes_cxcywh.shape[1] == 0:
        return proposal_boxes_cxcywh.new_zeros(proposal_boxes_cxcywh.shape[:2])
    proposal = _cxcywh_to_xyxy(proposal_boxes_cxcywh.detach())
    target = _cxcywh_to_xyxy(gt_boxes_cxcywh.detach())
    top_left = torch.maximum(proposal[:, :, None, :2], target[:, None, :, :2])
    bottom_right = torch.minimum(proposal[:, :, None, 2:], target[:, None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(-1)
    proposal_area = (proposal[..., 2:] - proposal[..., :2]).clamp_min(0.0).prod(-1)
    target_area = (target[..., 2:] - target[..., :2]).clamp_min(0.0).prod(-1)
    union = proposal_area[:, :, None] + target_area[:, None, :] - intersection
    iou = intersection / union.clamp_min(1e-8)
    iou = iou.masked_fill(~gt_valid_mask[:, None].to(dtype=torch.bool), 0.0)
    return iou.amax(dim=-1).clamp(0.0, 1.0).pow(gamma).detach()


def quality_focal_seed_loss(selection_logits: Tensor, quality_targets: Tensor,
                            valid_mask: Tensor, beta: float = 2.0,
                            positive_threshold: float = 0.1) -> Tensor:
    if selection_logits.shape != quality_targets.shape or valid_mask.shape != selection_logits.shape:
        raise ValueError("seed logits, targets and valid mask must share [B,N]")
    valid = valid_mask.to(dtype=torch.bool) & torch.isfinite(selection_logits)
    targets = quality_targets.detach().clamp(0.0, 1.0)
    element = F.binary_cross_entropy_with_logits(
        selection_logits, targets, reduction="none")
    modulation = (selection_logits.sigmoid() - targets).abs().pow(beta)
    element = (element * modulation).masked_fill(~valid, 0.0)
    positive_count = ((targets >= positive_threshold) & valid).sum().to(
        dtype=element.dtype)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(positive_count, op=dist.ReduceOp.SUM)
        positive_count = positive_count / dist.get_world_size()
    return element.sum() / positive_count.clamp_min(1.0)


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center, size = boxes[..., :2], boxes[..., 2:].clamp_min(0.0)
    return torch.cat((center - size / 2.0, center + size / 2.0), dim=-1)
