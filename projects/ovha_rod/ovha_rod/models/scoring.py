from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def masked_expression_score(token_logits: Tensor, text_valid_mask: Tensor,
                            temperature: float = 1.0) -> Tensor:
    """Smooth expression score for the shared referent-head control.

    This function is deliberately not used for encoder Top-K; the parent token
    max is retained there for exact zero-initialization equivalence.
    """
    if token_logits.ndim != 3 or text_valid_mask.shape != (
            token_logits.shape[0], token_logits.shape[2]):
        raise ValueError("expected logits [B,Q,T] and mask [B,T]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = text_valid_mask.to(dtype=torch.bool)
    if (~valid.any(dim=1)).any():
        raise ValueError("each sample needs at least one valid token")
    scaled = token_logits / temperature
    scaled = scaled.masked_fill(~valid[:, None], float("-inf"))
    return torch.logsumexp(scaled, dim=-1) * temperature


def referent_focal_loss(scores: Tensor, positive_mask: Tensor,
                        valid_mask: Tensor | None = None,
                        alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    if scores.shape != positive_mask.shape:
        raise ValueError("scores and positive mask must have the same shape")
    valid = (torch.ones_like(positive_mask, dtype=torch.bool)
             if valid_mask is None else valid_mask.to(dtype=torch.bool))
    target = positive_mask.to(dtype=scores.dtype)
    bce = F.binary_cross_entropy_with_logits(scores, target, reduction="none")
    probability = scores.sigmoid()
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_factor = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = (alpha_factor * (1.0 - pt).pow(gamma) * bce).masked_fill(~valid, 0.0)
    return loss.sum() / valid.sum().clamp_min(1)
