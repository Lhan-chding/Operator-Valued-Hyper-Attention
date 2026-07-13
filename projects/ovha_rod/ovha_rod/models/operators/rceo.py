from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .tensor_validation import tensor_value_checks_enabled


@dataclass(frozen=True)
class RCEOResult:
    """Inference-only reliability and centered router log prior."""

    reliability: Tensor
    log_prior: Tensor


class RCEO(nn.Module):
    """Reliability calibration from decoder state, boxes, and score history."""

    def __init__(
        self, d_model: int, operator_count: int, prior_cap: float = 2.0
    ) -> None:
        super().__init__()
        if d_model <= 0 or operator_count <= 0:
            raise ValueError("RCEO dimensions must be positive")
        if prior_cap <= 0:
            raise ValueError("prior_cap must be positive")
        self.d_model = int(d_model)
        self.operator_count = int(operator_count)
        self.prior_cap = float(prior_cap)
        hidden_dim = max(d_model, operator_count)
        self.query_norm = nn.LayerNorm(d_model)
        self.features = nn.Sequential(
            nn.Linear(d_model + 5, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, operator_count)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        query: Tensor,
        boxes: Tensor,
        referent_score: Tensor,
        valid: Tensor,
    ) -> RCEOResult:
        self._validate_inputs(query, boxes, referent_score, valid)
        features = torch.cat(
            (self.query_norm(query), boxes, referent_score[..., None]), dim=-1)
        raw = self.output(self.features(features))
        reliability = torch.sigmoid(raw)
        centered = raw - raw.mean(dim=-1, keepdim=True)
        bounded = self.prior_cap * torch.tanh(centered / self.prior_cap)
        bounded = bounded - bounded.mean(dim=-1, keepdim=True)
        maximum = bounded.abs().amax(dim=-1, keepdim=True).clamp_min(
            self.prior_cap)
        log_prior = bounded * (self.prior_cap / maximum)
        mask = valid[..., None]
        return RCEOResult(
            reliability=torch.where(
                mask, reliability, torch.zeros_like(reliability)),
            log_prior=torch.where(mask, log_prior, torch.zeros_like(log_prior)),
        )

    def _validate_inputs(
        self,
        query: Tensor,
        boxes: Tensor,
        referent_score: Tensor,
        valid: Tensor,
    ) -> None:
        if query.ndim != 3 or query.shape[-1] != self.d_model:
            raise ValueError("query must have shape [B,Q,D]")
        batch_queries = query.shape[:2]
        if boxes.shape != (*batch_queries, 4):
            raise ValueError("boxes must have shape [B,Q,4]")
        if referent_score.shape != batch_queries:
            raise ValueError("referent_score must have shape [B,Q]")
        if valid.shape != batch_queries or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,Q] tensor")
        tensors = (query, boxes, referent_score)
        if any(not value.is_floating_point() for value in tensors):
            raise ValueError("RCEO inputs must be floating point")
        if any(value.device != query.device or value.dtype != query.dtype
               for value in tensors[1:]):
            raise ValueError("RCEO tensors must share device and dtype")
        if valid.device != query.device:
            raise ValueError("valid must share the query device")
        if (tensor_value_checks_enabled(query)
                and any(not torch.isfinite(value).all() for value in tensors)):
            raise ValueError("RCEO inputs must contain only finite values")
