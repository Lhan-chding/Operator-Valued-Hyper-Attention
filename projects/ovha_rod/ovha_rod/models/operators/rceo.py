from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RCEOResult:
    """Reliability estimates and a centered router log prior."""

    reliability: Tensor
    log_prior: Tensor


class RCEO(nn.Module):
    """Reliability calibration from inference-available quality evidence.

    RCEO intentionally excludes decoder semantics, predicted boxes, matching
    scores, and ground-truth labels.  It learns only from operator
    availability, query validity, text-mask coverage, and visual valid-ratio
    coverage.  This keeps reliability independent from the prediction it is
    calibrating and makes missing evidence an explicit input rather than an
    implicit confidence shortcut.
    """

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
        hidden_dim = max(self.d_model, self.operator_count)
        # Per operator availability plus text/visual coverage and an explicit
        # presence bit for each optional evidence source.
        feature_dim = self.operator_count + 4
        self.features = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, self.operator_count)
        # A new bank is exactly neutral until evidence supports learning a
        # reliability prior.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        operator_available: Tensor,
        valid: Tensor,
        text_valid: Optional[Tensor],
        valid_ratios: Optional[Tensor],
    ) -> RCEOResult:
        batch_size, query_count = self._validate_inputs(
            operator_available, valid, text_valid, valid_ratios)
        dtype = self.output.weight.dtype
        availability = operator_available.to(dtype=dtype)
        text_coverage, text_present = self._text_quality(
            valid, text_valid, dtype)
        visual_coverage, visual_present = self._visual_quality(
            valid, valid_ratios, dtype)
        features = torch.cat(
            (
                availability,
                text_coverage,
                text_present,
                visual_coverage,
                visual_present,
            ),
            dim=-1,
        )
        expected = (batch_size, query_count, self.operator_count + 4)
        if features.shape != expected:
            raise RuntimeError("RCEO feature construction violated its contract")

        raw = self.output(self.features(features))
        reliability = torch.sigmoid(raw)
        centered = raw - raw.mean(dim=-1, keepdim=True)
        bounded = self.prior_cap * torch.tanh(centered / self.prior_cap)
        bounded = bounded - bounded.mean(dim=-1, keepdim=True)
        maximum = bounded.abs().amax(dim=-1, keepdim=True).clamp_min(
            self.prior_cap)
        log_prior = bounded * (self.prior_cap / maximum)
        query_mask = valid[..., None]
        return RCEOResult(
            reliability=torch.where(
                query_mask, reliability, torch.zeros_like(reliability)),
            log_prior=torch.where(
                query_mask, log_prior, torch.zeros_like(log_prior)),
        )

    @staticmethod
    def _text_quality(
        valid: Tensor,
        text_valid: Optional[Tensor],
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        shape = (*valid.shape, 1)
        if text_valid is None:
            return (
                torch.zeros(shape, dtype=dtype, device=valid.device),
                torch.zeros(shape, dtype=dtype, device=valid.device),
            )
        coverage = text_valid.to(dtype=dtype).mean(dim=1, keepdim=True)
        coverage = coverage[:, None, :].expand(shape)
        present = torch.ones(shape, dtype=dtype, device=valid.device)
        return coverage, present

    @staticmethod
    def _visual_quality(
        valid: Tensor,
        valid_ratios: Optional[Tensor],
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        shape = (*valid.shape, 1)
        if valid_ratios is None:
            return (
                torch.zeros(shape, dtype=dtype, device=valid.device),
                torch.zeros(shape, dtype=dtype, device=valid.device),
            )
        coverage = valid_ratios.to(dtype=dtype).prod(dim=-1).mean(
            dim=1, keepdim=True)
        coverage = coverage[:, None, :].expand(shape)
        present = torch.ones(shape, dtype=dtype, device=valid.device)
        return coverage, present

    def _validate_inputs(
        self,
        operator_available: Tensor,
        valid: Tensor,
        text_valid: Optional[Tensor],
        valid_ratios: Optional[Tensor],
    ) -> Tuple[int, int]:
        if (
            operator_available.ndim != 3
            or operator_available.shape[-1] != self.operator_count
        ):
            raise ValueError(
                "operator_available must have shape [B,Q,operator_count]")
        if operator_available.dtype != torch.bool:
            raise ValueError("operator_available must be boolean")
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,Q] tensor")
        if operator_available.shape[:2] != valid.shape:
            raise ValueError(
                "operator_available batch/query axes must match valid")
        batch_queries = valid.shape
        if valid.device != operator_available.device:
            raise ValueError("valid and operator_available must share a device")

        if text_valid is not None:
            if (
                text_valid.ndim != 2
                or text_valid.shape[0] != batch_queries[0]
                or text_valid.shape[1] == 0
            ):
                raise ValueError("text_valid must have shape [B,T] with T > 0")
            if text_valid.dtype != torch.bool:
                raise ValueError("text_valid must be boolean")
            if text_valid.device != operator_available.device:
                raise ValueError("text_valid must share the operator device")

        if valid_ratios is not None:
            if (
                valid_ratios.ndim != 3
                or valid_ratios.shape[0] != batch_queries[0]
                or valid_ratios.shape[1] == 0
                or valid_ratios.shape[-1] != 2
            ):
                raise ValueError(
                    "valid_ratios must have shape [B,L,2] with L > 0")
            if not valid_ratios.is_floating_point():
                raise ValueError("valid_ratios must be floating point")
            if valid_ratios.device != operator_available.device:
                raise ValueError("valid_ratios must share the operator device")
            # Contract tests stay strict on CPU.  CUDA value reductions are
            # guarded at the bank boundary to avoid a host sync per layer.
            if valid_ratios.device.type != "cuda":
                if not torch.isfinite(valid_ratios).all():
                    raise ValueError(
                        "valid_ratios must contain only finite values")
                in_range = (valid_ratios > 0.0) & (valid_ratios <= 1.0)
                if not in_range.all():
                    raise ValueError("valid_ratios values must be in range (0, 1]")

        return int(batch_queries[0]), int(batch_queries[1])
