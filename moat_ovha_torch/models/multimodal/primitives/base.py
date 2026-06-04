from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalModelInputs

if TYPE_CHECKING:
    from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank


@dataclass(frozen=True)
class CandidateOutput:
    value: torch.Tensor
    feature: torch.Tensor
    diagnostics: dict[str, Any]


class MultimodalCandidatePrimitive(nn.Module):
    name: str

    def forward(
        self,
        batch: MultimodalModelInputs,
        memory_slot: torch.Tensor,
        evidence: "MultimodalEvidenceBank",
        params: dict[str, torch.Tensor],
        output_dim: int,
    ) -> CandidateOutput:
        raise NotImplementedError


def apply_scale_bias(value: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    scale = params.get("scale")
    bias = params.get("bias")
    if scale is not None:
        value = value * scale
    if bias is not None:
        value = value + bias
    return value
