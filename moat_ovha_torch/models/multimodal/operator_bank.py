from __future__ import annotations

import torch
from torch import nn

from moat_ovha_torch.models.multimodal.primitives.alignment_transport import CATOPrimitive
from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput
from moat_ovha_torch.models.multimodal.primitives.low_rank_interaction import LRIOPrimitive
from moat_ovha_torch.models.multimodal.primitives.semantic_prototype import SPOPrimitive
from moat_ovha_torch.models.multimodal.primitives.typed_local_evidence import TLEOPrimitive


MULTIMODAL_CANDIDATE_NAMES = ("TLEO", "SPO", "LRIO", "CATO")
FORBIDDEN_V1_STACK_NAMES = ("RCEO", "MMRO", "CTRO", "TLDO", "OMRO")


def assert_candidate_names(names: tuple[str, ...] | list[str]) -> None:
    if tuple(names) != MULTIMODAL_CANDIDATE_NAMES:
        raise ValueError("Only TLEO / SPO / LRIO / CATO may enter the v1 candidate stack")


def assert_stackable(outputs: dict[str, CandidateOutput], batch_size: int, q_count: int, dy: int) -> None:
    names = tuple(outputs)
    assert_candidate_names(names)
    for forbidden in FORBIDDEN_V1_STACK_NAMES:
        if forbidden in outputs:
            raise ValueError("Only TLEO / SPO / LRIO / CATO may enter the v1 candidate stack")
    for name, out in outputs.items():
        if tuple(out.value.shape) != (batch_size, q_count, dy):
            raise ValueError(
                f"Candidate {name} returned {tuple(out.value.shape)}, expected {(batch_size, q_count, dy)}. "
                "Only candidate operators may enter router mixture."
            )


def make_candidate_bank(d_model: int, output_dim: int) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "TLEO": TLEOPrimitive(d_model, output_dim),
            "SPO": SPOPrimitive(d_model, output_dim),
            "LRIO": LRIOPrimitive(d_model, output_dim),
            "CATO": CATOPrimitive(d_model, output_dim),
        }
    )


def stack_candidate_values(outputs: dict[str, CandidateOutput]) -> torch.Tensor:
    assert_candidate_names(tuple(outputs))
    return torch.stack([outputs[name].value for name in MULTIMODAL_CANDIDATE_NAMES], dim=-2)
