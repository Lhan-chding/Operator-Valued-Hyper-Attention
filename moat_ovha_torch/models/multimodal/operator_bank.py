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
    values = tuple(names)
    if not values:
        raise ValueError("At least one TLEO / SPO / LRIO / CATO candidate must enter the v1 candidate stack")
    allowed = set(MULTIMODAL_CANDIDATE_NAMES)
    invalid = sorted(name for name in values if name not in allowed)
    if invalid:
        raise ValueError(f"Only TLEO / SPO / LRIO / CATO may enter the v1 candidate stack: {invalid}")


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
        if len(out.feature.shape) != 3 or tuple(out.feature.shape[:2]) != (batch_size, q_count):
            raise ValueError(
                f"Candidate {name} feature returned {tuple(out.feature.shape)}, "
                f"expected leading axes {(batch_size, q_count)} for CandidateOutput.feature."
            )


def make_candidate_bank(
    d_model: int,
    output_dim: int,
    candidate_names: tuple[str, ...] = MULTIMODAL_CANDIDATE_NAMES,
    lrio_pairs: tuple[tuple[str, str], ...] | None = None,
) -> nn.ModuleDict:
    assert_candidate_names(candidate_names)
    modules = {}
    for name in candidate_names:
        if name == "TLEO":
            modules[name] = TLEOPrimitive(d_model, output_dim)
        elif name == "SPO":
            modules[name] = SPOPrimitive(d_model, output_dim)
        elif name == "LRIO":
            modules[name] = LRIOPrimitive(d_model, output_dim, pairs=lrio_pairs or LRIOPrimitive.default_pairs())
        elif name == "CATO":
            modules[name] = CATOPrimitive(d_model, output_dim)
    return nn.ModuleDict(modules)


def stack_candidate_values(
    outputs: dict[str, CandidateOutput],
    candidate_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    names = tuple(outputs) if candidate_names is None else tuple(candidate_names)
    assert_candidate_names(names)
    return torch.stack([outputs[name].value for name in names], dim=-2)
