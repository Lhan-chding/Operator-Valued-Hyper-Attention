from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch
from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank, MultimodalEvidenceEncoder
from moat_ovha_torch.models.multimodal.joint_router_adapter import MultimodalJointRouterAdapter
from moat_ovha_torch.models.multimodal.memory import MultimodalOperatorMemory
from moat_ovha_torch.models.multimodal.operator_bank import (
    MULTIMODAL_CANDIDATE_NAMES,
    assert_stackable,
    make_candidate_bank,
    stack_candidate_values,
)
from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput
from moat_ovha_torch.models.multimodal.reliability_prior import RCEOReliabilityPrior, ReliabilityPrior


@dataclass(frozen=True)
class MultimodalOVHAOutput:
    y_hat: torch.Tensor
    candidate_values: torch.Tensor
    router_weights: torch.Tensor
    router_logits: torch.Tensor
    router_logit_parts: dict[str, torch.Tensor]
    candidate_outputs: dict[str, CandidateOutput]
    reliability_prior: ReliabilityPrior | None
    diagnostics: dict[str, Any]
    evidence: MultimodalEvidenceBank | None = None


class MultimodalOVHA(nn.Module):
    def __init__(
        self,
        field_dims: dict[str, int],
        query_dim: int,
        output_dim: int,
        d_model: int = 64,
        memory_tokens: int = 4,
        candidate_names: tuple[str, ...] = MULTIMODAL_CANDIDATE_NAMES,
        use_reliability_prior: bool = True,
    ):
        super().__init__()
        if tuple(candidate_names) != MULTIMODAL_CANDIDATE_NAMES:
            raise ValueError("MultimodalOVHA v1 requires exactly TLEO / SPO / LRIO / CATO")
        self.output_dim = output_dim
        self.candidate_names = candidate_names
        self.evidence_encoder = MultimodalEvidenceEncoder(field_dims=field_dims, query_dim=query_dim, d_model=d_model)
        self.memory_encoder = MultimodalOperatorMemory(d_model=d_model, memory_tokens=memory_tokens, candidate_names=candidate_names)
        self.reliability_prior = RCEOReliabilityPrior(d_model=d_model, candidate_names=candidate_names) if use_reliability_prior else None
        self.joint_router_adapter = MultimodalJointRouterAdapter(d_model=d_model, candidate_names=candidate_names)
        self.candidate_primitives = make_candidate_bank(d_model=d_model, output_dim=output_dim)

    def forward(
        self,
        batch: MultimodalEpisodeBatch,
        *,
        router_weight_override: torch.Tensor | None = None,
    ) -> MultimodalOVHAOutput:
        batch.model_inputs()
        evidence = self.evidence_encoder(batch)
        memory_bank = self.memory_encoder(evidence.global_features)
        reliability = self.reliability_prior(batch, evidence) if self.reliability_prior is not None else None
        router_output, params = self.joint_router_adapter(memory_bank, evidence, reliability)

        candidate_outputs: dict[str, CandidateOutput] = {}
        for name in self.candidate_names:
            candidate_outputs[name] = self.candidate_primitives[name](
                batch=batch,
                memory_slot=memory_bank[name],
                evidence=evidence,
                params=params[name],
                output_dim=self.output_dim,
            )
        batch_size, q_count = batch.target_y.shape[0], batch.target_y.shape[1]
        assert_stackable(candidate_outputs, batch_size, q_count, self.output_dim)
        candidate_values = stack_candidate_values(candidate_outputs)
        router_weights = _effective_router_weights(router_output.weights, router_weight_override)
        y_hat = (router_weights.unsqueeze(-1) * candidate_values).sum(dim=-2)
        candidate_losses = _candidate_losses(candidate_outputs, batch.target_y, batch.target_mask)
        diagnostics = {
            **router_output.diagnostics,
            "router_logit_parts": {
                key: value.detach()
                for key, value in router_output.logit_parts.items()
            },
            "candidate_loss": candidate_losses,
            "adapter_params": _adapter_param_diagnostics(params),
            "adapter_params_detail": _adapter_param_details(params),
            "memory_slot_norm": {name: memory_bank[name].norm(dim=-1).mean() for name in self.candidate_names},
            "candidate_diagnostics": _candidate_diagnostics(candidate_outputs, candidate_losses, reliability),
            "stackability_passed": True,
            "reliability": reliability.diagnostics if reliability is not None else {},
            "router_override": {
                "applied": router_weight_override is not None,
                "source": "training_only_supplied_weights" if router_weight_override is not None else "learned_router",
            },
        }
        return MultimodalOVHAOutput(
            y_hat=y_hat,
            candidate_values=candidate_values,
            router_weights=router_weights,
            router_logits=router_output.logits,
            router_logit_parts=router_output.logit_parts,
            candidate_outputs=candidate_outputs,
            reliability_prior=reliability,
            diagnostics=diagnostics,
            evidence=evidence,
        )


def _effective_router_weights(
    learned_weights: torch.Tensor,
    router_weight_override: torch.Tensor | None,
) -> torch.Tensor:
    if router_weight_override is None:
        return learned_weights
    return router_weight_override.to(device=learned_weights.device, dtype=learned_weights.dtype)


def _candidate_losses(
    candidate_outputs: dict[str, CandidateOutput],
    target_y: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask = target_mask.to(dtype=target_y.dtype, device=target_y.device).unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return {
        name: ((output.value - target_y).square() * mask).sum() / denom
        for name, output in candidate_outputs.items()
    }


def _adapter_param_diagnostics(params: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "TLEO_lengthscale": params["TLEO"]["lengthscale"].mean(),
        "SPO_temperature": params["SPO"]["prototype_temperature"].mean(),
        "LRIO_rank_entropy": _entropy(params["LRIO"]["rank_logits"]),
        "CATO_alignment_temperature": params["CATO"]["alignment_temperature"].mean(),
    }


def _adapter_param_details(params: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {f"{key}_mean": tensor.mean() for key, tensor in values.items()}
        for name, values in params.items()
    }


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()


def _candidate_diagnostics(
    candidate_outputs: dict[str, CandidateOutput],
    candidate_losses: dict[str, torch.Tensor],
    reliability: ReliabilityPrior | None,
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for name, output in candidate_outputs.items():
        values = dict(output.diagnostics)
        values["candidate_loss"] = candidate_losses[name]
        diagnostics[name] = values
    diagnostics["RCEO"] = dict(reliability.diagnostics) if reliability is not None else {}
    return diagnostics
