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
        use_evidence_router: bool = True,
        router_weight_policy: dict[str, str] | None = None,
        lrio_pairs: tuple[tuple[str, str], ...] | None = None,
    ):
        super().__init__()
        candidate_names = tuple(candidate_names)
        _validate_candidate_subset(candidate_names)
        self.output_dim = output_dim
        self.candidate_names = candidate_names
        self.use_evidence_router = bool(use_evidence_router)
        self.router_weight_policy = _validated_router_weight_policy(router_weight_policy, candidate_names)
        self.evidence_encoder = MultimodalEvidenceEncoder(field_dims=field_dims, query_dim=query_dim, d_model=d_model)
        self.memory_encoder = MultimodalOperatorMemory(d_model=d_model, memory_tokens=memory_tokens, candidate_names=candidate_names)
        self.reliability_prior = (
            RCEOReliabilityPrior(d_model=d_model, candidate_names=candidate_names, lrio_pairs=lrio_pairs or ())
            if use_reliability_prior
            else None
        )
        self.joint_router_adapter = MultimodalJointRouterAdapter(
            d_model=d_model,
            candidate_names=candidate_names,
            use_evidence_router=self.use_evidence_router,
            lrio_pairs=lrio_pairs,
        )
        self.candidate_primitives = make_candidate_bank(
            d_model=d_model,
            output_dim=output_dim,
            candidate_names=candidate_names,
            lrio_pairs=lrio_pairs,
        )

    def forward(
        self,
        batch: MultimodalEpisodeBatch,
        *,
        router_weight_override: torch.Tensor | None = None,
    ) -> MultimodalOVHAOutput:
        batch.model_inputs()
        evidence = self.evidence_encoder(batch)
        memory_bank = self.memory_encoder(evidence.global_features, evidence)
        reliability = self.reliability_prior(batch, evidence) if self.reliability_prior is not None else None
        router_output, params = self.joint_router_adapter(memory_bank, evidence, reliability)
        if reliability is not None and "LRIO" in params:
            params["LRIO"] = {
                **params["LRIO"],
                "pair_reliability": reliability.pair_reliability,
                "pair_names": reliability.pair_names,
            }

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
        candidate_values = stack_candidate_values(candidate_outputs, self.candidate_names)
        operator_admission_gate = _operator_admission_gate(candidate_outputs, self.candidate_names, candidate_values)
        router_weights = _effective_router_weights(
            router_output.weights,
            router_weight_override,
            self.candidate_names,
            self.router_weight_policy,
            operator_admission_gate["tensor"],
        )
        y_hat = (router_weights.unsqueeze(-1) * candidate_values).sum(dim=-2)
        candidate_losses = _candidate_losses(candidate_outputs, batch.target_y, batch.target_mask)
        raw_router_load_by_candidate = router_output.diagnostics.get("router_load_by_candidate", {})
        memory_diagnostics = self._memory_differentiation_diagnostics(
            batch=batch,
            evidence=evidence,
            memory_bank=memory_bank,
            params=params,
            router_weights=router_weights,
            candidate_values=candidate_values,
        )
        diagnostics = {
            **router_output.diagnostics,
            "raw_router_load_by_candidate": raw_router_load_by_candidate,
            "router_load_by_candidate": _router_load_by_candidate(router_weights, self.candidate_names),
            "router_logit_parts": {
                key: value.detach()
                for key, value in router_output.logit_parts.items()
            },
            "candidate_loss": candidate_losses,
            "candidate_loss_by_sample": _candidate_losses_by_sample(candidate_outputs, batch.target_y, batch.target_mask),
            "candidate_value_stats": _candidate_value_stats(candidate_outputs),
            "adapter_params": _adapter_param_diagnostics(params),
            "adapter_params_detail": _adapter_param_details(params),
            "memory_slot_norm": {name: memory_bank[name].norm(dim=-1).mean() for name in self.candidate_names},
            **memory_diagnostics,
            "operator_admission_gate": operator_admission_gate["diagnostics"],
            "candidate_diagnostics": _candidate_diagnostics(candidate_outputs, candidate_losses, reliability),
            "stackability_passed": True,
            "reliability": reliability.diagnostics if reliability is not None else {},
            "router_override": {
                "applied": router_weight_override is not None,
                "source": "training_only_supplied_weights" if router_weight_override is not None else "learned_router",
            },
            "router_weight_policy": _router_weight_policy_diagnostics(self.router_weight_policy),
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

    def _memory_differentiation_diagnostics(
        self,
        *,
        batch: MultimodalEpisodeBatch,
        evidence: MultimodalEvidenceBank,
        memory_bank: dict[str, torch.Tensor],
        params: dict[str, dict[str, torch.Tensor]],
        router_weights: torch.Tensor,
        candidate_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        diagnostics = {
            "memory_slot_orthogonality": _memory_slot_orthogonality(memory_bank, self.candidate_names),
            "memory_zero_out_delta": torch.zeros((), dtype=candidate_values.dtype, device=candidate_values.device),
            "memory_swap_delta": torch.zeros((), dtype=candidate_values.dtype, device=candidate_values.device),
        }
        if self.training:
            return diagnostics
        with torch.no_grad():
            zero_bank = {name: torch.zeros_like(memory_bank[name]) for name in self.candidate_names}
            zero_values = self._candidate_values_with_memory(batch, evidence, zero_bank, params)
            zero_prediction = (router_weights.unsqueeze(-1) * zero_values).sum(dim=-2)
            full_prediction = (router_weights.unsqueeze(-1) * candidate_values).sum(dim=-2)
            diagnostics["memory_zero_out_delta"] = (full_prediction - zero_prediction).abs().mean()
            if len(self.candidate_names) > 1:
                shifted_names = self.candidate_names[1:] + self.candidate_names[:1]
                swap_bank = {
                    name: memory_bank[shifted_names[index]]
                    for index, name in enumerate(self.candidate_names)
                }
                swap_values = self._candidate_values_with_memory(batch, evidence, swap_bank, params)
                swap_prediction = (router_weights.unsqueeze(-1) * swap_values).sum(dim=-2)
                diagnostics["memory_swap_delta"] = (full_prediction - swap_prediction).abs().mean()
        return diagnostics

    def _candidate_values_with_memory(
        self,
        batch: MultimodalEpisodeBatch,
        evidence: MultimodalEvidenceBank,
        memory_bank: dict[str, torch.Tensor],
        params: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        outputs = {
            name: self.candidate_primitives[name](
                batch=batch,
                memory_slot=memory_bank[name],
                evidence=evidence,
                params=params[name],
                output_dim=self.output_dim,
            )
            for name in self.candidate_names
        }
        return stack_candidate_values(outputs, self.candidate_names)


def _effective_router_weights(
    learned_weights: torch.Tensor,
    router_weight_override: torch.Tensor | None,
    candidate_names: tuple[str, ...],
    router_weight_policy: dict[str, str] | None,
    admission_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = (
        learned_weights
        if router_weight_override is None
        else router_weight_override.to(device=learned_weights.device, dtype=learned_weights.dtype)
    )
    if router_weight_policy is None:
        return _apply_admission_gate(weights, admission_gate)
    if "only" in router_weight_policy:
        candidate_index = candidate_names.index(router_weight_policy["only"])
        only = torch.zeros_like(weights)
        only[..., candidate_index] = 1.0
        return _apply_admission_gate(only, admission_gate)
    if "drop" in router_weight_policy:
        candidate_index = candidate_names.index(router_weight_policy["drop"])
        kept = weights.clone()
        kept[..., candidate_index] = 0.0
        kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return _apply_admission_gate(kept, admission_gate)
    return _apply_admission_gate(weights, admission_gate)


def _apply_admission_gate(weights: torch.Tensor, admission_gate: torch.Tensor | None) -> torch.Tensor:
    if admission_gate is None:
        return weights
    gate = admission_gate.to(dtype=weights.dtype, device=weights.device)
    gated = weights * gate
    denom = gated.sum(dim=-1, keepdim=True)
    fallback = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.where(denom > 1e-8, gated / denom.clamp_min(1e-8), fallback)


def _operator_admission_gate(
    candidate_outputs: dict[str, CandidateOutput],
    candidate_names: tuple[str, ...],
    candidate_values: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    gates = []
    diagnostics = {}
    batch_size, query_count = candidate_values.shape[0], candidate_values.shape[1]
    for name in candidate_names:
        output = candidate_outputs[name]
        gate = torch.ones(batch_size, query_count, dtype=candidate_values.dtype, device=candidate_values.device)
        if name == "LRIO":
            pair_count = output.diagnostics.get("pair_count")
            if pair_count is not None and float(pair_count.detach().item()) <= 0.0:
                gate = torch.zeros_like(gate)
        if name == "CATO":
            source_gate = output.diagnostics.get("source_gate", {})
            if isinstance(source_gate, dict) and not source_gate:
                gate = torch.zeros_like(gate)
        gates.append(gate)
        diagnostics[name] = gate.mean()
    return {
        "tensor": torch.stack(gates, dim=-1),
        "diagnostics": diagnostics,
    }


def _router_load_by_candidate(
    weights: torch.Tensor,
    candidate_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return {
        name: weights[..., index].mean()
        for index, name in enumerate(candidate_names)
    }


def _validated_router_weight_policy(
    router_weight_policy: dict[str, str] | None,
    candidate_names: tuple[str, ...],
) -> dict[str, str] | None:
    if router_weight_policy is None:
        return None
    keys = tuple(router_weight_policy)
    if keys not in (("only",), ("drop",)):
        raise ValueError("router_weight_policy must contain exactly one of: only, drop")
    candidate = router_weight_policy[keys[0]]
    if candidate not in candidate_names:
        raise ValueError(f"router_weight_policy candidate must be one of {candidate_names}: {candidate}")
    return dict(router_weight_policy)


def _validate_candidate_subset(candidate_names: tuple[str, ...]) -> None:
    if not candidate_names:
        raise ValueError("MultimodalOVHA requires at least one active candidate")
    allowed = set(MULTIMODAL_CANDIDATE_NAMES)
    invalid = sorted(name for name in candidate_names if name not in allowed)
    if invalid:
        raise ValueError(f"MultimodalOVHA candidates must be TLEO/SPO/LRIO/CATO: {invalid}")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("MultimodalOVHA candidate_names must not contain duplicates")


def _router_weight_policy_diagnostics(router_weight_policy: dict[str, str] | None) -> dict[str, str] | None:
    if router_weight_policy is None:
        return None
    if "only" in router_weight_policy:
        return {"mode": "only", "candidate": router_weight_policy["only"]}
    if "drop" in router_weight_policy:
        return {"mode": "drop", "candidate": router_weight_policy["drop"]}
    return None


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


def _candidate_losses_by_sample(
    candidate_outputs: dict[str, CandidateOutput],
    target_y: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask = target_mask.to(dtype=target_y.dtype, device=target_y.device).unsqueeze(-1)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return {
        name: ((output.value - target_y).square() * mask).sum(dim=-1) / denom
        for name, output in candidate_outputs.items()
    }


def _memory_slot_orthogonality(
    memory_bank: dict[str, torch.Tensor],
    candidate_names: tuple[str, ...],
) -> torch.Tensor:
    if len(candidate_names) < 2:
        sample = next(iter(memory_bank.values()))
        return torch.zeros((), dtype=sample.dtype, device=sample.device)
    vectors = torch.stack([memory_bank[name].mean(dim=1) for name in candidate_names], dim=1)
    flattened = vectors.transpose(0, 1).reshape(len(candidate_names), -1)
    normalized = torch.nn.functional.normalize(flattened, dim=-1)
    gram = normalized @ normalized.transpose(0, 1)
    eye = torch.eye(len(candidate_names), dtype=gram.dtype, device=gram.device)
    return ((gram - eye).square().sum() / max(len(candidate_names) * (len(candidate_names) - 1), 1)).clamp_min(0.0)


def _adapter_param_diagnostics(params: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    diagnostics = {}
    if "TLEO" in params:
        diagnostics["TLEO_lengthscale"] = params["TLEO"]["lengthscale"].mean()
    if "SPO" in params:
        diagnostics["SPO_temperature"] = params["SPO"]["prototype_temperature"].mean()
    if "LRIO" in params:
        diagnostics["LRIO_rank_entropy"] = _entropy(params["LRIO"]["rank_logits"])
        if "rank_logits_by_pair" in params["LRIO"]:
            diagnostics["LRIO_pair_rank_entropy"] = _entropy(params["LRIO"]["rank_logits_by_pair"])
    if "CATO" in params:
        diagnostics["CATO_alignment_temperature"] = params["CATO"]["alignment_temperature"].mean()
    return diagnostics


def _adapter_param_details(params: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            f"{key}_mean": tensor.mean()
            for key, tensor in values.items()
            if hasattr(tensor, "mean")
        }
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


def _candidate_value_stats(candidate_outputs: dict[str, CandidateOutput]) -> dict[str, dict[str, torch.Tensor]]:
    stats = {}
    for name, output in candidate_outputs.items():
        values = output.value.detach()
        abs_values = values.abs().reshape(-1)
        stats[name] = {
            "mean": values.mean(),
            "std": values.std(unbiased=False),
            "min": values.min(),
            "max": values.max(),
            "p95_abs": torch.quantile(abs_values, 0.95) if abs_values.numel() else torch.zeros((), device=values.device),
        }
    return stats
