from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXPECTED_STAGE_SEQUENCES = {
    "controlled_multimodal": ("T0", "T1", "T2", "T3", "T4"),
    "phrase_region_grounding": ("T0", "T5"),
    "sentiment_emotion": ("T0", "T5"),
    "robustness_eval": ("T0", "T6"),
}

ALLOWED_V1_ADAPTER_PARAMS = {
    "TLEO": ("lengthscale", "local_temperature", "scale", "bias"),
    "SPO": ("prototype_temperature", "prototype_logits_shift", "scale", "bias"),
    "LRIO": ("rank_logits", "interaction_temperature", "scale", "bias"),
    "CATO": ("alignment_temperature", "transport_scale", "scale", "bias"),
}

FORBIDDEN_V1_ADAPTER_PARAMS = (
    "dynamic_sinkhorn_epsilon",
    "dynamic_null_policy",
    "repair_variance",
    "conflict_threshold",
    "lag_centers",
    "deformable_geometry_warp",
)

HIDDEN_CONTROLLED_ONLY_LOSSES = (
    "router_ce_true_active_operator",
    "adapter_kl_true_params",
    "true_alignment_ce",
    "cato_true_alignment_ce",
    "lrio_rank_kl",
    "spo_prototype_kl",
    "tleo_lengthscale_huber",
    "rceo_reliability_huber",
)

PUBLIC_ALLOWED_LOSSES = (
    "cache_validation",
    "task_loss",
    "candidate_individual_loss",
    "public_alignment_ce",
    "public_contrastive_retrieval",
    "weak_rceo_unimodal_disagreement_marked",
    "weak_modality_dropout_consistency_marked",
)

CONTROLLED_ALLOWED_EXTRA_LOSSES = HIDDEN_CONTROLLED_ONLY_LOSSES + (
    "cache_validation",
    "task_loss",
    "candidate_individual_loss",
    "cato_alignment_ce",
)

ROBUSTNESS_ALLOWED_LOSSES = ("cache_validation", "robustness_evaluation_only", "task_loss")


@dataclass(frozen=True)
class TrainingProtocolReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_training_protocol(plan: dict[str, Any]) -> TrainingProtocolReport:
    errors: list[str] = []
    warnings: list[str] = []
    task_type = str(plan.get("task_type", ""))
    stages = tuple(plan.get("training_stages", ()))
    _validate_stage_sequence(task_type, stages, errors)
    losses_by_stage = plan.get("losses_by_stage", {})
    _validate_losses(task_type, losses_by_stage, errors)
    _validate_adapter_params(plan.get("adapter_params_by_candidate", {}), errors)
    return TrainingProtocolReport(ok=not errors, errors=errors, warnings=warnings)


def _validate_stage_sequence(task_type: str, stages: tuple[str, ...], errors: list[str]) -> None:
    expected = EXPECTED_STAGE_SEQUENCES.get(task_type)
    if expected is None:
        errors.append(f"unknown multimodal training task_type: {task_type}")
        return
    if stages != expected:
        errors.append(f"{task_type} training_stages must be {expected}, got {stages}")


def _validate_losses(task_type: str, losses_by_stage: dict[str, list[str]], errors: list[str]) -> None:
    for stage, losses in sorted(losses_by_stage.items()):
        for loss in losses:
            if task_type == "controlled_multimodal":
                if loss not in CONTROLLED_ALLOWED_EXTRA_LOSSES:
                    errors.append(f"unknown controlled loss {loss} in {stage}")
            elif task_type == "robustness_eval":
                if loss not in ROBUSTNESS_ALLOWED_LOSSES:
                    errors.append(f"robustness stage does not train hidden/public auxiliary loss: {loss}")
            else:
                if loss in HIDDEN_CONTROLLED_ONLY_LOSSES or loss.startswith("true_"):
                    errors.append(f"hidden loss is controlled-only and forbidden for public data: {loss} in {stage}")
                elif loss == "rceo_unimodal_disagreement":
                    errors.append("weak reliability loss must be explicitly marked: use weak_rceo_unimodal_disagreement_marked")
                elif loss not in PUBLIC_ALLOWED_LOSSES:
                    errors.append(f"unknown or unmarked public loss {loss} in {stage}")


def _validate_adapter_params(params_by_candidate: dict[str, list[str]], errors: list[str]) -> None:
    for candidate, allowed in ALLOWED_V1_ADAPTER_PARAMS.items():
        params = tuple(params_by_candidate.get(candidate, ()))
        if not params:
            errors.append(f"adapter params missing for {candidate}")
            continue
        for param in params:
            if param in FORBIDDEN_V1_ADAPTER_PARAMS:
                errors.append(f"v1 adapter param is forbidden: {candidate}.{param}")
            elif param not in allowed:
                errors.append(f"unknown v1 adapter param for {candidate}: {param}")
