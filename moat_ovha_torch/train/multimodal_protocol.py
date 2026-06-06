from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CONTROLLED_TASK_TYPES = ("controlled_multimodal", "controlled_relation_operator")
REGION_TEXT_TASK_TYPES = (
    "phrase_region_grounding",
    "region_text_grounding",
    "refcoco",
    "flickr30k_entities",
    "visual_genome",
)
SENTIMENT_EMOTION_TASK_TYPES = (
    "sentiment_emotion",
    "sentiment_regression",
    "emotion_classification",
    "cmu_mosei",
    "cmu_mosi",
    "meld",
    "iemocap",
)


EXPECTED_STAGE_SEQUENCES = {
    **{task_type: ("T0", "T1", "T2", "T3", "T4") for task_type in CONTROLLED_TASK_TYPES},
    **{task_type: ("T0", "T5") for task_type in REGION_TEXT_TASK_TYPES},
    **{task_type: ("T0", "T5") for task_type in SENTIMENT_EMOTION_TASK_TYPES},
    "robustness_eval": ("T0", "T6"),
}

CONTROLLED_REQUIRED_STAGE_LOSSES = {
    "T0": ("cache_validation",),
    "T1": ("task_loss", "candidate_individual_loss"),
    "T2": ("task_loss", "router_ce_true_active_operator", "adapter_kl_true_params"),
    "T3": ("task_loss", "router_ce_true_active_operator"),
    "T4": (
        "task_loss",
        "cato_alignment_ce",
        "lrio_rank_kl",
        "spo_prototype_kl",
        "tleo_lengthscale_huber",
        "rceo_reliability_huber",
    ),
}
PUBLIC_REQUIRED_STAGE_LOSSES = {
    "T0": ("cache_validation",),
    "T5": ("task_loss",),
}
ROBUSTNESS_REQUIRED_STAGE_LOSSES = {
    "T0": ("cache_validation",),
    "T6": ("robustness_evaluation_only",),
}
REQUIRED_STAGE_LOSSES = {
    **{task_type: CONTROLLED_REQUIRED_STAGE_LOSSES for task_type in CONTROLLED_TASK_TYPES},
    **{task_type: PUBLIC_REQUIRED_STAGE_LOSSES for task_type in REGION_TEXT_TASK_TYPES},
    **{task_type: PUBLIC_REQUIRED_STAGE_LOSSES for task_type in SENTIMENT_EMOTION_TASK_TYPES},
    "robustness_eval": ROBUSTNESS_REQUIRED_STAGE_LOSSES,
}

TANSO_ALLOWED_ADAPTER_PARAMS = (
    "audio_shift_scale",
    "vision_shift_scale",
    "shift_temperature",
    "audio_lag_logits",
    "vision_lag_logits",
    "lag_width",
    "temporal_temperature",
    "scale",
    "bias",
)

ALLOWED_V1_ADAPTER_PARAMS = {
    "PRSO": ("alignment_temperature", "scale", "bias"),
    "SRO": ("scale", "bias"),
    "TLEO": ("lengthscale", "local_temperature", "scale", "bias"),
    "SPO": ("prototype_temperature", "prototype_logits_shift", "scale", "bias"),
    "LRIO": ("rank_logits", "rank_logits_by_pair", "interaction_temperature", "interaction_temperature_by_pair", "scale", "bias"),
    "CATO": ("alignment_temperature", "transport_scale", "scale", "bias"),
    "TANSO": TANSO_ALLOWED_ADAPTER_PARAMS,
    "TANSOBase": TANSO_ALLOWED_ADAPTER_PARAMS,
    "TANSOShift": TANSO_ALLOWED_ADAPTER_PARAMS,
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
    "active_operator_ce",
    "router_ce_active_operator",
    "router_ce_true_active_operator",
    "true_active_operator_ce",
    "adapter_kl_true_params",
    "adapter_huber_true_params",
    "true_adapter_params_kl",
    "true_adapter_params_huber",
    "true_alignment_ce",
    "true_alignment_kl",
    "cato_alignment_ce",
    "cato_alignment_kl",
    "cato_true_alignment_ce",
    "cato_true_alignment_kl",
    "lrio_rank_kl",
    "true_rank_logits_kl",
    "spo_prototype_kl",
    "true_prototype_logits_kl",
    "tleo_lengthscale_huber",
    "tleo_log_lengthscale_huber",
    "true_lengthscale_huber",
    "rceo_reliability_huber",
    "rceo_reliability_monotonic_loss",
    "true_reliability_huber",
)

PUBLIC_ALLOWED_LOSSES = (
    "cache_validation",
    "task_loss",
    "candidate_individual_loss",
    "public_alignment_ce",
    "public_contrastive_retrieval",
    "spo_prototype_diversity",
    "router_marginal_utility",
    "residual_norm_shrinkage",
    "huber_l1_task_loss",
    "ordinal_acc5_acc7_auxiliary",
    "residual_gate_utility_loss",
    "residual_oracle_gate_loss",
    "tanso_source_oracle_gate_loss",
    "val_affine_calibration",
    "weak_rceo_unimodal_disagreement_marked",
    "weak_modality_dropout_consistency_marked",
    "weak_unimodal_entropy_calibration_marked",
    "weak_cross_modal_disagreement_marked",
)
PUBLIC_MARKED_WEAK_LOSSES = (
    "weak_rceo_unimodal_disagreement_marked",
    "weak_modality_dropout_consistency_marked",
    "weak_unimodal_entropy_calibration_marked",
    "weak_cross_modal_disagreement_marked",
)

CONTROLLED_ALLOWED_EXTRA_LOSSES = HIDDEN_CONTROLLED_ONLY_LOSSES + (
    "cache_validation",
    "task_loss",
    "candidate_individual_loss",
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
    _validate_losses(task_type, losses_by_stage, plan.get("loss_metadata", {}), errors)
    _validate_adapter_params(
        plan.get("adapter_params_by_candidate", {}),
        tuple(str(name) for name in plan.get("candidate_names", ())),
        errors,
    )
    return TrainingProtocolReport(ok=not errors, errors=errors, warnings=warnings)


def _validate_stage_sequence(task_type: str, stages: tuple[str, ...], errors: list[str]) -> None:
    expected = EXPECTED_STAGE_SEQUENCES.get(task_type)
    if expected is None:
        errors.append(f"unknown multimodal training task_type: {task_type}")
        return
    if stages != expected:
        errors.append(f"{task_type} training_stages must be {expected}, got {stages}")


def _validate_losses(
    task_type: str,
    losses_by_stage: dict[str, list[str]],
    loss_metadata: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    _validate_stage_loss_contract(task_type, losses_by_stage, errors)
    for stage, losses in sorted(losses_by_stage.items()):
        for loss in losses:
            if task_type in CONTROLLED_TASK_TYPES:
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
                elif loss in PUBLIC_MARKED_WEAK_LOSSES:
                    _validate_marked_weak_loss(loss, loss_metadata, errors)
                elif loss == "candidate_individual_loss":
                    _validate_public_candidate_loss_metadata(loss_metadata, errors)
            _validate_loss_weight_metadata(loss, loss_metadata, errors)


def _validate_stage_loss_contract(
    task_type: str,
    losses_by_stage: dict[str, list[str]],
    errors: list[str],
) -> None:
    expected_stages = EXPECTED_STAGE_SEQUENCES.get(task_type)
    if expected_stages is None:
        return
    if not isinstance(losses_by_stage, dict):
        errors.append("losses_by_stage must be a stage-to-loss mapping")
        return
    expected_stage_set = set(expected_stages)
    for stage in expected_stages:
        if stage not in losses_by_stage:
            errors.append(f"losses_by_stage missing stage: {stage}")
    for stage in sorted(set(losses_by_stage) - expected_stage_set):
        errors.append(f"losses_by_stage contains stage not in training_stages: {stage}")
    required_by_stage = REQUIRED_STAGE_LOSSES.get(task_type, {})
    for stage, required_losses in sorted(required_by_stage.items()):
        stage_losses = tuple(losses_by_stage.get(stage, ()))
        for loss in required_losses:
            if loss not in stage_losses:
                errors.append(f"{task_type} {stage} must include required loss/record: {loss}")


def _validate_marked_weak_loss(
    loss: str,
    loss_metadata: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    metadata = loss_metadata.get(loss)
    if not isinstance(metadata, dict):
        errors.append(f"weak loss requires structured marking metadata: {loss}")
        return
    if metadata.get("supervision_type") != "weak":
        errors.append(f"weak loss metadata must set supervision_type=weak: {loss}")
    if not str(metadata.get("source", "")).strip():
        errors.append(f"weak loss metadata must include non-empty source: {loss}")
    if metadata.get("must_report_as") != "weak":
        errors.append(f"weak loss metadata must set must_report_as=weak: {loss}")


def _validate_loss_weight_metadata(
    loss: str,
    loss_metadata: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    metadata = loss_metadata.get(loss)
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        errors.append(f"loss metadata must be structured: {loss}")
        return
    if "weight" not in metadata:
        return
    try:
        weight = float(metadata["weight"])
    except (TypeError, ValueError):
        errors.append(f"loss weight must be a finite non-negative number: {loss}")
        return
    if not math.isfinite(weight) or weight < 0.0:
        errors.append(f"loss weight must be a finite non-negative number: {loss}")


def _validate_public_candidate_loss_metadata(
    loss_metadata: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    metadata = loss_metadata.get("candidate_individual_loss", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        errors.append("loss metadata must be structured: candidate_individual_loss")
        return
    try:
        weight = float(metadata.get("weight", 0.0))
    except (TypeError, ValueError):
        weight = float("nan")
    diagnostic_only = bool(metadata.get("diagnostic_only", True))
    if weight != 0.0 or not diagnostic_only:
        errors.append(
            "public candidate_individual_loss must be diagnostic_only with weight 0.0; "
            "only controlled data may train every candidate against the same target"
        )


def _validate_adapter_params(
    params_by_candidate: dict[str, list[str]],
    candidate_names: tuple[str, ...],
    errors: list[str],
) -> None:
    expected_candidates = candidate_names or tuple(params_by_candidate)
    for candidate in expected_candidates:
        allowed = ALLOWED_V1_ADAPTER_PARAMS.get(candidate)
        if allowed is None:
            errors.append(f"unknown v1 adapter candidate: {candidate}")
            continue
        params = tuple(params_by_candidate.get(candidate, ()))
        if not params:
            errors.append(f"adapter params missing for {candidate}")
            continue
        for param in sorted({param for param in params if params.count(param) > 1}):
            errors.append(f"adapter params for {candidate} contains duplicate v1 param: {param}")
        for param in allowed:
            if param not in params:
                errors.append(f"adapter params for {candidate} missing required v1 param: {param}")
        for param in params:
            if param in FORBIDDEN_V1_ADAPTER_PARAMS:
                errors.append(f"v1 adapter param is forbidden: {candidate}.{param}")
            elif param not in allowed:
                errors.append(f"unknown v1 adapter param for {candidate}: {param}")
