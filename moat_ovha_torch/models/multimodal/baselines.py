from __future__ import annotations


REGION_TEXT_BASELINES = (
    "text_only",
    "region_only",
    "concat_fusion",
    "cross_attention_transformer",
    "modality_expert_moe",
    "clip_style_region_text_retrieval",
    "cato_only",
    "ovha_no_cato",
    "ovha_no_rceo",
    "ovha_no_evidence_router",
)

SENTIMENT_EMOTION_BASELINES = (
    "concat_fusion",
    "tfn_lmf",
    "mult_style_crossmodal_transformer",
    "misa_shared_private",
    "modality_expert_moe",
    "quality_aware_fusion",
    "ovha_no_lrio",
    "ovha_no_spo",
    "ovha_no_rceo",
    "ovha_no_evidence_router",
)

CONTROLLED_BASELINES = (
    "tleo_only",
    "spo_only",
    "lrio_only",
    "cato_only",
    "no_rceo",
    "no_evidence_router",
    "no_reliability_prior",
    "memory_only_router",
    "evidence_only_router",
    "no_operator_memory",
    "no_hyper_adapter",
    "modality_expert_moe",
    "concat_transformer",
)

REGION_TEXT_EXTERNAL_REFERENCES = ("MDETR", "GLIP", "GroundingDINO")
SENTIMENT_EMOTION_EXTERNAL_REFERENCES = ()
CONTROLLED_EXTERNAL_REFERENCES = ()


def baseline_names_for_task(task_type: str) -> tuple[str, ...]:
    if task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities"}:
        return REGION_TEXT_BASELINES
    if task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
        return SENTIMENT_EMOTION_BASELINES
    if task_type in {"controlled_multimodal", "controlled_relation_operator"}:
        return CONTROLLED_BASELINES
    raise ValueError(f"unknown multimodal task type for baseline registry: {task_type}")


def external_reference_names_for_task(task_type: str) -> tuple[str, ...]:
    if task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities"}:
        return REGION_TEXT_EXTERNAL_REFERENCES
    if task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
        return SENTIMENT_EMOTION_EXTERNAL_REFERENCES
    if task_type in {"controlled_multimodal", "controlled_relation_operator"}:
        return CONTROLLED_EXTERNAL_REFERENCES
    raise ValueError(f"unknown multimodal task type for external reference registry: {task_type}")


def missing_required_baselines(task_type: str, baseline_names: tuple[str, ...]) -> tuple[str, ...]:
    required = set(baseline_names_for_task(task_type))
    present = set(baseline_names)
    return tuple(sorted(required - present))


def forbidden_external_references(task_type: str, baseline_names: tuple[str, ...]) -> tuple[str, ...]:
    external = set(external_reference_names_for_task(task_type))
    present = set(baseline_names)
    return tuple(sorted(external & present))


def assert_same_feature_baseline_policy(config) -> None:
    if not config.enforce_same_features_for_baselines:
        raise ValueError("all OVHA/baseline comparisons must use the same frozen features")
    missing = missing_required_baselines(config.task_type, config.baseline_names)
    if missing:
        raise ValueError(f"missing required same-feature baselines: {', '.join(missing)}")
    forbidden = forbidden_external_references(config.task_type, config.baseline_names)
    if forbidden:
        raise ValueError(
            "external references must not be listed as same-feature baselines: "
            + ", ".join(forbidden)
        )
