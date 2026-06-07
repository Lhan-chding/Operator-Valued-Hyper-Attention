from __future__ import annotations


REGION_TEXT_BASELINES = (
    "random_valid",
    "train_slot_prior",
    "box_prior",
    "prso_clip_similarity",
    "candidate_mlp_reranker",
    "cross_attention_reranker",
    "index_prior_only",
    "text_only",
    "region_only",
    "concat_fusion",
    "cato_only",
    "ovha_no_cato",
    "ovha_no_rceo",
    "ovha_no_evidence_router",
)

SENTIMENT_EMOTION_BASELINES = (
    "text_only",
    "audio_only",
    "vision_only",
    "concat_fusion",
    "spo_only",
    "lrio_only",
    "ovha_tanso_only",
    "ovha_spo_lrio",
    "ovha_lrio_tanso",
    "ovha_all_candidates_exploratory",
    "ovha_no_rceo",
    "ovha_with_evidence_router",
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

REGION_TEXT_SANITY_PROBES = (
    "random_valid",
    "train_slot_prior",
    "index_prior_only",
    "text_only",
    "region_only",
    "concat_fusion",
)
REGION_TEXT_STRONG_RERANKERS = (
    "box_prior",
    "prso_clip_similarity",
    "candidate_mlp_reranker",
    "cross_attention_reranker",
)
REGION_TEXT_OVHA_ABLATIONS = ("cato_only", "ovha_no_cato", "ovha_no_rceo", "ovha_no_evidence_router")
SENTIMENT_EMOTION_SANITY_PROBES = ("text_only", "audio_only", "vision_only", "concat_fusion")
SENTIMENT_EMOTION_OVHA_ABLATIONS = (
    "spo_only",
    "lrio_only",
    "ovha_tanso_only",
    "ovha_spo_lrio",
    "ovha_lrio_tanso",
    "ovha_all_candidates_exploratory",
    "ovha_no_rceo",
    "ovha_with_evidence_router",
)
SENTIMENT_EMOTION_MECHANISM_BASELINES = (
    "raw_tanso_mlp",
    "ovha_tanso_no_source_gate",
    "ovha_tanso_no_hyper_adapter",
    "ovha_tanso_no_operator_memory",
    "ovha_tanso_no_gate_aux",
)

REGION_TEXT_EXTERNAL_REFERENCES = (
    "MDETR",
    "GLIP",
    "GroundingDINO",
    "GroundingDINO-1.5",
    "TransVG",
    "LAVT",
    "SeqTR",
)
SENTIMENT_EMOTION_EXTERNAL_REFERENCES = (
    "TFN",
    "LMF",
    "MulT",
    "MISA",
    "MAG-BERT",
    "Self-MM",
)
CONTROLLED_EXTERNAL_REFERENCES = ()


def baseline_names_for_task(task_type: str) -> tuple[str, ...]:
    if task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}:
        return REGION_TEXT_BASELINES
    if task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
        return SENTIMENT_EMOTION_BASELINES
    if task_type in {"controlled_multimodal", "controlled_relation_operator"}:
        return CONTROLLED_BASELINES
    raise ValueError(f"unknown multimodal task type for baseline registry: {task_type}")


def external_reference_names_for_task(task_type: str) -> tuple[str, ...]:
    if task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}:
        return REGION_TEXT_EXTERNAL_REFERENCES
    if task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
        return SENTIMENT_EMOTION_EXTERNAL_REFERENCES
    if task_type in {"controlled_multimodal", "controlled_relation_operator"}:
        return CONTROLLED_EXTERNAL_REFERENCES
    raise ValueError(f"unknown multimodal task type for external reference registry: {task_type}")


def same_feature_probe_names_for_task(task_type: str) -> tuple[str, ...]:
    if task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}:
        return REGION_TEXT_SANITY_PROBES
    if task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
        return SENTIMENT_EMOTION_SANITY_PROBES
    if task_type in {"controlled_multimodal", "controlled_relation_operator"}:
        return CONTROLLED_BASELINES
    raise ValueError(f"unknown multimodal task type for same-feature probe registry: {task_type}")


def ovha_ablation_names_for_task(task_type: str) -> tuple[str, ...]:
    if task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}:
        return REGION_TEXT_OVHA_ABLATIONS
    if task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
        return SENTIMENT_EMOTION_OVHA_ABLATIONS
    if task_type in {"controlled_multimodal", "controlled_relation_operator"}:
        return ()
    raise ValueError(f"unknown multimodal task type for OVHA ablation registry: {task_type}")


def baseline_protocol_for_name(task_type: str, baseline_name: str) -> str:
    if task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
        if baseline_name in SENTIMENT_EMOTION_MECHANISM_BASELINES:
            if baseline_name == "raw_tanso_mlp":
                return "same_feature_mechanism_baseline"
            return "internal_ovha_mechanism_ablation"
    if baseline_name in same_feature_probe_names_for_task(task_type):
        return "same_feature_sanity_probe"
    if task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}:
        if baseline_name in REGION_TEXT_STRONG_RERANKERS:
            return "same_candidate_strong_reranker"
    if baseline_name in ovha_ablation_names_for_task(task_type):
        return "internal_ovha_ablation"
    if baseline_name in external_reference_names_for_task(task_type):
        return "external_sota_reference_or_reproduction"
    raise ValueError(f"unknown baseline/reference for {task_type}: {baseline_name}")


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
