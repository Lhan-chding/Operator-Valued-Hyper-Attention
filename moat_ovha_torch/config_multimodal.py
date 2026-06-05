from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moat_ovha_torch.models.multimodal.baselines import (
    forbidden_external_references,
    missing_required_baselines,
)
from moat_ovha_torch.eval.multimodal_robustness import DEFAULT_REQUIRED_STRESS_FAMILIES
from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol


DEFAULT_CANDIDATE_NAMES = ("TLEO", "SPO", "LRIO", "CATO")
CONTROLLED_TRAINING_STAGES = ("T0", "T1", "T2", "T3", "T4")
PUBLIC_TRAINING_STAGES = ("T0", "T5")
ROBUSTNESS_EVAL_STAGES = ("T0", "T6")
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


@dataclass(frozen=True)
class MultimodalExperimentConfig:
    name: str
    dataset_name: str
    task_type: str
    cache_version: str
    cache_root: Path
    output_dir: Path
    seeds: tuple[int, ...]
    training_stages: tuple[str, ...]
    candidate_names: tuple[str, ...]
    candidate_pool_names: tuple[str, ...]
    baseline_names: tuple[str, ...]
    eval_splits: tuple[str, ...]
    eval_episode_count: int
    enforce_same_features_for_baselines: bool = True
    fail_on_missing_cache_artifact: bool = True
    allow_hidden_losses: bool = False
    require_public_alignment_labels: bool = False
    robustness_corruptions: tuple[str, ...] = ()
    losses_by_stage: dict[str, tuple[str, ...]] | None = None
    loss_metadata: dict[str, dict[str, Any]] | None = None
    adapter_params_by_candidate: dict[str, tuple[str, ...]] | None = None
    lrio_pairs: tuple[tuple[str, str], ...] = ()
    use_evidence_router: bool = True
    composition_mode: str = "convex_mixture"
    base_candidate: str | None = None
    residual_candidates: tuple[str, ...] = ()
    main_model_name: str = "ovha_full"

    @classmethod
    def from_file(cls, path: Path | str) -> "MultimodalExperimentConfig":
        return cls.from_mapping(json.loads(Path(path).read_text()))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "MultimodalExperimentConfig":
        seeds = tuple(int(seed) for seed in mapping.get("seeds", ()))
        if len(seeds) < 3:
            raise ValueError("multimodal experiments require at least 3 seeds")
        candidate_names = tuple(mapping.get("candidate_names", DEFAULT_CANDIDATE_NAMES))
        _validate_candidate_names(candidate_names)
        candidate_pool_names = tuple(mapping.get("candidate_pool_names", candidate_names))
        _validate_candidate_names(candidate_pool_names)
        config = cls(
            name=str(mapping["name"]),
            dataset_name=str(mapping["dataset_name"]),
            task_type=str(mapping.get("task_type", mapping["dataset_name"])),
            cache_version=str(mapping.get("cache_version", "v0.1")),
            cache_root=Path(mapping.get("cache_root", "data/multimodal_cache")),
            output_dir=Path(mapping.get("output_dir", "outputs/multimodal")),
            seeds=seeds,
            training_stages=tuple(mapping.get("training_stages", ())),
            candidate_names=candidate_names,
            candidate_pool_names=candidate_pool_names,
            baseline_names=tuple(mapping.get("baseline_names", ())),
            eval_splits=tuple(mapping.get("eval_splits", ("val", "test"))),
            eval_episode_count=int(mapping.get("eval_episode_count", 0)),
            enforce_same_features_for_baselines=bool(mapping.get("enforce_same_features_for_baselines", True)),
            fail_on_missing_cache_artifact=bool(mapping.get("fail_on_missing_cache_artifact", True)),
            allow_hidden_losses=bool(mapping.get("allow_hidden_losses", False)),
            require_public_alignment_labels=bool(mapping.get("require_public_alignment_labels", False)),
            robustness_corruptions=tuple(mapping.get("robustness_corruptions", ())),
            losses_by_stage=_tuple_mapping(mapping.get("losses_by_stage")),
            loss_metadata=dict(mapping.get("loss_metadata", {})) if "loss_metadata" in mapping else None,
            adapter_params_by_candidate=_tuple_mapping(mapping.get("adapter_params_by_candidate")),
            lrio_pairs=_pair_tuple(mapping.get("lrio_pairs")),
            use_evidence_router=bool(mapping.get("use_evidence_router", True)),
            composition_mode=str(mapping.get("composition_mode", "convex_mixture")),
            base_candidate=str(mapping["base_candidate"]) if mapping.get("base_candidate") is not None else None,
            residual_candidates=tuple(str(candidate) for candidate in mapping.get("residual_candidates", ())),
            main_model_name=str(mapping.get("main_model_name", "ovha_full")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.training_stages:
            raise ValueError("training_stages must be explicit")
        expected_stages = _expected_training_stages(self.task_type, self.robustness_corruptions)
        if self.training_stages != expected_stages:
            raise ValueError(
                f"training_stages for {self.task_type} must be {expected_stages}, got {self.training_stages}"
            )
        if self.eval_episode_count <= 0:
            raise ValueError("eval_episode_count must be positive")
        if not self.baseline_names:
            raise ValueError("baseline_names must be explicit")
        if not self.enforce_same_features_for_baselines:
            raise ValueError("same frozen features for OVHA and baselines are mandatory")
        if not self.fail_on_missing_cache_artifact:
            raise ValueError("missing cache artifacts must fail fast")
        if self.task_type != "controlled_multimodal" and self.allow_hidden_losses:
            raise ValueError("hidden losses are controlled-only and forbidden for public data")
        if self.robustness_corruptions:
            _validate_robustness_corruptions(self.robustness_corruptions)
        missing_primary = sorted(set(self.candidate_names) - set(self.candidate_pool_names))
        if missing_primary:
            raise ValueError("candidate_pool_names must include every primary candidate: " + ", ".join(missing_primary))
        if not self.main_model_name.strip():
            raise ValueError("main_model_name must be non-empty")
        _validate_embedded_training_protocol(self)
        _validate_public_alignment_label_contract(self)
        missing = missing_required_baselines(self.task_type, (self.main_model_name, *self.baseline_names))
        if missing:
            raise ValueError(f"missing required same-feature baselines: {', '.join(missing)}")
        forbidden = forbidden_external_references(self.task_type, self.baseline_names)
        if forbidden:
            raise ValueError(
                "external references must not be listed as same-feature baselines: "
                + ", ".join(forbidden)
            )
        _validate_composition_config(self)


def _expected_training_stages(task_type: str, robustness_corruptions: tuple[str, ...]) -> tuple[str, ...]:
    if robustness_corruptions:
        return ROBUSTNESS_EVAL_STAGES
    if task_type in CONTROLLED_TASK_TYPES:
        return CONTROLLED_TRAINING_STAGES
    if task_type in REGION_TEXT_TASK_TYPES or task_type in SENTIMENT_EMOTION_TASK_TYPES:
        return PUBLIC_TRAINING_STAGES
    raise ValueError(f"unknown multimodal task_type for training stages: {task_type}")


def _validate_embedded_training_protocol(config: MultimodalExperimentConfig) -> None:
    if config.losses_by_stage is None:
        raise ValueError("losses_by_stage must be explicit")
    if config.adapter_params_by_candidate is None:
        raise ValueError("adapter_params_by_candidate must be explicit")
    report = validate_training_protocol(
        {
            "task_type": "robustness_eval" if config.robustness_corruptions else config.task_type,
            "training_stages": config.training_stages,
            "losses_by_stage": config.losses_by_stage,
            "loss_metadata": config.loss_metadata or {},
            "adapter_params_by_candidate": config.adapter_params_by_candidate,
        }
    )
    if not report.ok:
        raise ValueError("; ".join(report.errors))


def _validate_public_alignment_label_contract(config: MultimodalExperimentConfig) -> None:
    if config.robustness_corruptions or config.task_type not in REGION_TEXT_TASK_TYPES:
        return
    t5_losses = tuple((config.losses_by_stage or {}).get("T5", ()))
    if "public_alignment_ce" in t5_losses and not config.require_public_alignment_labels:
        raise ValueError("region-text public_alignment_ce requires require_public_alignment_labels=true")


def _validate_composition_config(config: MultimodalExperimentConfig) -> None:
    if config.composition_mode in {"convex_mixture", "tanso_base"}:
        if config.base_candidate is not None or config.residual_candidates:
            raise ValueError(f"{config.composition_mode} composition must not set base_candidate or residual_candidates")
        if config.composition_mode == "tanso_base" and config.candidate_names != ("TANSOBase",):
            raise ValueError("tanso_base composition requires candidate_names=['TANSOBase']")
        return
    if config.composition_mode != "base_plus_residual":
        raise ValueError("composition_mode must be convex_mixture, tanso_base, or base_plus_residual")
    if config.base_candidate not in config.candidate_names:
        raise ValueError("base_plus_residual base_candidate must be an active candidate")
    if not config.residual_candidates:
        raise ValueError("base_plus_residual requires residual_candidates")
    invalid = [
        candidate
        for candidate in config.residual_candidates
        if candidate not in config.candidate_names or candidate == config.base_candidate
    ]
    if invalid:
        raise ValueError("base_plus_residual residual_candidates must be active non-base candidates: " + ", ".join(invalid))


def _validate_robustness_corruptions(robustness_corruptions: tuple[str, ...]) -> None:
    observed = _observed_robustness_families(robustness_corruptions)
    missing = sorted(set(DEFAULT_REQUIRED_STRESS_FAMILIES) - observed)
    if missing:
        raise ValueError("missing required robustness stress families: " + ", ".join(missing))


def _observed_robustness_families(robustness_corruptions: tuple[str, ...]) -> set[str]:
    alias_to_family = {
        _normalize_stress_name(alias): family
        for family, aliases in DEFAULT_REQUIRED_STRESS_FAMILIES.items()
        for alias in (family, *aliases)
    }
    observed = set()
    for corruption in robustness_corruptions:
        family = alias_to_family.get(_normalize_stress_name(corruption))
        if family is not None:
            observed.add(family)
    return observed


def _normalize_stress_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _tuple_mapping(value: Any) -> dict[str, tuple[str, ...]] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("embedded training protocol mapping fields must be objects")
    return {str(key): tuple(str(item) for item in values) for key, values in value.items()}


def _pair_tuple(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("lrio_pairs must be a list of modality pairs")
    pairs = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("lrio_pairs entries must be two-modality lists")
        left, right = str(item[0]), str(item[1])
        if not left or not right or left == right:
            raise ValueError("lrio_pairs entries must contain two distinct modalities")
        pairs.append((left, right))
    return tuple(pairs)


def _validate_candidate_names(candidate_names: tuple[str, ...]) -> None:
    if not candidate_names:
        raise ValueError("candidate_names must contain at least one v1 candidate")
    allowed = {"TLEO", "SPO", "LRIO", "CATO", "PRSO", "SRO", "TANSO", "TANSOBase", "TANSOShift"}
    invalid = sorted(set(candidate_names) - allowed)
    if invalid:
        raise ValueError("candidate_names contains unknown v1 candidate(s): " + ", ".join(invalid))
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("candidate_names must not contain duplicates")
