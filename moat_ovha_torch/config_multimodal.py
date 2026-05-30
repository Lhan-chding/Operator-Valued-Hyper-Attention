from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATE_NAMES = ("TLEO", "SPO", "LRIO", "CATO")
CONTROLLED_TRAINING_STAGES = ("T0", "T1", "T2", "T3", "T4")
PUBLIC_TRAINING_STAGES = ("T0", "T5")
ROBUSTNESS_EVAL_STAGES = ("T0", "T6")


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
    baseline_names: tuple[str, ...]
    eval_splits: tuple[str, ...]
    eval_episode_count: int
    enforce_same_features_for_baselines: bool = True
    fail_on_missing_cache_artifact: bool = True
    allow_hidden_losses: bool = False
    require_public_alignment_labels: bool = False
    robustness_corruptions: tuple[str, ...] = ()

    @classmethod
    def from_file(cls, path: Path | str) -> "MultimodalExperimentConfig":
        return cls.from_mapping(json.loads(Path(path).read_text()))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "MultimodalExperimentConfig":
        seeds = tuple(int(seed) for seed in mapping.get("seeds", ()))
        if len(seeds) < 3:
            raise ValueError("multimodal experiments require at least 3 seeds")
        candidate_names = tuple(mapping.get("candidate_names", DEFAULT_CANDIDATE_NAMES))
        if candidate_names != DEFAULT_CANDIDATE_NAMES:
            raise ValueError("multimodal v1 candidate_names must be exactly TLEO/SPO/LRIO/CATO")
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
            baseline_names=tuple(mapping.get("baseline_names", ())),
            eval_splits=tuple(mapping.get("eval_splits", ("val", "test"))),
            eval_episode_count=int(mapping.get("eval_episode_count", 0)),
            enforce_same_features_for_baselines=bool(mapping.get("enforce_same_features_for_baselines", True)),
            fail_on_missing_cache_artifact=bool(mapping.get("fail_on_missing_cache_artifact", True)),
            allow_hidden_losses=bool(mapping.get("allow_hidden_losses", False)),
            require_public_alignment_labels=bool(mapping.get("require_public_alignment_labels", False)),
            robustness_corruptions=tuple(mapping.get("robustness_corruptions", ())),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.training_stages:
            raise ValueError("training_stages must be explicit")
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
