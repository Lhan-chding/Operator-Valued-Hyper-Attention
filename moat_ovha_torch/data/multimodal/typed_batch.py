from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FORBIDDEN_MULTIMODAL_INPUT_KEYS = (
    "true_active_operator",
    "true_router",
    "true_adapter",
    "true_alignment",
    "true_rank",
    "true_prototype",
    "true_lengthscale",
    "true_reliability",
    "true_corruption",
    "hidden",
    "oracle",
    "corruption_strength",
    "mismatch_source_id",
)


@dataclass(frozen=True)
class TokenField:
    modality: str
    x: Any
    pos: Any
    mask: Any
    quality: Any | None = None
    attrs: dict[str, Any] | None = None


@dataclass(frozen=True)
class QueryField:
    x: Any
    pos: Any
    query_type: Any
    mask: Any


@dataclass(frozen=True)
class SupervisionBank:
    task_label: Any | None
    alignment_pairs: Any | None
    alignment_weights: Any | None
    bbox_targets: Any | None
    region_targets: Any | None
    timestamp_targets: Any | None
    modality_missing_mask: Any | None
    corruption_metadata: dict[str, Any] | None
    weak_labels: dict[str, Any] | None
    weak_label_confidence: dict[str, Any] | None
    pseudo_label_source: dict[str, str] | None


@dataclass(frozen=True)
class ProvenanceBank:
    source_id: list[str]
    original_split: list[str]
    raw_ref: list[str]
    license_tag: list[str]
    preprocessing_version: str
    feature_extractor_version: dict[str, str]
    pseudo_label_version: dict[str, str]


@dataclass(frozen=True)
class MultimodalEpisodeBatch:
    fields: dict[str, TokenField]
    query: QueryField
    target_y: Any
    target_mask: Any
    task_type: str
    split: str
    source_dataset: str
    supervision: SupervisionBank
    provenance: ProvenanceBank
    hidden: dict[str, Any] | None = None

    def model_inputs(self) -> dict[str, Any]:
        values = {
            "fields": self.fields,
            "query": self.query,
            "target_mask": self.target_mask,
            "task_type": self.task_type,
            "split": self.split,
            "source_dataset": self.source_dataset,
        }
        assert_no_multimodal_metadata_leakage(values)
        return values


def assert_no_multimodal_metadata_leakage(inputs: dict[str, Any]) -> None:
    lowered = _flatten_keys(inputs)
    for key in lowered:
        for forbidden in FORBIDDEN_MULTIMODAL_INPUT_KEYS:
            if forbidden in key:
                raise ValueError(f"multimodal metadata leakage detected in model input key: {key}")


def _flatten_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}".lower() if prefix else str(key).lower()
            keys.add(name)
            keys.update(_flatten_keys(item, name))
    elif hasattr(value, "__dataclass_fields__"):
        for key in value.__dataclass_fields__:
            item = getattr(value, key)
            name = f"{prefix}.{key}".lower() if prefix else key.lower()
            keys.add(name)
            keys.update(_flatten_keys(item, name))
    return keys
