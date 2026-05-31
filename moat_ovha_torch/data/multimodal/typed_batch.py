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
class BatchContractReport:
    ok: bool
    errors: list[str]


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
        contract = validate_multimodal_batch_contract(self)
        if not contract.ok:
            raise ValueError("invalid multimodal batch contract: " + "; ".join(contract.errors))
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


def validate_multimodal_batch_contract(batch: MultimodalEpisodeBatch) -> BatchContractReport:
    errors: list[str] = []
    _validate_episode_identity(batch, errors)
    query_x = _shape("query.x", batch.query.x, errors)
    query_pos = _shape("query.pos", batch.query.pos, errors)
    query_type = _shape("query.query_type", batch.query.query_type, errors)
    query_mask = _shape("query.mask", batch.query.mask, errors)
    target_y = _shape("target_y", batch.target_y, errors)
    target_mask = _shape("target_mask", batch.target_mask, errors)

    _require_rank("query.x", query_x, 3, errors)
    _require_rank("query.pos", query_pos, 3, errors)
    _require_rank_one_of("query.query_type", query_type, (2, 3), errors)
    _require_rank("query.mask", query_mask, 2, errors)
    _require_rank("target_y", target_y, 3, errors)
    _require_rank("target_mask", target_mask, 2, errors)

    batch_query = _infer_batch_query_shape(query_x, target_y, query_mask, query_type)
    if batch_query is not None:
        batch_size, query_count = batch_query
        _require_first_dims("query.pos", query_pos, (batch_size, query_count), errors)
        _require_first_dims("query.query_type", query_type, (batch_size, query_count), errors)
        _require_exact_shape("query.mask", query_mask, (batch_size, query_count), errors)
        _require_first_dims("target_y", target_y, (batch_size, query_count), errors)
        _require_exact_shape("target_mask", target_mask, (batch_size, query_count), errors)
        _validate_provenance_bank(batch.provenance, batch_size, batch.split, errors)
        modality_count = len(batch.fields) if isinstance(batch.fields, dict) else 0
        _validate_supervision_bank(batch.supervision, batch_size, query_count, modality_count, errors)
    else:
        errors.append("cannot infer shared [B,Q] from query.x, target_y, or query.mask")

    if not batch.fields:
        errors.append("fields must contain at least one TokenField")
    for name, field in sorted(batch.fields.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or not name:
            errors.append("fields keys must be non-empty strings")
            continue
        if _contains_forbidden_metadata_key(name):
            errors.append(f"fields.{name} must not expose controlled or hidden metadata as model input")
        if not isinstance(field.modality, str) or not field.modality:
            errors.append(f"fields.{name}.modality must be a non-empty string")
        elif _contains_forbidden_metadata_key(field.modality):
            errors.append(f"fields.{name}.modality must not expose controlled or hidden metadata as model input")
        elif field.modality != name:
            errors.append(f"fields.{name}.modality must match dictionary key")
        field_x = _shape(f"fields.{name}.x", field.x, errors)
        field_pos = _shape(f"fields.{name}.pos", field.pos, errors)
        field_mask = _shape(f"fields.{name}.mask", field.mask, errors)
        _require_rank(f"fields.{name}.x", field_x, 3, errors)
        _require_rank(f"fields.{name}.pos", field_pos, 3, errors)
        _require_rank(f"fields.{name}.mask", field_mask, 2, errors)
        if field_x is not None and len(field_x) >= 2:
            if batch_query is not None and field_x[0] != batch_query[0]:
                errors.append(f"fields.{name}.x batch dimension must match query batch")
            _require_first_dims(f"fields.{name}.pos", field_pos, field_x[:2], errors, target_name=f"fields.{name}.x")
            _require_exact_shape(f"fields.{name}.mask", field_mask, field_x[:2], errors, target_name=f"fields.{name}.x")
            _validate_quality_shape(name, field.quality, field_x, errors)
            _validate_field_attrs(name, field.attrs, errors)
    return BatchContractReport(ok=not errors, errors=errors)


def assert_no_multimodal_metadata_leakage(inputs: dict[str, Any]) -> None:
    lowered = _flatten_keys(inputs)
    for key in lowered:
        if _contains_forbidden_metadata_key(key):
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


def _shape(name: str, value: Any, errors: list[str]) -> tuple[int, ...] | None:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        errors.append(f"{name} must expose a tensor-like shape")
        return None
    try:
        return tuple(int(dim) for dim in raw_shape)
    except (TypeError, ValueError):
        errors.append(f"{name} shape must be an integer tuple")
        return None


def _require_rank(name: str, shape: tuple[int, ...] | None, rank: int, errors: list[str]) -> None:
    if shape is not None and len(shape) != rank:
        errors.append(f"{name} must have rank {rank}, got {len(shape)}")


def _require_rank_one_of(name: str, shape: tuple[int, ...] | None, ranks: tuple[int, ...], errors: list[str]) -> None:
    if shape is not None and len(shape) not in ranks:
        allowed = "/".join(str(rank) for rank in ranks)
        errors.append(f"{name} must have rank {allowed}, got {len(shape)}")


def _infer_batch_query_shape(
    query_x: tuple[int, ...] | None,
    target_y: tuple[int, ...] | None,
    query_mask: tuple[int, ...] | None,
    query_type: tuple[int, ...] | None,
) -> tuple[int, int] | None:
    for shape, ranks in ((query_x, (3,)), (target_y, (3,)), (query_mask, (2,)), (query_type, (2, 3))):
        if shape is not None and len(shape) in ranks:
            return shape[0], shape[1]
    return None


def _require_first_dims(
    name: str,
    shape: tuple[int, ...] | None,
    expected: tuple[int, int],
    errors: list[str],
    *,
    target_name: str = "[B,Q]",
) -> None:
    if shape is not None and len(shape) >= 2 and shape[:2] != expected:
        if len(shape) == 2:
            errors.append(f"{name} shape must be {target_name}: expected {expected}, got {shape}")
        else:
            errors.append(f"{name} first two dims must match {target_name}: expected {expected}, got {shape[:2]}")


def _require_exact_shape(
    name: str,
    shape: tuple[int, ...] | None,
    expected: tuple[int, int],
    errors: list[str],
    *,
    target_name: str = "[B,Q]",
) -> None:
    if shape is not None and shape != expected:
        errors.append(f"{name} shape must be {target_name}: expected {expected}, got {shape}")


def _validate_quality_shape(name: str, quality: Any | None, field_shape: tuple[int, ...], errors: list[str]) -> None:
    if quality is None:
        return
    quality_shape = _shape(f"fields.{name}.quality", quality, errors)
    if quality_shape is None:
        return
    if quality_shape == (field_shape[0], 1):
        return
    if quality_shape == (field_shape[0], field_shape[1], 1):
        return
    errors.append(
        f"fields.{name}.quality shape must be [B,1] or [B,N,1]: "
        f"expected {(field_shape[0], 1)} or {(field_shape[0], field_shape[1], 1)}, got {quality_shape}"
    )


def _validate_episode_identity(batch: MultimodalEpisodeBatch, errors: list[str]) -> None:
    for key in ("task_type", "split", "source_dataset"):
        value = getattr(batch, key)
        if not _is_non_empty_string(value):
            errors.append(f"{key} must be a non-empty string")
            continue
        if not _is_normalized_non_empty_string(value):
            errors.append(f"{key} must be a non-empty normalized string")


def _validate_field_attrs(name: str, attrs: dict[str, Any] | None, errors: list[str]) -> None:
    if attrs is None:
        return
    if not isinstance(attrs, dict):
        errors.append(f"fields.{name}.attrs must be a dict keyed by non-empty strings")
        return
    for key in attrs:
        if not isinstance(key, str) or not key:
            errors.append(f"fields.{name}.attrs keys must be non-empty strings")
            continue
        if _contains_forbidden_metadata_key(key):
            errors.append(
                f"fields.{name}.attrs must not expose controlled or hidden metadata as model input: {key}"
            )


def _contains_forbidden_metadata_key(key: str) -> bool:
    normalized = _normalized_metadata_key(key)
    return any(forbidden in normalized for forbidden in FORBIDDEN_MULTIMODAL_INPUT_KEYS)


def _normalized_metadata_key(key: str) -> str:
    lowered = key.lower()
    return "".join(character if character.isalnum() else "_" for character in lowered)


def _validate_provenance_bank(provenance: ProvenanceBank, batch_size: int, split: str, errors: list[str]) -> None:
    for key in ("source_id", "original_split", "raw_ref", "license_tag"):
        values = getattr(provenance, key)
        if not isinstance(values, list):
            errors.append(f"provenance.{key} must be a list of strings")
            continue
        if len(values) != batch_size:
            errors.append(f"provenance.{key} length must match batch size {batch_size}, got {len(values)}")
        if any(not _is_non_empty_string(value) for value in values):
            errors.append(f"provenance.{key} entries must be non-empty strings")
        if key in {"source_id", "original_split"} and any(
            isinstance(value, str) and not _is_normalized_non_empty_string(value) for value in values
        ):
            errors.append(f"provenance.{key} entries must be non-empty normalized strings")
    if isinstance(split, str) and split:
        original_split = provenance.original_split
        if isinstance(original_split, list) and any(value != split for value in original_split if isinstance(value, str) and value):
            errors.append(f"provenance.original_split entries must match batch split {split}")
    if not _is_non_empty_string(provenance.preprocessing_version):
        errors.append("provenance.preprocessing_version must be a non-empty string")
    _validate_string_version_map(
        "provenance.feature_extractor_version",
        provenance.feature_extractor_version,
        errors,
        require_non_empty=True,
    )
    _validate_string_version_map(
        "provenance.pseudo_label_version",
        provenance.pseudo_label_version,
        errors,
        require_non_empty=False,
    )


def _validate_string_version_map(
    name: str,
    value: Any,
    errors: list[str],
    *,
    require_non_empty: bool,
) -> None:
    if not isinstance(value, dict) or (require_non_empty and not value):
        errors.append(f"{name} must map strings to non-empty strings")
        return
    for key, version in value.items():
        if not _is_non_empty_string(key) or not _is_non_empty_string(version):
            errors.append(f"{name} must map strings to non-empty strings")
            return


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_normalized_non_empty_string(value: str) -> bool:
    return bool(value) and value == value.strip()


def _validate_supervision_bank(
    supervision: SupervisionBank,
    batch_size: int,
    query_count: int,
    modality_count: int,
    errors: list[str],
) -> None:
    _validate_alignment_supervision(supervision, batch_size, query_count, errors)
    _validate_missing_modality_supervision(supervision, batch_size, modality_count, errors)
    _validate_corruption_metadata(supervision, batch_size, query_count, modality_count, errors)
    weak_labels = supervision.weak_labels
    if weak_labels is None:
        return
    if not isinstance(weak_labels, dict) or any(not isinstance(key, str) or not key for key in weak_labels):
        errors.append("supervision.weak_labels must be a dict keyed by non-empty strings")
        return
    expected_keys = set(weak_labels)
    confidence = supervision.weak_label_confidence
    if not isinstance(confidence, dict) or set(confidence) != expected_keys:
        errors.append("supervision.weak_label_confidence keys must match weak_labels keys")
    sources = supervision.pseudo_label_source
    if not isinstance(sources, dict) or set(sources) != expected_keys:
        errors.append("supervision.pseudo_label_source keys must match weak_labels keys")
    if not _is_string_to_non_empty_string_map(sources):
        errors.append("supervision.pseudo_label_source must map strings to non-empty strings")
    if not isinstance(confidence, dict):
        return
    for key, weak_value in sorted(weak_labels.items()):
        weak_shape = _shape(f"supervision.weak_labels.{key}", weak_value, errors)
        if weak_shape is not None:
            _require_first_dims(
                f"supervision.weak_labels.{key}",
                weak_shape,
                (batch_size, query_count),
                errors,
            )
        confidence_value = confidence.get(key)
        confidence_shape = _shape(f"supervision.weak_label_confidence.{key}", confidence_value, errors)
        if weak_shape is not None and confidence_shape is not None and confidence_shape != weak_shape:
            errors.append(f"supervision.weak_label_confidence.{key} shape must match weak label")


def _is_string_to_non_empty_string_map(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) and key and isinstance(item, str) and item for key, item in value.items())


def _validate_alignment_supervision(
    supervision: SupervisionBank,
    batch_size: int,
    query_count: int,
    errors: list[str],
) -> None:
    if supervision.alignment_pairs is not None:
        pair_shape = _shape("supervision.alignment_pairs", supervision.alignment_pairs, errors)
        _require_rank("supervision.alignment_pairs", pair_shape, 3, errors)
        _require_first_dims("supervision.alignment_pairs", pair_shape, (batch_size, query_count), errors)
        if pair_shape is not None and len(pair_shape) == 3 and pair_shape[-1] != 2:
            errors.append("supervision.alignment_pairs last dimension must be 2")
    if supervision.alignment_weights is not None:
        weight_shape = _shape("supervision.alignment_weights", supervision.alignment_weights, errors)
        _require_rank("supervision.alignment_weights", weight_shape, 2, errors)
        _require_exact_shape("supervision.alignment_weights", weight_shape, (batch_size, query_count), errors)


def _validate_missing_modality_supervision(
    supervision: SupervisionBank,
    batch_size: int,
    modality_count: int,
    errors: list[str],
) -> None:
    if supervision.modality_missing_mask is None:
        return
    mask_shape = _shape("supervision.modality_missing_mask", supervision.modality_missing_mask, errors)
    _require_rank("supervision.modality_missing_mask", mask_shape, 2, errors)
    if modality_count > 0:
        _require_exact_shape(
            "supervision.modality_missing_mask",
            mask_shape,
            (batch_size, modality_count),
            errors,
            target_name="[B,M]",
        )


def _validate_corruption_metadata(
    supervision: SupervisionBank,
    batch_size: int,
    query_count: int,
    modality_count: int,
    errors: list[str],
) -> None:
    metadata = supervision.corruption_metadata
    if metadata is None:
        return
    if not isinstance(metadata, dict) or not metadata:
        errors.append("supervision.corruption_metadata must be a non-empty dict keyed by non-empty strings")
        return
    for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
        if not isinstance(key, str) or not key:
            errors.append("supervision.corruption_metadata keys must be non-empty strings")
            continue
        value_shape = _shape(f"supervision.corruption_metadata.{key}", value, errors)
        if value_shape is None:
            continue
        if not value_shape:
            errors.append(f"supervision.corruption_metadata.{key} must have at least one batch dimension")
            continue
        if value_shape[0] != batch_size:
            errors.append(f"supervision.corruption_metadata.{key} first dimension must match batch size {batch_size}")
            continue
        if len(value_shape) >= 2 and value_shape[1] not in {1, query_count, modality_count}:
            errors.append(
                f"supervision.corruption_metadata.{key} second dimension must be 1, Q={query_count}, or M={modality_count}"
            )
