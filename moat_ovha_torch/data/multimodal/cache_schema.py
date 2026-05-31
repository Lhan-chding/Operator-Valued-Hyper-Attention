from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_DATA_CARD_KEYS = (
    "dataset_name",
    "cache_version",
    "modalities",
    "tasks",
    "operator_supervision",
    "leakage_controls",
)

REQUIRED_OPERATOR_SUPERVISION_KEYS = ("TLEO", "SPO", "LRIO", "CATO", "RCEO")
FAILED_SAMPLE_MANIFEST_REQUIRED_KEYS = ("source_id", "split", "reason")
SAMPLE_RECORD_MANIFEST_REQUIRED_KEYS = (
    "source_id",
    "split",
    "original_split",
    "raw_ref",
    "license_tag",
    "preprocessing_version",
)
SENTIMENT_SAMPLE_RECORD_REQUIRED_KEYS = (
    "utterance_id",
    "dialogue_id",
    "transcript_source",
    "missing_modality_mask_ref",
    "corruption_metadata_ref",
)
GROUNDING_SAMPLE_RECORD_REQUIRED_KEYS = (
    "image_id",
    "caption_id",
    "phrase_span",
    "region_box",
    "candidate_region_source",
    "box_coordinate_convention",
)
GROUNDING_SAMPLE_RECORD_STRING_KEYS = (
    "image_id",
    "caption_id",
    "candidate_region_source",
    "box_coordinate_convention",
)
TOKEN_FIELD_MANIFEST_REQUIRED_KEYS = ("x", "pos", "mask")
TOKEN_FIELD_MANIFEST_ALLOWED_ROOTS = {
    "x": "token_fields",
    "pos": "positions",
    "mask": "masks",
}
GROUNDING_REQUIRED_SUPERVISION_PATTERNS = (
    "alignment_pairs_{split}.parquet",
    "bbox_targets_{split}.npy",
    "region_targets_{split}.npy",
)
SENTIMENT_REQUIRED_SUPERVISION_PATTERNS = ("missing_modality_mask_{split}.npy",)
RCEO_REQUIRED_SUPERVISION_PATTERNS = ("corruption_{split}.parquet",)
WEAK_LABEL_SUPERVISION_PATTERN = "weak_labels_{split}.parquet"
ALLOWED_WEAK_LABEL_SUPERVISION_TYPES = frozenset({"weak", "pseudo"})
FORBIDDEN_MODEL_INPUT_MODALITIES = frozenset(
    {
        "corruption_metadata",
        "corruption_strength",
        "dataset_hidden_metadata",
        "hidden",
        "hidden_metadata",
        "mismatch_source_id",
        "oracle",
        "true_active_operator",
        "true_adapter",
        "true_adapter_params",
        "true_alignment",
        "true_alignment_pairs",
        "true_corruption_level",
        "true_lengthscale",
        "true_prototype",
        "true_prototype_logits",
        "true_rank",
        "true_rank_logits",
        "true_reliability",
        "true_router",
        "true_router_weights",
    }
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CHECKSUM_MANIFEST_NAME = "checksums.json"
PSEUDO_LABEL_ALLOWED_SOURCE_SPLITS = frozenset({"train"})


@dataclass(frozen=True)
class MultimodalCacheLayout:
    cache_root: Path
    dataset_name: str
    version: str

    def __init__(self, cache_root: Path | str, dataset_name: str, version: str):
        object.__setattr__(self, "cache_root", Path(cache_root))
        object.__setattr__(self, "dataset_name", dataset_name)
        object.__setattr__(self, "version", version)

    @property
    def root(self) -> Path:
        return self.cache_root / self.dataset_name / self.version


@dataclass(frozen=True)
class CacheValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


def required_cache_files(layout: MultimodalCacheLayout, splits: tuple[str, ...] = ("train",)) -> set[Path]:
    root = layout.root
    required = {
        root / "data_card.json",
        root / "splits.json",
        root / "samples.parquet",
        root / "checksums.json",
        root / "provenance" / "feature_versions.json",
        root / "provenance" / "pseudo_label_versions.json",
    }
    for split in splits:
        required.update(
            {
                root / "provenance" / f"source_ids_{split}.txt",
                root / "provenance" / f"sample_records_{split}.jsonl",
                root / "provenance" / f"failed_samples_{split}.jsonl",
                root / "token_fields" / f"manifest_{split}.json",
                root / "masks" / f"text_mask_{split}.npy",
                root / "supervision" / f"task_labels_{split}.npy",
            }
        )
    return required


def validate_cache_layout(layout: MultimodalCacheLayout, splits: tuple[str, ...] = ("train",)) -> CacheValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    required_files = required_cache_files(layout, splits)
    for path in sorted(required_cache_files(layout, splits)):
        if not path.exists():
            errors.append(f"missing required cache artifact: {path.relative_to(layout.root)}")

    data_card_path = layout.root / "data_card.json"
    data_card: dict[str, Any] = {}
    checksums: dict[str, Any] | None = None
    if data_card_path.exists():
        try:
            data_card = json.loads(data_card_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid data_card.json: {exc}")
        else:
            if not isinstance(data_card, dict):
                errors.append("data_card.json must be a JSON object")
                data_card = {}
            else:
                for key in REQUIRED_DATA_CARD_KEYS:
                    if key not in data_card:
                        errors.append(f"data_card.json missing required key: {key}")
                _validate_data_card_identity(layout, data_card, errors)
                _validate_data_card_modalities(data_card, errors)
                _validate_data_card_string_list(data_card, "tasks", errors)
                _validate_operator_supervision(data_card.get("operator_supervision"), errors)
                _validate_leakage_controls(data_card.get("leakage_controls"), errors)

    checksums_path = layout.root / "checksums.json"
    if checksums_path.exists():
        try:
            loaded_checksums = json.loads(checksums_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid checksums.json: {exc}")
        else:
            if not isinstance(loaded_checksums, dict) or not loaded_checksums:
                errors.append("checksums.json must be a non-empty object of artifact hashes")
            else:
                checksums = loaded_checksums
                _validate_checksum_coverage(layout, required_files, checksums, errors)

    _validate_source_split_controls(layout, splits, errors)
    _validate_split_manifest_consistency(layout, splits, errors)
    _validate_feature_parity(layout, data_card, errors)
    _validate_pseudo_label_provenance(layout, splits, errors)
    _validate_sample_record_manifests(layout, splits, data_card, errors)
    _validate_failed_sample_manifests(layout, splits, errors)
    _validate_token_field_manifests(layout, splits, data_card, checksums, errors)
    _validate_supervision_artifacts(layout, splits, data_card, checksums, errors)

    return CacheValidationReport(ok=not errors, errors=errors, warnings=warnings)


def default_data_card(
    dataset_name: str,
    cache_version: str,
    modalities: list[str],
    tasks: list[str],
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "cache_version": cache_version,
        "modalities": modalities,
        "tasks": tasks,
        "operator_supervision": {
            "TLEO": "local region/token structure",
            "SPO": "class/prototype/task label",
            "LRIO": "paired modality interaction",
            "CATO": "phrase-region or source-target alignment",
            "RCEO": "quality/corruption/missing metadata",
        },
        "leakage_controls": {
            "split_by_source_id": True,
            "deduplicate_by_source_id": True,
            "pseudo_labels_generated_without_test_labels": True,
            "same_features_for_baselines": True,
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checksum_coverage(
    layout: MultimodalCacheLayout,
    required_files: set[Path],
    checksums: dict[str, Any],
    errors: list[str],
) -> None:
    for relative, digest in sorted(checksums.items()):
        if not isinstance(relative, str):
            errors.append("checksums.json keys must be relative path strings")
            continue
        if relative == CHECKSUM_MANIFEST_NAME:
            continue
        path = _checksum_artifact_path(layout, relative, errors)
        if path is None:
            continue
        if not path.exists():
            errors.append(f"checksums.json references missing artifact: {relative}")
            continue
        if not path.is_file():
            errors.append(f"checksums.json key must point to a file artifact: {relative}")
            continue
        _validate_checksum_value(relative, path, digest, errors)
    for path in sorted(required_files):
        if not path.exists():
            continue
        relative = str(path.relative_to(layout.root))
        if relative == CHECKSUM_MANIFEST_NAME:
            continue
        if relative not in checksums:
            errors.append(f"checksums.json missing hash for required artifact: {relative}")
    for path in sorted(layout.root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(layout.root))
        if relative == CHECKSUM_MANIFEST_NAME:
            continue
        if relative not in checksums:
            errors.append(f"checksums.json missing hash for cache artifact: {relative}")


def _checksum_artifact_path(layout: MultimodalCacheLayout, relative: str, errors: list[str]) -> Path | None:
    artifact_relative = Path(relative)
    if artifact_relative.is_absolute():
        errors.append(f"checksums.json key must be relative path within cache root: {relative}")
        return None
    artifact_path = (layout.root / artifact_relative).resolve()
    try:
        artifact_path.relative_to(layout.root.resolve())
    except ValueError:
        errors.append(f"checksums.json key must stay within cache root: {relative}")
        return None
    return artifact_path


def _validate_checksum_value(relative: str, path: Path, digest: Any, errors: list[str]) -> None:
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        errors.append(f"checksums.json hash for artifact must be lowercase SHA-256: {relative}")
        return
    actual = file_sha256(path)
    if digest != actual:
        errors.append(f"checksums.json hash mismatch for artifact: {relative}")


def _validate_operator_supervision(operator_supervision: Any, errors: list[str]) -> None:
    if not isinstance(operator_supervision, dict):
        errors.append("data_card.json operator_supervision must be an object")
        return
    for operator in REQUIRED_OPERATOR_SUPERVISION_KEYS:
        if operator not in operator_supervision:
            errors.append(f"data_card.json operator_supervision missing required operator: {operator}")
            continue
        if not _is_non_empty_string(operator_supervision.get(operator)):
            errors.append(f"data_card.json operator_supervision.{operator} must be a non-empty string")


def _validate_data_card_identity(
    layout: MultimodalCacheLayout,
    data_card: dict[str, Any],
    errors: list[str],
) -> None:
    if "dataset_name" in data_card and data_card.get("dataset_name") != layout.dataset_name:
        errors.append(f"data_card.json dataset_name must match cache layout: expected {layout.dataset_name}")
    if "cache_version" in data_card and data_card.get("cache_version") != layout.version:
        errors.append(f"data_card.json cache_version must match cache layout: expected {layout.version}")


def _validate_data_card_string_list(data_card: dict[str, Any], key: str, errors: list[str]) -> None:
    values = data_card.get(key)
    if not isinstance(values, list) or not values or any(not _is_non_empty_string(value) for value in values):
        errors.append(f"data_card.json {key} must be a non-empty list of strings")
        return
    if any(value != value.strip() for value in values):
        errors.append(f"data_card.json {key} entries must be non-empty normalized strings")
        return
    if len(values) != len(set(values)):
        errors.append(f"data_card.json {key} must not contain duplicate entries")


def _validate_data_card_modalities(data_card: dict[str, Any], errors: list[str]) -> None:
    _validate_data_card_string_list(data_card, "modalities", errors)
    values = data_card.get("modalities")
    if not isinstance(values, list):
        return
    for modality in values:
        if isinstance(modality, str) and _contains_forbidden_model_input_modality(modality):
            errors.append(
                "data_card.json modalities must not expose controlled or hidden metadata as model input: "
                f"{modality}"
            )


def _contains_forbidden_model_input_modality(modality: str) -> bool:
    normalized = _normalized_model_input_name(modality)
    return any(forbidden in normalized for forbidden in FORBIDDEN_MODEL_INPUT_MODALITIES)


def _normalized_model_input_name(value: str) -> str:
    lowered = value.strip().lower()
    return "".join(character if character.isalnum() else "_" for character in lowered)


def _validate_leakage_controls(controls: Any, errors: list[str]) -> None:
    if not isinstance(controls, dict):
        errors.append("data_card.json leakage_controls must be an object")
        return
    for key in (
        "split_by_source_id",
        "deduplicate_by_source_id",
        "pseudo_labels_generated_without_test_labels",
        "same_features_for_baselines",
    ):
        if controls.get(key) is not True:
            errors.append(f"data_card.json leakage_controls.{key} must be true")


def _validate_source_split_controls(layout: MultimodalCacheLayout, splits: tuple[str, ...], errors: list[str]) -> None:
    seen: dict[str, str] = {}
    for split in splits:
        ids = _read_source_ids(layout, split, errors)
        if ids is None:
            continue
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate source_id within split {split}")
        for source_id in ids:
            previous = seen.get(source_id)
            if previous is not None and previous != split:
                errors.append(f"source_id appears in multiple splits: {source_id} ({previous}, {split})")
            seen[source_id] = split


def _validate_split_manifest_consistency(layout: MultimodalCacheLayout, splits: tuple[str, ...], errors: list[str]) -> None:
    path = layout.root / "splits.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"invalid splits.json: {exc}")
        return
    if not isinstance(payload, dict):
        errors.append("splits.json must be an object keyed by split")
        return
    split_source_ids = _validate_split_manifest_source_lists(payload, errors)
    _validate_split_manifest_global_source_disjointness(split_source_ids, errors)
    for split in splits:
        expected_source_ids = split_source_ids.get(split)
        if expected_source_ids is None:
            errors.append(f"splits.json missing source_id list for split: {split}")
            continue
        source_path = layout.root / "provenance" / f"source_ids_{split}.txt"
        if not source_path.exists():
            continue
        actual = _read_source_ids(layout, split)
        if actual is None:
            continue
        if set(expected_source_ids) != set(actual):
            errors.append(f"provenance/source_ids_{split}.txt must match splits.json {split} entries")


def _validate_split_manifest_source_lists(payload: dict[str, Any], errors: list[str]) -> dict[str, list[str]]:
    split_source_ids: dict[str, list[str]] = {}
    for split, expected in sorted(payload.items()):
        if not isinstance(split, str) or not split:
            errors.append("splits.json split names must be non-empty strings")
            continue
        if split != split.strip():
            errors.append("splits.json split names must be non-empty normalized strings")
            continue
        if not isinstance(expected, list):
            errors.append(f"splits.json {split} must be a source_id list")
            continue
        invalid_source_ids = [source_id for source_id in expected if not isinstance(source_id, str) or not source_id]
        if invalid_source_ids:
            errors.append(f"splits.json {split} source_id entries must be non-empty strings")
            continue
        unnormalized_source_ids = [
            source_id
            for source_id in expected
            if isinstance(source_id, str) and source_id != source_id.strip()
        ]
        if unnormalized_source_ids:
            errors.append(f"splits.json {split} source_id entries must be non-empty normalized strings")
            continue
        expected_source_ids = list(expected)
        duplicate_source_ids = sorted(
            {source_id for source_id in expected_source_ids if expected_source_ids.count(source_id) > 1}
        )
        for source_id in duplicate_source_ids:
            errors.append(f"splits.json {split} contains duplicate source_id: {source_id}")
        split_source_ids[split] = expected_source_ids
    return split_source_ids


def _validate_split_manifest_global_source_disjointness(
    split_source_ids: dict[str, list[str]],
    errors: list[str],
) -> None:
    seen: dict[str, str] = {}
    for split, source_ids in sorted(split_source_ids.items()):
        for source_id in source_ids:
            previous = seen.get(source_id)
            if previous is not None and previous != split:
                errors.append(
                    f"splits.json source_id appears in multiple splits: {source_id} ({previous}, {split})"
                )
            seen[source_id] = split


def _validate_feature_parity(layout: MultimodalCacheLayout, data_card: dict[str, Any], errors: list[str]) -> None:
    controls = data_card.get("leakage_controls", {}) if isinstance(data_card, dict) else {}
    if not isinstance(controls, dict) or controls.get("same_features_for_baselines") is not True:
        return
    path = layout.root / "provenance" / "feature_versions.json"
    if not path.exists():
        return
    try:
        feature_versions = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"invalid feature_versions.json: {exc}")
        return
    if not isinstance(feature_versions, dict):
        errors.append("feature_versions.json must be a JSON object")
        return
    baselines = feature_versions.get("baselines", {})
    if not isinstance(baselines, dict) or not baselines:
        errors.append("feature_versions.json baselines must be a non-empty object when same_features_for_baselines is true")
        return
    if "ovha_full" not in baselines:
        errors.append("feature_versions.json baselines must include ovha_full reference")
        return
    modalities = data_card.get("modalities", [])
    modality_names = [str(modality) for modality in modalities] if isinstance(modalities, list) else []
    if isinstance(modalities, list):
        for modality_name in modality_names:
            if modality_name not in feature_versions:
                errors.append(f"feature_versions.json missing feature extractor version for modality: {modality_name}")
                continue
            _validate_feature_version_value(
                "feature_versions.json",
                modality_name,
                feature_versions.get(modality_name),
                errors,
            )
    ovha_reference = baselines["ovha_full"]
    if not isinstance(ovha_reference, dict):
        errors.append("feature_versions.json baselines.ovha_full must be an object")
        return
    reference = dict(ovha_reference)
    if isinstance(modalities, list):
        for modality_name in modality_names:
            if modality_name not in reference:
                errors.append(
                    "feature_versions.json baselines.ovha_full missing "
                    f"feature extractor version for modality: {modality_name}"
                )
                continue
            if not _validate_feature_version_value(
                "feature_versions.json baselines.ovha_full",
                modality_name,
                reference[modality_name],
                errors,
            ):
                continue
            if feature_versions.get(modality_name) != reference[modality_name]:
                errors.append(
                    f"feature_versions.json modality {modality_name} "
                    "must match baselines.ovha_full reference"
                )
    for model_name, versions in sorted(baselines.items()):
        if model_name == "ovha_full":
            continue
        if not isinstance(versions, dict):
            errors.append(f"feature_versions.json baselines.{model_name} must be an object")
            continue
        for modality_name in modality_names:
            if modality_name in versions:
                _validate_feature_version_value(
                    f"feature_versions.json baselines.{model_name}",
                    modality_name,
                    versions[modality_name],
                    errors,
                )
        if versions != reference:
            errors.append(
                "same_features_for_baselines is true but "
                f"{model_name} differs from ovha_full"
            )


def _validate_feature_version_value(
    context: str,
    modality_name: str,
    value: Any,
    errors: list[str],
) -> bool:
    if not _is_non_empty_string(value):
        errors.append(
            f"{context} feature extractor version for modality {modality_name} "
            "must be a non-empty string"
        )
        return False
    return True


def _validate_pseudo_label_provenance(
    layout: MultimodalCacheLayout,
    splits: tuple[str, ...],
    errors: list[str],
) -> None:
    path = layout.root / "provenance" / "pseudo_label_versions.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"invalid pseudo_label_versions.json: {exc}")
        return
    if not isinstance(payload, dict):
        errors.append("pseudo_label_versions.json must be an object")
        return
    generated_from = payload.get("generated_from_splits", [])
    if not isinstance(generated_from, list) or any(not isinstance(split, str) or not split for split in generated_from):
        errors.append("pseudo_label_versions.json generated_from_splits must be a list of split names")
        return
    known_splits = set(splits) | _read_split_manifest_names(layout) | {"train"}
    for split in generated_from:
        if split not in known_splits:
            errors.append(f"pseudo_label_versions.json generated_from_splits contains unknown split: {split}")
            continue
        if not _is_allowed_pseudo_label_source_split(split):
            if _is_test_split_name(split):
                errors.append("pseudo labels must not be generated from test split")
            else:
                errors.append(f"pseudo labels must not be generated from evaluation split: {split}")
    weak_label_artifacts_exist = _weak_label_artifacts_exist(layout, splits)
    if generated_from and not _is_non_empty_string(payload.get("version")):
        errors.append("pseudo_label_versions.json version must be a non-empty string")
    if weak_label_artifacts_exist:
        if not generated_from:
            errors.append(
                "pseudo_label_versions.json generated_from_splits must be non-empty when weak label artifacts exist"
            )
        if not _is_non_empty_string(payload.get("version")):
            errors.append(
                "pseudo_label_versions.json version must be a non-empty string when weak label artifacts exist"
            )
        _validate_weak_label_provenance(payload.get("label_provenance"), errors)


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _weak_label_artifacts_exist(layout: MultimodalCacheLayout, splits: tuple[str, ...]) -> bool:
    return any(
        (layout.root / "supervision" / WEAK_LABEL_SUPERVISION_PATTERN.format(split=split)).exists()
        for split in splits
    )


def _validate_weak_label_provenance(provenance: Any, errors: list[str]) -> None:
    if not isinstance(provenance, dict):
        errors.append(
            "pseudo_label_versions.json label_provenance must be an object when weak label artifacts exist"
        )
        return
    supervision_type = _normalized_label_supervision_type(provenance.get("supervision_type"))
    must_report_as = _normalized_label_supervision_type(provenance.get("must_report_as"))
    if supervision_type not in ALLOWED_WEAK_LABEL_SUPERVISION_TYPES:
        errors.append(
            "pseudo_label_versions.json label_provenance.supervision_type must be weak or pseudo "
            "when weak label artifacts exist"
        )
    if not isinstance(provenance.get("source"), str) or not provenance.get("source").strip():
        errors.append("pseudo_label_versions.json label_provenance.source must be a non-empty string")
    if must_report_as != supervision_type or must_report_as not in ALLOWED_WEAK_LABEL_SUPERVISION_TYPES:
        errors.append("pseudo_label_versions.json label_provenance.must_report_as must match supervision_type")


def _normalized_label_supervision_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pseudo_label": "pseudo",
        "pseudo_labels": "pseudo",
        "weak_label": "weak",
        "weak_labels": "weak",
    }
    return aliases.get(normalized, normalized)


def _is_allowed_pseudo_label_source_split(split: str) -> bool:
    return _normalized_split_name(split) in PSEUDO_LABEL_ALLOWED_SOURCE_SPLITS


def _is_test_split_name(split: str) -> bool:
    return _normalized_split_name(split) == "test"


def _normalized_split_name(split: str) -> str:
    return split.strip().lower().replace("-", "_").replace(" ", "_")


def _read_split_manifest_names(layout: MultimodalCacheLayout) -> set[str]:
    path = layout.root / "splits.json"
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return set()
    if not isinstance(payload, dict):
        return set()
    return {split for split in payload if isinstance(split, str) and split}


def _validate_failed_sample_manifests(
    layout: MultimodalCacheLayout,
    splits: tuple[str, ...],
    errors: list[str],
) -> None:
    retained_source_ids_by_split = {
        split: set(_read_source_ids(layout, split) or [])
        for split in splits
    }
    for split in splits:
        path = layout.root / "provenance" / f"failed_samples_{split}.jsonl"
        if not path.exists():
            continue
        retained_source_ids = retained_source_ids_by_split.get(split, set())
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name} line {line_number} is not valid JSON: {exc}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"{path.name} line {line_number} must be a JSON object")
                continue
            missing = [key for key in FAILED_SAMPLE_MANIFEST_REQUIRED_KEYS if key not in payload]
            if missing:
                errors.append(f"{path.name} line {line_number} missing required keys: {', '.join(missing)}")
            invalid_string_fields = [
                key
                for key in FAILED_SAMPLE_MANIFEST_REQUIRED_KEYS
                if key in payload and not _is_non_empty_string(payload.get(key))
            ]
            if invalid_string_fields:
                errors.append(
                    f"{path.name} line {line_number} required fields must be non-empty strings: "
                    f"{', '.join(invalid_string_fields)}"
                )
            if payload.get("split") != split:
                errors.append(f"{path.name} line {line_number} split must match {split}")
            source_id = payload.get("source_id")
            if isinstance(source_id, str) and (not source_id or source_id != source_id.strip()):
                errors.append(f"{path.name} line {line_number} source_id must be a non-empty normalized source_id")
            if isinstance(source_id, str) and source_id and source_id in retained_source_ids:
                errors.append(
                    f"{path.name} source_id must not also appear in retained "
                    f"split source ids: {source_id}"
                )
            if isinstance(source_id, str) and source_id:
                for retained_split, retained_ids in sorted(retained_source_ids_by_split.items()):
                    if retained_split == split:
                        continue
                    if source_id in retained_ids:
                        errors.append(
                            f"{path.name} source_id must not also appear in retained "
                            f"{retained_split} split source ids: {source_id}"
                        )


def _validate_sample_record_manifests(
    layout: MultimodalCacheLayout,
    splits: tuple[str, ...],
    data_card: dict[str, Any],
    errors: list[str],
) -> None:
    for split in splits:
        path = layout.root / "provenance" / f"sample_records_{split}.jsonl"
        if not path.exists():
            continue
        records = _read_jsonl_objects(path, errors)
        observed_source_ids: list[str] = []
        seen_source_ids: set[str] = set()
        for line_number, payload in records:
            missing = [key for key in SAMPLE_RECORD_MANIFEST_REQUIRED_KEYS if key not in payload]
            if missing:
                errors.append(f"{path.name} line {line_number} missing required keys: {', '.join(missing)}")
            invalid_string_fields = [
                key
                for key in SAMPLE_RECORD_MANIFEST_REQUIRED_KEYS
                if key in payload and not _is_non_empty_string(payload.get(key))
            ]
            if invalid_string_fields:
                errors.append(
                    f"{path.name} line {line_number} required fields must be non-empty strings: "
                    f"{', '.join(invalid_string_fields)}"
                )
            original_split = payload.get("original_split")
            if isinstance(original_split, str) and (not original_split or original_split != original_split.strip()):
                errors.append(
                    f"{path.name} line {line_number} original_split "
                    "must be a non-empty normalized split provenance"
                )
            if payload.get("split") != split:
                errors.append(f"{path.name} line {line_number} split must match {split}")
            _validate_grounding_sample_record(path.name, line_number, payload, data_card, errors)
            _validate_sentiment_sample_record(layout, split, path.name, line_number, payload, data_card, errors)
            source_id = payload.get("source_id")
            if isinstance(source_id, str) and source_id:
                if source_id in seen_source_ids:
                    errors.append(f"duplicate source_id within {path.name}: {source_id}")
                seen_source_ids.add(source_id)
                observed_source_ids.append(source_id)
        expected_source_ids = _read_source_ids(layout, split)
        if expected_source_ids is None:
            continue
        if set(observed_source_ids) != set(expected_source_ids):
            errors.append(
                f"provenance/sample_records_{split}.jsonl source_id set must match "
                f"provenance/source_ids_{split}.txt"
            )


def _read_jsonl_objects(path: Path, errors: list[str]) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name} line {line_number} is not valid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path.name} line {line_number} must be a JSON object")
            continue
        records.append((line_number, payload))
    return records


def _validate_grounding_sample_record(
    manifest_name: str,
    line_number: int,
    payload: dict[str, Any],
    data_card: dict[str, Any],
    errors: list[str],
) -> None:
    if not _data_card_requires_grounding_metadata(data_card):
        return
    missing = [key for key in GROUNDING_SAMPLE_RECORD_REQUIRED_KEYS if key not in payload]
    if missing:
        for key in missing:
            errors.append(f"{manifest_name} line {line_number} missing grounding metadata keys: {key}")
    invalid_strings = [
        key
        for key in GROUNDING_SAMPLE_RECORD_STRING_KEYS
        if key in payload and (not isinstance(payload.get(key), str) or not payload.get(key))
    ]
    if invalid_strings:
        errors.append(
            f"{manifest_name} line {line_number} grounding metadata fields must be non-empty strings: "
            f"{', '.join(invalid_strings)}"
        )
    if "phrase_span" in payload and not _valid_phrase_span(payload.get("phrase_span")):
        errors.append(f"{manifest_name} line {line_number} phrase_span must contain integer start/end with end > start")
    if "region_box" in payload and not _valid_region_box(payload.get("region_box")):
        errors.append(f"{manifest_name} line {line_number} region_box must contain four finite coordinates")


def _data_card_requires_grounding_metadata(data_card: dict[str, Any]) -> bool:
    if not isinstance(data_card, dict):
        return False
    tasks = data_card.get("tasks", [])
    return isinstance(tasks, list) and any(_requires_grounding_supervision(task) for task in tasks)


def _valid_phrase_span(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    start = value.get("start")
    end = value.get("end")
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and not isinstance(start, bool)
        and not isinstance(end, bool)
        and 0 <= start < end
    )


def _valid_region_box(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    for coordinate in value:
        if isinstance(coordinate, bool):
            return False
        try:
            numeric = float(coordinate)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(numeric):
            return False
    return True


def _validate_sentiment_sample_record(
    layout: MultimodalCacheLayout,
    split: str,
    manifest_name: str,
    line_number: int,
    payload: dict[str, Any],
    data_card: dict[str, Any],
    errors: list[str],
) -> None:
    if not _data_card_requires_sentiment_metadata(data_card):
        return
    required_keys = list(SENTIMENT_SAMPLE_RECORD_REQUIRED_KEYS)
    if _speaker_id_available(data_card):
        required_keys.append("speaker_id")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        for key in missing:
            errors.append(f"{manifest_name} line {line_number} missing sentiment/emotion metadata keys: {key}")
    invalid = [
        key
        for key in required_keys
        if key in payload and (not isinstance(payload.get(key), str) or not payload.get(key))
    ]
    if invalid:
        errors.append(
            f"{manifest_name} line {line_number} sentiment/emotion metadata fields must be non-empty strings: "
            f"{', '.join(invalid)}"
        )
    for key in ("missing_modality_mask_ref", "corruption_metadata_ref"):
        relative = payload.get(key)
        if not isinstance(relative, str) or not relative:
            continue
        _validate_sample_record_artifact_ref(layout, split, manifest_name, line_number, key, relative, errors)


def _data_card_requires_sentiment_metadata(data_card: dict[str, Any]) -> bool:
    if not isinstance(data_card, dict):
        return False
    tasks = data_card.get("tasks", [])
    return isinstance(tasks, list) and any(_requires_sentiment_supervision(task) for task in tasks)


def _speaker_id_available(data_card: dict[str, Any]) -> bool:
    availability = data_card.get("metadata_availability", {}) if isinstance(data_card, dict) else {}
    return isinstance(availability, dict) and availability.get("speaker_id") is True


def _validate_sample_record_artifact_ref(
    layout: MultimodalCacheLayout,
    split: str,
    manifest_name: str,
    line_number: int,
    key: str,
    relative: str,
    errors: list[str],
) -> None:
    expected_relative = {
        "missing_modality_mask_ref": f"supervision/missing_modality_mask_{split}.npy",
        "corruption_metadata_ref": f"supervision/corruption_{split}.parquet",
    }[key]
    if relative != expected_relative:
        errors.append(
            f"{manifest_name} line {line_number} {key} must reference the current split artifact: {expected_relative}"
        )
        return
    path = (layout.root / relative).resolve()
    try:
        path.relative_to(layout.root.resolve())
    except ValueError:
        errors.append(f"{manifest_name} line {line_number} {key} must stay within cache root: {relative}")
        return
    if not path.exists():
        errors.append(f"{manifest_name} line {line_number} {key} references missing artifact: {relative}")


def _read_source_ids(
    layout: MultimodalCacheLayout,
    split: str,
    errors: list[str] | None = None,
) -> list[str] | None:
    path = layout.root / "provenance" / f"source_ids_{split}.txt"
    if not path.exists():
        return None
    source_ids: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line or line != line.strip():
            if errors is not None:
                errors.append(
                    f"provenance/source_ids_{split}.txt line {line_number} "
                    "must be a non-empty normalized source_id"
                )
            continue
        source_ids.append(line)
    return source_ids


def _validate_token_field_manifests(
    layout: MultimodalCacheLayout,
    splits: tuple[str, ...],
    data_card: dict[str, Any],
    checksums: dict[str, Any] | None,
    errors: list[str],
) -> None:
    modalities = data_card.get("modalities", []) if isinstance(data_card, dict) else []
    if not isinstance(modalities, list) or not modalities:
        return
    expected_modalities = tuple(str(modality) for modality in modalities)
    for split in splits:
        manifest_path = layout.root / "token_fields" / f"manifest_{split}.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid {manifest_path.relative_to(layout.root)}: {exc}")
            continue
        if not isinstance(manifest, dict):
            errors.append(f"{manifest_path.relative_to(layout.root)} must be a JSON object keyed by modality")
            continue
        unexpected_modalities = sorted(str(modality) for modality in manifest if str(modality) not in expected_modalities)
        for modality in unexpected_modalities:
            errors.append(f"{manifest_path.relative_to(layout.root)} contains undeclared modality: {modality}")
        for modality in expected_modalities:
            entry = manifest.get(modality)
            if not isinstance(entry, dict):
                errors.append(f"{manifest_path.relative_to(layout.root)} missing modality entry: {modality}")
                continue
            missing = [key for key in TOKEN_FIELD_MANIFEST_REQUIRED_KEYS if not entry.get(key)]
            if missing:
                errors.append(
                    f"{manifest_path.relative_to(layout.root)} entry for {modality} "
                    f"missing required keys: {', '.join(missing)}"
                )
            for key in TOKEN_FIELD_MANIFEST_REQUIRED_KEYS:
                relative_path = entry.get(key)
                if not relative_path:
                    continue
                if not isinstance(relative_path, str):
                    errors.append(
                        f"{manifest_path.relative_to(layout.root)} entry for {modality}.{key} must be a relative path string"
                    )
                    continue
                _validate_manifest_shard_path(layout, manifest_path, modality, key, relative_path, checksums, errors)


def _validate_manifest_shard_path(
    layout: MultimodalCacheLayout,
    manifest_path: Path,
    modality: str,
    key: str,
    relative_path: str,
    checksums: dict[str, Any] | None,
    errors: list[str],
) -> None:
    shard_relative = Path(relative_path)
    manifest_name = manifest_path.relative_to(layout.root)
    if shard_relative.is_absolute():
        errors.append(f"{manifest_name} entry for {modality}.{key} must be relative, got {relative_path}")
        return
    if _contains_forbidden_model_input_modality(relative_path):
        errors.append(
            f"{manifest_name} entry for {modality}.{key} must not reference "
            f"controlled or hidden metadata: {relative_path}"
        )
        return
    shard_path = (layout.root / shard_relative).resolve()
    try:
        normalized_relative = shard_path.relative_to(layout.root.resolve())
    except ValueError:
        errors.append(f"{manifest_name} entry for {modality}.{key} escapes cache root: {relative_path}")
        return
    expected_root = TOKEN_FIELD_MANIFEST_ALLOWED_ROOTS[key]
    if not normalized_relative.parts or normalized_relative.parts[0] != expected_root:
        errors.append(
            f"{manifest_name} entry for {modality}.{key} must stay under {expected_root}/: {relative_path}"
        )
        return
    if not shard_path.exists():
        errors.append(f"{manifest_name} points to missing {key} shard for {modality}: {normalized_relative}")
        return
    checksum_key = str(normalized_relative)
    if checksums and checksum_key not in checksums:
        errors.append(f"checksums.json missing hash for token field shard: {checksum_key}")


def _validate_supervision_artifacts(
    layout: MultimodalCacheLayout,
    splits: tuple[str, ...],
    data_card: dict[str, Any],
    checksums: dict[str, Any] | None,
    errors: list[str],
) -> None:
    for split in splits:
        weak_label_path = layout.root / "supervision" / WEAK_LABEL_SUPERVISION_PATTERN.format(split=split)
        if weak_label_path.exists() and checksums:
            weak_label_relative = str(weak_label_path.relative_to(layout.root))
            if weak_label_relative not in checksums:
                errors.append(f"checksums.json missing hash for weak label artifact: {weak_label_relative}")
    if not isinstance(data_card, dict) or not data_card:
        return
    for split in splits:
        for relative in _required_supervision_artifact_names(data_card, split):
            path = layout.root / "supervision" / relative
            normalized = str(path.relative_to(layout.root))
            if not path.exists():
                errors.append(f"missing required cache artifact: {normalized}")
                continue
            if checksums and normalized not in checksums:
                errors.append(f"checksums.json missing hash for required supervision artifact: {normalized}")


def _required_supervision_artifact_names(data_card: dict[str, Any], split: str) -> tuple[str, ...]:
    required: list[str] = []
    tasks = data_card.get("tasks", [])
    if isinstance(tasks, list) and any(_requires_grounding_supervision(task) for task in tasks):
        required.extend(pattern.format(split=split) for pattern in GROUNDING_REQUIRED_SUPERVISION_PATTERNS)
    if isinstance(tasks, list) and any(_requires_sentiment_supervision(task) for task in tasks):
        required.extend(pattern.format(split=split) for pattern in SENTIMENT_REQUIRED_SUPERVISION_PATTERNS)
    operator_supervision = data_card.get("operator_supervision", {})
    if isinstance(operator_supervision, dict) and operator_supervision.get("RCEO"):
        required.extend(pattern.format(split=split) for pattern in RCEO_REQUIRED_SUPERVISION_PATTERNS)
    return tuple(required)


def _requires_grounding_supervision(task: Any) -> bool:
    if not isinstance(task, str):
        return False
    lowered = task.lower()
    return "grounding" in lowered or "phrase_region" in lowered or "region_text" in lowered


def _requires_sentiment_supervision(task: Any) -> bool:
    if not isinstance(task, str):
        return False
    lowered = task.lower()
    return "sentiment" in lowered or "emotion" in lowered or "mosei" in lowered or "meld" in lowered or "iemocap" in lowered
