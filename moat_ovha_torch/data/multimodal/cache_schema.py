from __future__ import annotations

import hashlib
import json
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
SAMPLE_RECORD_MANIFEST_REQUIRED_KEYS = ("source_id", "split", "raw_ref", "license_tag")
TOKEN_FIELD_MANIFEST_REQUIRED_KEYS = ("x", "pos", "mask")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CHECKSUM_MANIFEST_NAME = "checksums.json"


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
            for key in REQUIRED_DATA_CARD_KEYS:
                if key not in data_card:
                    errors.append(f"data_card.json missing required key: {key}")
            _validate_data_card_identity(layout, data_card, errors)
            _validate_data_card_string_list(data_card, "modalities", errors)
            _validate_data_card_string_list(data_card, "tasks", errors)
            _validate_operator_supervision(data_card.get("operator_supervision"), errors)
            controls = data_card.get("leakage_controls", {})
            for key in (
                "split_by_source_id",
                "deduplicate_by_source_id",
                "pseudo_labels_generated_without_test_labels",
                "same_features_for_baselines",
            ):
                if controls.get(key) is not True:
                    errors.append(f"data_card.json leakage_controls.{key} must be true")

    checksums_path = layout.root / "checksums.json"
    if checksums_path.exists():
        try:
            loaded_checksums = json.loads(checksums_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid checksums.json: {exc}")
        else:
            if not isinstance(loaded_checksums, dict) or not loaded_checksums:
                warnings.append("checksums.json is empty; formal runs require file hashes")
            else:
                checksums = loaded_checksums
                _validate_checksum_coverage(layout, required_files, checksums, errors)

    _validate_source_split_controls(layout, splits, errors)
    _validate_split_manifest_consistency(layout, splits, errors)
    _validate_feature_parity(layout, data_card, errors)
    _validate_pseudo_label_provenance(layout, errors)
    _validate_sample_record_manifests(layout, splits, errors)
    _validate_failed_sample_manifests(layout, splits, errors)
    _validate_token_field_manifests(layout, splits, data_card, checksums, errors)

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
        path = layout.root / relative
        if path.exists():
            _validate_checksum_value(relative, path, digest, errors)
    for path in sorted(required_files):
        if not path.exists():
            continue
        relative = str(path.relative_to(layout.root))
        if relative == CHECKSUM_MANIFEST_NAME:
            continue
        if relative not in checksums:
            errors.append(f"checksums.json missing hash for required artifact: {relative}")


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
        if not operator_supervision.get(operator):
            errors.append(f"data_card.json operator_supervision missing required operator: {operator}")


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
    if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values):
        errors.append(f"data_card.json {key} must be a non-empty list of strings")
        return
    if len(values) != len(set(values)):
        errors.append(f"data_card.json {key} must not contain duplicate entries")


def _validate_source_split_controls(layout: MultimodalCacheLayout, splits: tuple[str, ...], errors: list[str]) -> None:
    seen: dict[str, str] = {}
    for split in splits:
        path = layout.root / "provenance" / f"source_ids_{split}.txt"
        if not path.exists():
            continue
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
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
    for split in splits:
        expected = payload.get(split)
        if not isinstance(expected, list):
            errors.append(f"splits.json missing source_id list for split: {split}")
            continue
        source_path = layout.root / "provenance" / f"source_ids_{split}.txt"
        if not source_path.exists():
            continue
        actual = [line.strip() for line in source_path.read_text().splitlines() if line.strip()]
        if set(str(source_id) for source_id in expected) != set(actual):
            errors.append(f"provenance/source_ids_{split}.txt must match splits.json {split} entries")


def _validate_feature_parity(layout: MultimodalCacheLayout, data_card: dict[str, Any], errors: list[str]) -> None:
    controls = data_card.get("leakage_controls", {}) if isinstance(data_card, dict) else {}
    if controls.get("same_features_for_baselines") is not True:
        return
    path = layout.root / "provenance" / "feature_versions.json"
    if not path.exists():
        return
    try:
        feature_versions = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"invalid feature_versions.json: {exc}")
        return
    baselines = feature_versions.get("baselines", {})
    if not isinstance(baselines, dict) or not baselines:
        errors.append("feature_versions.json baselines must be a non-empty object when same_features_for_baselines is true")
        return
    if "ovha_full" not in baselines:
        errors.append("feature_versions.json baselines must include ovha_full reference")
        return
    modalities = data_card.get("modalities", [])
    if isinstance(modalities, list):
        for modality in modalities:
            if str(modality) not in feature_versions:
                errors.append(f"feature_versions.json missing feature extractor version for modality: {modality}")
    ovha_reference = baselines["ovha_full"]
    if not isinstance(ovha_reference, dict):
        errors.append("feature_versions.json baselines.ovha_full must be an object")
        return
    reference = dict(ovha_reference)
    if isinstance(modalities, list):
        for modality in modalities:
            modality_name = str(modality)
            if not reference.get(modality_name):
                errors.append(
                    "feature_versions.json baselines.ovha_full missing "
                    f"feature extractor version for modality: {modality_name}"
                )
    for model_name, versions in sorted(baselines.items()):
        if model_name == "ovha_full":
            continue
        if dict(versions) != reference:
            errors.append(
                "same_features_for_baselines is true but "
                f"{model_name} differs from ovha_full"
            )


def _validate_pseudo_label_provenance(layout: MultimodalCacheLayout, errors: list[str]) -> None:
    path = layout.root / "provenance" / "pseudo_label_versions.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"invalid pseudo_label_versions.json: {exc}")
        return
    generated_from = payload.get("generated_from_splits", [])
    if "test" in set(generated_from):
        errors.append("pseudo labels must not be generated from test split")


def _validate_failed_sample_manifests(
    layout: MultimodalCacheLayout,
    splits: tuple[str, ...],
    errors: list[str],
) -> None:
    for split in splits:
        path = layout.root / "provenance" / f"failed_samples_{split}.jsonl"
        if not path.exists():
            continue
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
            missing = [key for key in FAILED_SAMPLE_MANIFEST_REQUIRED_KEYS if not payload.get(key)]
            if missing:
                errors.append(f"{path.name} line {line_number} missing required keys: {', '.join(missing)}")
            if payload.get("split") != split:
                errors.append(f"{path.name} line {line_number} split must match {split}")


def _validate_sample_record_manifests(
    layout: MultimodalCacheLayout,
    splits: tuple[str, ...],
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
            missing = [key for key in SAMPLE_RECORD_MANIFEST_REQUIRED_KEYS if not payload.get(key)]
            if missing:
                errors.append(f"{path.name} line {line_number} missing required keys: {', '.join(missing)}")
            if payload.get("split") != split:
                errors.append(f"{path.name} line {line_number} split must match {split}")
            source_id = payload.get("source_id")
            if source_id:
                normalized_source_id = str(source_id)
                if normalized_source_id in seen_source_ids:
                    errors.append(f"duplicate source_id within {path.name}: {normalized_source_id}")
                seen_source_ids.add(normalized_source_id)
                observed_source_ids.append(normalized_source_id)
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


def _read_source_ids(layout: MultimodalCacheLayout, split: str) -> list[str] | None:
    path = layout.root / "provenance" / f"source_ids_{split}.txt"
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


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
    shard_path = (layout.root / shard_relative).resolve()
    try:
        normalized_relative = shard_path.relative_to(layout.root.resolve())
    except ValueError:
        errors.append(f"{manifest_name} entry for {modality}.{key} escapes cache root: {relative_path}")
        return
    if not shard_path.exists():
        errors.append(f"{manifest_name} points to missing {key} shard for {modality}: {normalized_relative}")
        return
    checksum_key = str(normalized_relative)
    if checksums and checksum_key not in checksums:
        errors.append(f"checksums.json missing hash for token field shard: {checksum_key}")
