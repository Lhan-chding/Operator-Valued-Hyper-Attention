from __future__ import annotations

import hashlib
import json
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
    if data_card_path.exists():
        try:
            data_card = json.loads(data_card_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid data_card.json: {exc}")
        else:
            for key in REQUIRED_DATA_CARD_KEYS:
                if key not in data_card:
                    errors.append(f"data_card.json missing required key: {key}")
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
            checksums = json.loads(checksums_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid checksums.json: {exc}")
        else:
            if not isinstance(checksums, dict) or not checksums:
                warnings.append("checksums.json is empty; formal runs require file hashes")
            else:
                _validate_checksum_coverage(layout, required_files, checksums, errors)

    _validate_source_split_controls(layout, splits, errors)
    _validate_feature_parity(layout, data_card, errors)
    _validate_pseudo_label_provenance(layout, errors)

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
    for path in sorted(required_files):
        if not path.exists():
            continue
        relative = str(path.relative_to(layout.root))
        if relative not in checksums:
            errors.append(f"checksums.json missing hash for required artifact: {relative}")


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
        return
    reference = None
    for model_name, versions in sorted(baselines.items()):
        if reference is None:
            reference = dict(versions)
            continue
        if dict(versions) != reference:
            errors.append(
                "same_features_for_baselines is true but baseline feature versions differ: "
                f"{model_name}"
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
