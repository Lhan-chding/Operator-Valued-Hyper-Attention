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
    for path in sorted(required_cache_files(layout, splits)):
        if not path.exists():
            errors.append(f"missing required cache artifact: {path.relative_to(layout.root)}")

    data_card_path = layout.root / "data_card.json"
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
