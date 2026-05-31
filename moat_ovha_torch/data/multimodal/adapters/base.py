from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class MissingMultimodalDataError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class RawDatasetManifest:
    dataset_name: str
    raw_root: Path
    files: dict[str, Path]
    missing_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class TokenFieldShard:
    modality: str
    split: str
    x_path: Path
    pos_path: Path
    mask_path: Path
    quality_path: Path | None = None


@dataclass(frozen=True)
class SupervisionShard:
    split: str
    task_label_path: Path | None = None
    alignment_pairs_path: Path | None = None
    bbox_targets_path: Path | None = None
    region_targets_path: Path | None = None
    timestamp_targets_path: Path | None = None
    modality_missing_mask_path: Path | None = None
    corruption_metadata_path: Path | None = None
    weak_labels_path: Path | None = None


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


class MultimodalDatasetAdapter(Protocol):
    name: str
    version: str

    def discover_raw(self, raw_root: Path) -> RawDatasetManifest: ...

    def build_index(self, manifest: RawDatasetManifest) -> Any: ...

    def extract_token_fields(self, rows: Any, split: str) -> dict[str, TokenFieldShard]: ...

    def extract_supervision(self, rows: Any, split: str) -> SupervisionShard: ...

    def write_cache(self, cache_root: Path, split: str) -> None: ...

    def validate_cache(self, cache_root: Path) -> ValidationReport: ...


def require_files(dataset_name: str, raw_root: Path, relative_paths: tuple[str, ...]) -> RawDatasetManifest:
    files = {item: raw_root / item for item in relative_paths}
    missing = tuple(path for path in files.values() if not path.exists())
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise MissingMultimodalDataError(f"{dataset_name} raw data is incomplete; missing: {names}")
    return RawDatasetManifest(dataset_name=dataset_name, raw_root=raw_root, files=files)
