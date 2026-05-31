from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from moat_ovha_torch.data.multimodal.adapters.base import (
    RawDatasetManifest,
    SupervisionShard,
    TokenFieldShard,
    ValidationReport,
    require_files,
)
from moat_ovha_torch.data.multimodal.cache_schema import (
    MultimodalCacheLayout,
    default_data_card,
    file_sha256,
    validate_cache_layout,
)
from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task


@dataclass(frozen=True)
class RefCOCOAdapter:
    name: str = "refcoco"
    version: str = "v0.1"

    def discover_raw(self, raw_root: Path) -> RawDatasetManifest:
        return require_files(
            self.name,
            raw_root,
            (
                "annotations/instances.json",
                "annotations/refs.json",
                "features/text_features.npy",
                "features/region_features.npy",
                "splits.json",
            ),
        )

    def build_index(self, manifest: RawDatasetManifest):
        return {"manifest": manifest, "requires_dataframe": True}

    def extract_token_fields(self, rows, split: str) -> dict[str, TokenFieldShard]:
        root = Path(rows["cache_root"]) if isinstance(rows, dict) and "cache_root" in rows else Path(".")
        return {
            "text": TokenFieldShard(
                "text",
                split,
                root / "token_fields" / f"text_{split}.npy",
                root / "positions" / f"text_pos_{split}.npy",
                root / "masks" / f"text_mask_{split}.npy",
            ),
            "region": TokenFieldShard(
                "region",
                split,
                root / "token_fields" / f"region_{split}.npy",
                root / "positions" / f"region_pos_{split}.npy",
                root / "masks" / f"region_mask_{split}.npy",
            ),
        }

    def extract_supervision(self, rows, split: str) -> SupervisionShard:
        root = Path(rows["cache_root"]) if isinstance(rows, dict) and "cache_root" in rows else Path(".")
        return SupervisionShard(
            split=split,
            alignment_pairs_path=root / "supervision" / f"alignment_pairs_{split}.parquet",
            bbox_targets_path=root / "supervision" / f"bbox_targets_{split}.npy",
            region_targets_path=root / "supervision" / f"region_targets_{split}.npy",
        )

    def write_cache(
        self,
        manifest: RawDatasetManifest,
        cache_root: Path,
        split: str,
        cache_version: str,
    ) -> None:
        split_source_ids = _read_split_source_ids(manifest.files["splits.json"])
        if split not in split_source_ids:
            raise ValueError(f"{self.name} raw splits.json missing split: {split}")
        records_by_source_id = _grounding_records_by_source_id(manifest)
        missing_records = sorted(set(split_source_ids[split]) - set(records_by_source_id))
        if missing_records:
            raise ValueError(
                f"{self.name} raw annotations missing sample records for split {split}: "
                + ", ".join(missing_records)
            )

        layout = MultimodalCacheLayout(cache_root, self.name, cache_version)
        root = layout.root
        _ensure_cache_dirs(root)
        _write_common_cache_files(root, manifest, self.name, cache_version, split_source_ids)
        _write_feature_versions(root, manifest)
        _write_split_cache_files(root, manifest, split, split_source_ids[split], records_by_source_id)
        _write_checksums(root)

    def validate_cache(self, cache_root: Path, cache_version: str | None = None) -> ValidationReport:
        version = cache_version or self.version
        report = validate_cache_layout(MultimodalCacheLayout(cache_root, self.name, version), splits=("train", "val", "test"))
        return ValidationReport(report.ok, report.errors, report.warnings)


def _ensure_cache_dirs(root: Path) -> None:
    for folder in ("masks", "positions", "provenance", "supervision", "token_fields"):
        (root / folder).mkdir(parents=True, exist_ok=True)


def _write_common_cache_files(
    root: Path,
    manifest: RawDatasetManifest,
    dataset_name: str,
    cache_version: str,
    split_source_ids: dict[str, list[str]],
) -> None:
    data_card = default_data_card(
        dataset_name,
        cache_version,
        ["text", "region"],
        ["phrase_region_grounding"],
    )
    (root / "data_card.json").write_text(json.dumps(data_card, sort_keys=True) + "\n")
    (root / "splits.json").write_text(json.dumps(split_source_ids, sort_keys=True) + "\n")
    (root / "raw_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": manifest.dataset_name,
                "raw_root": str(manifest.raw_root),
                "files": {name: str(path) for name, path in sorted(manifest.files.items())},
            },
            sort_keys=True,
        )
        + "\n"
    )
    (root / "samples.parquet").write_text(
        "\n".join(
            json.dumps({"source_id": source_id, "split": split}, sort_keys=True)
            for split, source_ids in sorted(split_source_ids.items())
            for source_id in source_ids
        )
        + "\n"
    )
    (root / "provenance" / "pseudo_label_versions.json").write_text(
        json.dumps({"generated_from_splits": [], "version": "none"}, sort_keys=True) + "\n"
    )


def _write_feature_versions(root: Path, manifest: RawDatasetManifest) -> None:
    text_version = "raw_sha256:" + file_sha256(manifest.files["features/text_features.npy"])
    region_version = "raw_sha256:" + file_sha256(manifest.files["features/region_features.npy"])
    reference = {"text": text_version, "region": region_version}
    baselines = {"ovha_full": dict(reference)}
    for baseline in baseline_names_for_task("phrase_region_grounding"):
        baselines[baseline] = dict(reference)
    payload = {"text": text_version, "region": region_version, "baselines": baselines}
    (root / "provenance" / "feature_versions.json").write_text(json.dumps(payload, sort_keys=True) + "\n")


def _write_split_cache_files(
    root: Path,
    manifest: RawDatasetManifest,
    split: str,
    source_ids: list[str],
    records_by_source_id: dict[str, dict[str, Any]],
) -> None:
    row_indices = _row_indices_for_source_ids(source_ids, records_by_source_id)
    feature_sources = {
        "text": manifest.files["features/text_features.npy"],
        "region": manifest.files["features/region_features.npy"],
    }
    feature_shards: dict[str, np.ndarray] = {}
    for modality, source in feature_sources.items():
        shard = _load_and_select_rows(source, row_indices, artifact_name=f"{modality} features")
        feature_shards[modality] = shard
        _write_array(root / "token_fields" / f"{modality}_{split}.npy", shard)
        _write_position_and_mask_artifacts(root, modality, split, shard)
    token_manifest = {
        "text": {
            "x": f"token_fields/text_{split}.npy",
            "pos": f"positions/text_pos_{split}.npy",
            "mask": f"masks/text_mask_{split}.npy",
        },
        "region": {
            "x": f"token_fields/region_{split}.npy",
            "pos": f"positions/region_pos_{split}.npy",
            "mask": f"masks/region_mask_{split}.npy",
        },
    }
    (root / "token_fields" / f"manifest_{split}.json").write_text(
        json.dumps(token_manifest, sort_keys=True) + "\n"
    )
    (root / "provenance" / f"source_ids_{split}.txt").write_text("\n".join(source_ids) + "\n")
    sample_records = [_sample_record(records_by_source_id[source_id], split) for source_id in source_ids]
    (root / "provenance" / f"sample_records_{split}.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in sample_records) + "\n"
    )
    _write_failed_sample_manifest(
        root / "provenance" / f"failed_samples_{split}.jsonl",
        manifest,
        split,
        retained_source_ids=set(source_ids),
    )
    selected_records = [records_by_source_id[source_id] for source_id in source_ids]
    candidate_region_count = _candidate_region_count(feature_shards["region"], split)
    target_region_indices = _target_region_indices(selected_records, candidate_region_count)
    _write_array(
        root / "supervision" / f"task_labels_{split}.npy",
        _region_distribution_targets(target_region_indices, candidate_region_count),
    )
    _write_alignment_pairs(
        root / "supervision" / f"alignment_pairs_{split}.parquet",
        split,
        selected_records,
        target_region_indices,
    )
    _write_array(root / "supervision" / f"bbox_targets_{split}.npy", _bbox_targets(selected_records))
    _write_array(root / "supervision" / f"region_targets_{split}.npy", target_region_indices.reshape(-1, 1))
    _write_corruption_metadata(root / "supervision" / f"corruption_{split}.parquet", split, source_ids)


def _write_array(destination: Path, array: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, array)


def _write_position_and_mask_artifacts(root: Path, modality: str, split: str, shard: np.ndarray) -> None:
    if shard.ndim < 2:
        raise ValueError(f"{modality} feature shard for split {split} must have at least [sample, token] axes")
    sample_count = int(shard.shape[0])
    token_count = int(shard.shape[1])
    positions = np.broadcast_to(
        np.arange(token_count, dtype=np.float32).reshape(1, token_count, 1),
        (sample_count, token_count, 1),
    ).copy()
    mask = np.ones((sample_count, token_count), dtype=bool)
    _write_array(root / "positions" / f"{modality}_pos_{split}.npy", positions)
    _write_array(root / "masks" / f"{modality}_mask_{split}.npy", mask)


def _load_and_select_rows(source: Path, row_indices: list[int], *, artifact_name: str) -> np.ndarray:
    array = np.load(source, allow_pickle=False)
    if array.ndim == 0:
        raise ValueError(f"{artifact_name} must have a sample axis: {source}")
    sample_count = int(array.shape[0])
    if any(row_index < 0 or row_index >= sample_count for row_index in row_indices):
        raise ValueError(f"{artifact_name} row index exceeds available rows in {source}")
    return np.asarray(array[row_indices]).copy()


def _candidate_region_count(region_shard: np.ndarray, split: str) -> int:
    if region_shard.ndim < 2:
        raise ValueError(f"region feature shard for split {split} must have candidate region axis")
    candidate_count = int(region_shard.shape[1])
    if candidate_count <= 0:
        raise ValueError(f"region feature shard for split {split} must contain candidate regions")
    return candidate_count


def _target_region_indices(records: list[dict[str, Any]], candidate_region_count: int) -> np.ndarray:
    targets: list[int] = []
    for record in records:
        value = record.get("target_region_index", 0)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"target_region_index must be an integer for source_id: {record.get('source_id')}")
        if value < 0 or value >= candidate_region_count:
            raise ValueError(
                f"target_region_index out of range for source_id: {record.get('source_id')} "
                f"(value={value}, candidate_count={candidate_region_count})"
            )
        targets.append(value)
    return np.asarray(targets, dtype=np.int64)


def _region_distribution_targets(target_region_indices: np.ndarray, candidate_region_count: int) -> np.ndarray:
    targets = np.zeros((int(target_region_indices.shape[0]), candidate_region_count), dtype=np.float32)
    targets[np.arange(int(target_region_indices.shape[0])), target_region_indices] = 1.0
    return targets


def _write_alignment_pairs(
    destination: Path,
    split: str,
    records: list[dict[str, Any]],
    target_region_indices: np.ndarray,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row_index, record in enumerate(records):
        rows.append(
            {
                "source_id": record["source_id"],
                "split": split,
                "phrase_span": record["phrase_span"],
                "target_region_index": int(target_region_indices[row_index]),
                "row_index": row_index,
            }
        )
    destination.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _bbox_targets(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(value) for value in record["region_box"]] for record in records],
        dtype=np.float32,
    )


def _write_corruption_metadata(destination: Path, split: str, source_ids: list[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "source_ids": source_ids,
        "corruption": "none",
    }
    destination.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _write_failed_sample_manifest(
    destination: Path,
    manifest: RawDatasetManifest,
    split: str,
    *,
    retained_source_ids: set[str],
) -> None:
    rows = _failed_sample_rows_for_split(manifest, split)
    overlaps = sorted(row["source_id"] for row in rows if row["source_id"] in retained_source_ids)
    if overlaps:
        raise ValueError(
            f"{manifest.dataset_name} failed sample manifest overlaps retained split {split}: "
            + ", ".join(overlaps)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))


def _failed_sample_rows_for_split(manifest: RawDatasetManifest, split: str) -> list[dict[str, str]]:
    path = _failed_sample_manifest_path(manifest)
    if path is None:
        return []
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {line_number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name} line {line_number} must be a JSON object")
        row = _normalize_failed_sample_row(path.name, line_number, payload)
        if row["split"] == split:
            rows.append(row)
    return rows


def _failed_sample_manifest_path(manifest: RawDatasetManifest) -> Path | None:
    for relative in ("provenance/failed_samples.jsonl", "annotations/failed_samples.jsonl"):
        path = manifest.raw_root / relative
        if path.exists():
            return path
    return None


def _normalize_failed_sample_row(source_name: str, line_number: int, payload: dict[str, Any]) -> dict[str, str]:
    required = ("source_id", "split", "reason")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{source_name} line {line_number} missing required keys: {', '.join(missing)}")
    normalized: dict[str, str] = {}
    for key in required:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{source_name} line {line_number} {key} must be a non-empty normalized string")
        normalized[key] = value
    return normalized


def _write_checksums(root: Path) -> None:
    checksums = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True) + "\n")


def _read_split_source_ids(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("raw splits.json must be an object keyed by split")
    split_source_ids: dict[str, list[str]] = {}
    for split, source_ids in sorted(payload.items()):
        if not isinstance(split, str) or not split.strip() or split != split.strip():
            raise ValueError("raw splits.json split names must be non-empty normalized strings")
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError(f"raw splits.json {split} must be a non-empty source_id list")
        invalid = [
            source_id
            for source_id in source_ids
            if not isinstance(source_id, str) or not source_id.strip() or source_id != source_id.strip()
        ]
        if invalid:
            raise ValueError(f"raw splits.json {split} contains invalid source_id entries")
        split_source_ids[split] = list(source_ids)
    return split_source_ids


def _grounding_records_by_source_id(manifest: RawDatasetManifest) -> dict[str, dict[str, Any]]:
    path = _annotation_record_path(manifest)
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        records = payload.get("records")
    else:
        records = payload
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path.name} must contain a non-empty records list")
    by_source_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path.name} records[{index}] must be an object")
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip() or source_id != source_id.strip():
            raise ValueError(f"{path.name} records[{index}] source_id must be a normalized string")
        if source_id in by_source_id:
            raise ValueError(f"{path.name} contains duplicate source_id: {source_id}")
        _validate_grounding_record(path.name, index, record)
        by_source_id[source_id] = {**record, "_cache_row_index": index}
    return by_source_id


def _row_indices_for_source_ids(
    source_ids: list[str],
    records_by_source_id: dict[str, dict[str, Any]],
) -> list[int]:
    row_indices: list[int] = []
    for source_id in source_ids:
        row_index = records_by_source_id[source_id].get("_cache_row_index")
        if not isinstance(row_index, int) or isinstance(row_index, bool):
            raise ValueError(f"annotation records missing cache row index for source_id: {source_id}")
        row_indices.append(row_index)
    return row_indices


def _annotation_record_path(manifest: RawDatasetManifest) -> Path:
    for relative in (
        "annotations/refs.json",
        "annotations/phrase_regions.json",
        "annotations/region_descriptions.json",
    ):
        path = manifest.files.get(relative)
        if path is not None:
            return path
    raise ValueError(f"{manifest.dataset_name} raw manifest has no supported grounding annotation records")


def _validate_grounding_record(source_name: str, index: int, record: dict[str, Any]) -> None:
    required = (
        "source_id",
        "split",
        "original_split",
        "raw_ref",
        "license_tag",
        "preprocessing_version",
        "image_id",
        "caption_id",
        "phrase_span",
        "region_box",
        "candidate_region_source",
        "box_coordinate_convention",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"{source_name} records[{index}] missing required keys: {', '.join(missing)}")
    for key in (
        "source_id",
        "split",
        "original_split",
        "raw_ref",
        "license_tag",
        "preprocessing_version",
        "image_id",
        "caption_id",
        "candidate_region_source",
        "box_coordinate_convention",
    ):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{source_name} records[{index}] {key} must be a non-empty normalized string")
    phrase_span = record.get("phrase_span")
    if not (
        isinstance(phrase_span, dict)
        and isinstance(phrase_span.get("start"), int)
        and isinstance(phrase_span.get("end"), int)
        and not isinstance(phrase_span.get("start"), bool)
        and not isinstance(phrase_span.get("end"), bool)
        and 0 <= phrase_span["start"] < phrase_span["end"]
    ):
        raise ValueError(f"{source_name} records[{index}] phrase_span must contain integer start/end")
    region_box = record.get("region_box")
    if not isinstance(region_box, list) or len(region_box) != 4:
        raise ValueError(f"{source_name} records[{index}] region_box must contain four coordinates")
    for coordinate in region_box:
        try:
            float(coordinate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source_name} records[{index}] region_box must be numeric") from exc
    target_region_index = record.get("target_region_index")
    if target_region_index is not None and (
        not isinstance(target_region_index, int) or isinstance(target_region_index, bool) or target_region_index < 0
    ):
        raise ValueError(f"{source_name} records[{index}] target_region_index must be a non-negative integer")


def _sample_record(record: dict[str, Any], split: str) -> dict[str, Any]:
    if record.get("split") != split:
        raise ValueError(f"record split does not match requested split for source_id: {record.get('source_id')}")
    return {
        "source_id": record["source_id"],
        "split": split,
        "original_split": record["original_split"],
        "raw_ref": record["raw_ref"],
        "license_tag": record["license_tag"],
        "preprocessing_version": record["preprocessing_version"],
        "image_id": record["image_id"],
        "caption_id": record["caption_id"],
        "phrase_span": record["phrase_span"],
        "region_box": [float(value) for value in record["region_box"]],
        "target_region_index": int(record.get("target_region_index", 0)),
        "candidate_region_source": record["candidate_region_source"],
        "box_coordinate_convention": record["box_coordinate_convention"],
    }
