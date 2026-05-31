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
class CMUMOSEIAdapter:
    name: str = "cmu_mosei"
    version: str = "v0.1"

    def discover_raw(self, raw_root: Path) -> RawDatasetManifest:
        return require_files(
            self.name,
            raw_root,
            (
                "features/text_features.npy",
                "features/audio_features.npy",
                "features/visual_features.npy",
                "labels/sentiment.npy",
                "labels/emotion.npy",
                "metadata/utterances.json",
                "metadata/dialogues.json",
                "metadata/feature_versions.json",
                "metadata/missing_modality_mask.npy",
                "metadata/corruption_transforms.json",
                "splits.json",
            ),
        )

    def build_index(self, manifest: RawDatasetManifest):
        return {"manifest": manifest, "requires_dataframe": True}

    def extract_token_fields(self, rows, split: str) -> dict[str, TokenFieldShard]:
        root = Path(rows["cache_root"]) if isinstance(rows, dict) and "cache_root" in rows else Path(".")
        return {
            "text": TokenFieldShard("text", split, root / f"text_{split}.npy", root / f"text_pos_{split}.npy", root / f"text_mask_{split}.npy"),
            "audio": TokenFieldShard("audio", split, root / f"audio_{split}.npy", root / f"audio_pos_{split}.npy", root / f"audio_mask_{split}.npy"),
            "vision": TokenFieldShard(
                "vision",
                split,
                root / f"vision_{split}.npy",
                root / f"vision_pos_{split}.npy",
                root / f"vision_mask_{split}.npy",
            ),
        }

    def extract_supervision(self, rows, split: str) -> SupervisionShard:
        root = Path(rows["cache_root"]) if isinstance(rows, dict) and "cache_root" in rows else Path(".")
        return SupervisionShard(
            split=split,
            task_label_path=root / f"sentiment_{split}.npy",
            modality_missing_mask_path=root / f"missing_modality_mask_{split}.npy",
            corruption_metadata_path=root / f"corruption_{split}.parquet",
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
        records_by_source_id = _utterance_records_by_source_id(manifest)
        missing_records = sorted(set(split_source_ids[split]) - set(records_by_source_id))
        if missing_records:
            raise ValueError(
                f"{self.name} raw utterance metadata missing sample records for split {split}: "
                + ", ".join(missing_records)
            )

        layout = MultimodalCacheLayout(cache_root, self.name, cache_version)
        root = layout.root
        _ensure_cache_dirs(root)
        speaker_id_available = _speaker_id_available(records_by_source_id)
        _write_common_cache_files(
            root,
            manifest,
            self.name,
            cache_version,
            split_source_ids,
            speaker_id_available=speaker_id_available,
        )
        _write_feature_versions(root, manifest, self.name)
        _write_split_cache_files(
            root,
            manifest,
            split,
            split_source_ids[split],
            records_by_source_id,
            speaker_id_available=speaker_id_available,
        )
        _write_checksums(root)

    def validate_cache(self, cache_root: Path) -> ValidationReport:
        report = validate_cache_layout(MultimodalCacheLayout(cache_root, self.name, self.version), splits=("train", "val", "test"))
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
    *,
    speaker_id_available: bool,
) -> None:
    data_card = default_data_card(
        dataset_name,
        cache_version,
        ["text", "audio", "vision"],
        ["sentiment_regression", "emotion_classification"],
    )
    if speaker_id_available:
        data_card["metadata_availability"] = {"speaker_id": True}
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


def _write_feature_versions(root: Path, manifest: RawDatasetManifest, dataset_name: str) -> None:
    raw_versions = _raw_feature_versions(manifest)
    reference = {
        "text": raw_versions["text"],
        "audio": raw_versions["audio"],
        "vision": raw_versions["vision"],
    }
    baselines = {"ovha_full": dict(reference)}
    for baseline in baseline_names_for_task(dataset_name):
        baselines[baseline] = dict(reference)
    payload = {**reference, "baselines": baselines}
    (root / "provenance" / "feature_versions.json").write_text(json.dumps(payload, sort_keys=True) + "\n")


def _raw_feature_versions(manifest: RawDatasetManifest) -> dict[str, str]:
    path = manifest.files["metadata/feature_versions.json"]
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "text": _feature_version_or_hash(payload, "text", manifest.files["features/text_features.npy"]),
        "audio": _feature_version_or_hash(payload, "audio", manifest.files["features/audio_features.npy"]),
        "vision": _feature_version_or_hash(
            payload,
            "vision",
            manifest.files["features/visual_features.npy"],
            alias="visual",
        ),
    }


def _feature_version_or_hash(payload: dict[str, Any], key: str, path: Path, *, alias: str | None = None) -> str:
    value = payload.get(key)
    if not isinstance(value, str) and alias is not None:
        value = payload.get(alias)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "raw_sha256:" + file_sha256(path)


def _write_split_cache_files(
    root: Path,
    manifest: RawDatasetManifest,
    split: str,
    source_ids: list[str],
    records_by_source_id: dict[str, dict[str, Any]],
    *,
    speaker_id_available: bool,
) -> None:
    row_indices = _row_indices_for_source_ids(source_ids, records_by_source_id)
    feature_sources = {
        "text": manifest.files["features/text_features.npy"],
        "audio": manifest.files["features/audio_features.npy"],
        "vision": manifest.files["features/visual_features.npy"],
    }
    for modality, source in feature_sources.items():
        shard = _load_and_select_rows(source, row_indices, artifact_name=f"{modality} features")
        _write_array(root / "token_fields" / f"{modality}_{split}.npy", shard)
        _write_position_and_mask_artifacts(root, modality, split, shard)
    token_manifest = {
        modality: {
            "x": f"token_fields/{modality}_{split}.npy",
            "pos": f"positions/{modality}_pos_{split}.npy",
            "mask": f"masks/{modality}_mask_{split}.npy",
        }
        for modality in ("text", "audio", "vision")
    }
    (root / "token_fields" / f"manifest_{split}.json").write_text(
        json.dumps(token_manifest, sort_keys=True) + "\n"
    )
    (root / "provenance" / f"source_ids_{split}.txt").write_text("\n".join(source_ids) + "\n")
    sample_records = [
        _sample_record(records_by_source_id[source_id], split, speaker_id_available=speaker_id_available)
        for source_id in source_ids
    ]
    (root / "provenance" / f"sample_records_{split}.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in sample_records) + "\n"
    )
    (root / "provenance" / f"failed_samples_{split}.jsonl").write_text("")
    task_labels = _load_and_select_rows(_label_source(manifest), row_indices, artifact_name="task labels")
    _write_array(root / "supervision" / f"task_labels_{split}.npy", task_labels)
    _write_optional_label_array(
        manifest.files.get("labels/emotion.npy"),
        root / "supervision" / f"emotion_labels_{split}.npy",
        row_indices,
    )
    missing_mask = _load_and_select_rows(
        manifest.files["metadata/missing_modality_mask.npy"],
        row_indices,
        artifact_name="missing modality mask",
    )
    _write_array(root / "supervision" / f"missing_modality_mask_{split}.npy", missing_mask)
    _write_split_corruption_metadata(
        manifest.files["metadata/corruption_transforms.json"],
        root / "supervision" / f"corruption_{split}.parquet",
        split,
        source_ids,
    )


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


def _write_optional_label_array(source: Path | None, destination: Path, row_indices: list[int]) -> None:
    if source is None:
        return
    label = _load_and_select_rows(source, row_indices, artifact_name=destination.name)
    _write_array(destination, label)


def _load_and_select_rows(source: Path, row_indices: list[int], *, artifact_name: str) -> np.ndarray:
    array = np.load(source, allow_pickle=False)
    if array.ndim == 0:
        raise ValueError(f"{artifact_name} must have a sample axis: {source}")
    sample_count = int(array.shape[0])
    if any(row_index < 0 or row_index >= sample_count for row_index in row_indices):
        raise ValueError(f"{artifact_name} row index exceeds available rows in {source}")
    return np.asarray(array[row_indices]).copy()


def _write_split_corruption_metadata(source: Path, destination: Path, split: str, source_ids: list[str]) -> None:
    try:
        raw_payload = json.loads(source.read_text())
    except json.JSONDecodeError:
        raw_payload = {"raw_ref": str(source)}
    payload = {
        "split": split,
        "source_ids": source_ids,
        "source": str(source),
        "raw_corruption_metadata": raw_payload,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _label_source(manifest: RawDatasetManifest) -> Path:
    return manifest.files.get("labels/sentiment.npy") or manifest.files["labels/emotion.npy"]


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


def _utterance_records_by_source_id(manifest: RawDatasetManifest) -> dict[str, dict[str, Any]]:
    path = manifest.files["metadata/utterances.json"]
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        records = payload.get("records")
    else:
        records = payload
    if not isinstance(records, list) or not records:
        raise ValueError("metadata/utterances.json must contain a non-empty records list")
    by_source_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"metadata/utterances.json records[{index}] must be an object")
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip() or source_id != source_id.strip():
            raise ValueError(f"metadata/utterances.json records[{index}] source_id must be a normalized string")
        if source_id in by_source_id:
            raise ValueError(f"metadata/utterances.json contains duplicate source_id: {source_id}")
        _validate_utterance_record(index, record)
        by_source_id[source_id] = {**record, "_cache_row_index": index}
    return by_source_id


def _validate_utterance_record(index: int, record: dict[str, Any]) -> None:
    required = (
        "source_id",
        "split",
        "original_split",
        "raw_ref",
        "license_tag",
        "preprocessing_version",
        "utterance_id",
        "dialogue_id",
        "transcript_source",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"metadata/utterances.json records[{index}] missing required keys: {', '.join(missing)}")
    for key in required:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"metadata/utterances.json records[{index}] {key} must be a non-empty normalized string")
    speaker_id = record.get("speaker_id")
    if speaker_id is not None and (
        not isinstance(speaker_id, str) or not speaker_id.strip() or speaker_id != speaker_id.strip()
    ):
        raise ValueError(f"metadata/utterances.json records[{index}] speaker_id must be a non-empty normalized string")


def _speaker_id_available(records_by_source_id: dict[str, dict[str, Any]]) -> bool:
    records = list(records_by_source_id.values())
    present = ["speaker_id" in record for record in records]
    if not any(present):
        return False
    if not all(present):
        raise ValueError("metadata/utterances.json speaker_id must be present for every record when provided")
    return True


def _row_indices_for_source_ids(
    source_ids: list[str],
    records_by_source_id: dict[str, dict[str, Any]],
) -> list[int]:
    row_indices: list[int] = []
    for source_id in source_ids:
        row_index = records_by_source_id[source_id].get("_cache_row_index")
        if not isinstance(row_index, int) or isinstance(row_index, bool):
            raise ValueError(f"metadata/utterances.json missing cache row index for source_id: {source_id}")
        row_indices.append(row_index)
    return row_indices


def _sample_record(record: dict[str, Any], split: str, *, speaker_id_available: bool) -> dict[str, Any]:
    if record.get("split") != split:
        raise ValueError(f"record split does not match requested split for source_id: {record.get('source_id')}")
    payload = {
        "source_id": record["source_id"],
        "split": split,
        "original_split": record["original_split"],
        "raw_ref": record["raw_ref"],
        "license_tag": record["license_tag"],
        "preprocessing_version": record["preprocessing_version"],
        "utterance_id": record["utterance_id"],
        "dialogue_id": record["dialogue_id"],
        "transcript_source": record["transcript_source"],
        "missing_modality_mask_ref": f"supervision/missing_modality_mask_{split}.npy",
        "corruption_metadata_ref": f"supervision/corruption_{split}.parquet",
    }
    if speaker_id_available:
        payload["speaker_id"] = record["speaker_id"]
    return payload
