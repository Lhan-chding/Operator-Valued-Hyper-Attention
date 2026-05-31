from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

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
            "text": TokenFieldShard("text", split, root / f"text_{split}.npy", root / f"text_pos_{split}.npy", root / f"text_mask_{split}.npy"),
            "region": TokenFieldShard(
                "region",
                split,
                root / f"region_{split}.npy",
                root / f"region_pos_{split}.npy",
                root / f"region_mask_{split}.npy",
            ),
        }

    def extract_supervision(self, rows, split: str) -> SupervisionShard:
        root = Path(rows["cache_root"]) if isinstance(rows, dict) and "cache_root" in rows else Path(".")
        return SupervisionShard(
            split=split,
            alignment_pairs_path=root / f"alignment_pairs_{split}.parquet",
            bbox_targets_path=root / f"bbox_targets_{split}.npy",
            region_targets_path=root / f"region_targets_{split}.npy",
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
    _copy_feature_shard(manifest.files["features/text_features.npy"], root / "token_fields" / f"text_{split}.npy")
    _copy_feature_shard(
        manifest.files["features/region_features.npy"],
        root / "token_fields" / f"region_{split}.npy",
    )
    for modality in ("text", "region"):
        (root / "positions" / f"{modality}_pos_{split}.npy").write_text(
            f"{modality} positions generated from raw {manifest.dataset_name} {split}\n"
        )
        (root / "masks" / f"{modality}_mask_{split}.npy").write_text(
            f"{modality} mask generated from raw {manifest.dataset_name} {split}\n"
        )
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
    (root / "provenance" / f"failed_samples_{split}.jsonl").write_text("")
    for name in (
        f"task_labels_{split}.npy",
        f"alignment_pairs_{split}.parquet",
        f"bbox_targets_{split}.npy",
        f"region_targets_{split}.npy",
        f"corruption_{split}.parquet",
    ):
        (root / "supervision" / name).write_text(f"{name} generated from raw {manifest.dataset_name}\n")


def _copy_feature_shard(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


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
        by_source_id[source_id] = record
    return by_source_id


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
        "candidate_region_source": record["candidate_region_source"],
        "box_coordinate_convention": record["box_coordinate_convention"],
    }
