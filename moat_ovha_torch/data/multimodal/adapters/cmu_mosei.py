from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moat_ovha_torch.data.multimodal.adapters.base import (
    RawDatasetManifest,
    SupervisionShard,
    TokenFieldShard,
    ValidationReport,
    require_files,
)
from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout


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

    def write_cache(self, cache_root: Path, split: str) -> None:
        raise NotImplementedError("CMU-MOSEI cache writing requires frozen text/audio/visual features; use scripts/multimodal/build_cache.py")

    def validate_cache(self, cache_root: Path) -> ValidationReport:
        report = validate_cache_layout(MultimodalCacheLayout(cache_root, self.name, self.version), splits=("train", "val", "test"))
        return ValidationReport(report.ok, report.errors, report.warnings)
