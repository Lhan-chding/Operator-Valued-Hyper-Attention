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
        return SupervisionShard(split=split, alignment_pairs_path=root / f"alignment_pairs_{split}.parquet", bbox_targets_path=root / f"bbox_targets_{split}.npy")

    def write_cache(self, cache_root: Path, split: str) -> None:
        raise NotImplementedError("RefCOCO cache writing requires raw annotations and frozen feature files; use scripts/multimodal/build_cache.py")

    def validate_cache(self, cache_root: Path) -> ValidationReport:
        report = validate_cache_layout(MultimodalCacheLayout(cache_root, self.name, self.version), splits=("train", "val", "test"))
        return ValidationReport(report.ok, report.errors, report.warnings)
