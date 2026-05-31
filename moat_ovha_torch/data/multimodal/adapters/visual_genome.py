from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moat_ovha_torch.data.multimodal.adapters.base import RawDatasetManifest, require_files
from moat_ovha_torch.data.multimodal.adapters.refcoco import RefCOCOAdapter


@dataclass(frozen=True)
class VisualGenomeAdapter(RefCOCOAdapter):
    name: str = "visual_genome"

    def discover_raw(self, raw_root: Path) -> RawDatasetManifest:
        return require_files(
            self.name,
            raw_root,
            (
                "annotations/region_descriptions.json",
                "annotations/objects.json",
                "annotations/relationships.json",
                "features/text_features.npy",
                "features/region_features.npy",
                "splits.json",
            ),
        )
