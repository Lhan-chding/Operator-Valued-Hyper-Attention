from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moat_ovha_torch.data.multimodal.adapters.base import RawDatasetManifest, require_files
from moat_ovha_torch.data.multimodal.adapters.cmu_mosei import CMUMOSEIAdapter


@dataclass(frozen=True)
class CMUMOSIAdapter(CMUMOSEIAdapter):
    name: str = "cmu_mosi"

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
