from dataclasses import dataclass
from pathlib import Path

from moat_ovha_torch.data.multimodal.adapters.base import RawDatasetManifest, require_files
from moat_ovha_torch.data.multimodal.adapters.cmu_mosei import CMUMOSEIAdapter


@dataclass(frozen=True)
class MELDAdapter(CMUMOSEIAdapter):
    name: str = "meld"

    def discover_raw(self, raw_root: Path) -> RawDatasetManifest:
        return require_files(
            self.name,
            raw_root,
            (
                "features/text_features.npy",
                "features/audio_features.npy",
                "features/visual_features.npy",
                "labels/emotion.npy",
                "metadata/dialogues.json",
                "splits.json",
            ),
        )
