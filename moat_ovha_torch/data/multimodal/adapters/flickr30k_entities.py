from dataclasses import dataclass
from pathlib import Path

from moat_ovha_torch.data.multimodal.adapters.base import RawDatasetManifest, require_files
from moat_ovha_torch.data.multimodal.adapters.refcoco import RefCOCOAdapter


@dataclass(frozen=True)
class Flickr30kEntitiesAdapter(RefCOCOAdapter):
    name: str = "flickr30k_entities"

    def discover_raw(self, raw_root: Path) -> RawDatasetManifest:
        return require_files(
            self.name,
            raw_root,
            (
                "annotations/phrase_regions.json",
                "annotations/captions.json",
                "features/text_features.npy",
                "features/region_features.npy",
                "splits.json",
            ),
        )
