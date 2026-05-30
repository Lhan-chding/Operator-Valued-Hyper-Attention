from dataclasses import dataclass

from moat_ovha_torch.data.multimodal.adapters.refcoco import RefCOCOAdapter


@dataclass(frozen=True)
class Flickr30kEntitiesAdapter(RefCOCOAdapter):
    name: str = "flickr30k_entities"
