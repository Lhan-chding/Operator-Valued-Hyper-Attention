from dataclasses import dataclass

from moat_ovha_torch.data.multimodal.adapters.cmu_mosei import CMUMOSEIAdapter


@dataclass(frozen=True)
class MELDAdapter(CMUMOSEIAdapter):
    name: str = "meld"
