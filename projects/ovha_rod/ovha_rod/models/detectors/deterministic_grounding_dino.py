from __future__ import annotations

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS

from ..positional_encoding import DeterministicSinePositionalEncoding


class DeterministicPositionalEncodingMixin:
    """Replace only the pinned parent's stateless positional encoder."""

    def _init_layers(self) -> None:
        positional_encoding_cfg = dict(self.positional_encoding)
        super()._init_layers()
        self.positional_encoding = DeterministicSinePositionalEncoding(
            **positional_encoding_cfg)


@MODELS.register_module()
class DeterministicGroundingDINO(
    DeterministicPositionalEncodingMixin,
    GroundingDINO,
):
    """Official GroundingDINO with deterministic padded-mask coordinates."""

