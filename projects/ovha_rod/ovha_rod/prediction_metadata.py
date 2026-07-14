from __future__ import annotations

from collections.abc import Mapping, Sized
from typing import Any, Protocol, TypeVar


ENCODER_QUERY_BOXES = "encoder_query_boxes"
PredictionT = TypeVar("PredictionT", bound="PredictionMetadataCarrier")


class PredictionMetadataCarrier(Protocol):
    @property
    def metainfo(self) -> Mapping[str, Any]: ...

    def clone(self: PredictionT) -> PredictionT: ...

    def set_metainfo(self, values: dict[str, Any]) -> None: ...


def with_encoder_query_boxes(
    prediction: PredictionT, boxes: Any,
) -> PredictionT:
    """Return a cloned prediction carrying variable-length oracle boxes."""
    updated = prediction.clone()
    updated.set_metainfo({ENCODER_QUERY_BOXES: boxes})
    return updated


def get_encoder_query_boxes(
    prediction: PredictionMetadataCarrier,
) -> Any | None:
    """Read oracle boxes; consumers must move tensor metainfo explicitly."""
    return prediction.metainfo.get(ENCODER_QUERY_BOXES)


def require_prediction_batch_alignment(
    *, predictions: Sized, encoder_boxes: Sized, image_metas: Sized,
) -> None:
    """Reject silent truncation when validation batch dimensions diverge."""
    counts = (len(predictions), len(encoder_boxes), len(image_metas))
    if len(set(counts)) != 1:
        raise ValueError(
            "validation batch lengths must match: "
            f"predictions={counts[0]}, encoder_boxes={counts[1]}, "
            f"image_metas={counts[2]}")
