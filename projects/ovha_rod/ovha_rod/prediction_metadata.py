from __future__ import annotations

from typing import Any, TypeVar


ENCODER_QUERY_BOXES = "encoder_query_boxes"
PredictionT = TypeVar("PredictionT")


def with_encoder_query_boxes(
    prediction: PredictionT, boxes: Any,
) -> PredictionT:
    """Return a cloned prediction carrying variable-length oracle boxes."""
    updated = prediction.clone()
    updated.set_metainfo({ENCODER_QUERY_BOXES: boxes})
    return updated


def get_encoder_query_boxes(prediction: Any) -> Any | None:
    """Read optional encoder-query boxes without treating them as instances."""
    return prediction.metainfo.get(ENCODER_QUERY_BOXES)
