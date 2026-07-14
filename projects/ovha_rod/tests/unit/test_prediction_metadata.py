import copy
import unittest

from ovha_rod.prediction_metadata import (
    get_encoder_query_boxes,
    require_prediction_batch_alignment,
    with_encoder_query_boxes,
)


class _StrictPrediction:
    """Minimal InstanceData-like object with length-checked data fields."""

    def __init__(self, length: int) -> None:
        self._length = length
        self._metainfo = {}

    def __setattr__(self, name, value) -> None:
        if (
            not name.startswith("_")
            and hasattr(value, "__len__")
            and len(value) != self._length
        ):
            raise AssertionError(
                f"field length {len(value)} does not match {self._length}")
        object.__setattr__(self, name, value)

    @property
    def metainfo(self) -> dict:
        return dict(self._metainfo)

    def set_metainfo(self, values: dict) -> None:
        self._metainfo = {**self._metainfo, **copy.deepcopy(values)}

    def clone(self):
        cloned = _StrictPrediction(self._length)
        cloned._metainfo = copy.deepcopy(self._metainfo)
        for name, value in self.__dict__.items():
            if not name.startswith("_"):
                object.__setattr__(cloned, name, copy.deepcopy(value))
        return cloned


class PredictionMetadataTests(unittest.TestCase):
    def test_encoder_queries_can_outnumber_final_predictions(self):
        prediction = _StrictPrediction(length=300)
        prediction.bboxes = [(0.0, 0.0, 1.0, 1.0)] * 300
        prediction.scores = [0.5] * 300
        prediction.labels = [0] * 300
        encoder_boxes = [(0.0, 0.0, 1.0, 1.0)] * 900

        updated = with_encoder_query_boxes(prediction, encoder_boxes)

        self.assertIsNot(updated, prediction)
        self.assertIsNone(get_encoder_query_boxes(prediction))
        self.assertEqual(get_encoder_query_boxes(updated), encoder_boxes)
        self.assertIsNot(get_encoder_query_boxes(updated), encoder_boxes)
        self.assertEqual(updated.bboxes, prediction.bboxes)
        self.assertEqual(updated.scores, prediction.scores)
        self.assertEqual(updated.labels, prediction.labels)
        self.assertIsNot(updated.bboxes, prediction.bboxes)
        self.assertNotIn("encoder_query_boxes", prediction.__dict__)

    def test_missing_encoder_query_metadata_returns_none(self):
        prediction = _StrictPrediction(length=300)

        self.assertIsNone(get_encoder_query_boxes(prediction))

    def test_mismatched_validation_batch_lengths_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "validation batch lengths"):
            require_prediction_batch_alignment(
                predictions=[object(), object()],
                encoder_boxes=[object()],
                image_metas=[{}, {}],
            )

        require_prediction_batch_alignment(
            predictions=[object(), object()],
            encoder_boxes=[object(), object()],
            image_metas=[{}, {}],
        )


if __name__ == "__main__":
    unittest.main()
