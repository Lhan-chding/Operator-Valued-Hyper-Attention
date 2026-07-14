import ast
import unittest
from pathlib import Path

import torch

from ovha_rod.evaluation.metrics import query_oracle_metrics, refexp_box_metrics


ROOT = Path(__file__).resolve().parents[2]


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
        self._metainfo = {**self._metainfo, **values}

    def clone(self):
        cloned = _StrictPrediction(self._length)
        cloned._metainfo = dict(self._metainfo)
        return cloned


class Phase1MetricTests(unittest.TestCase):
    def test_refexp_metrics_match_hand_computation(self):
        prediction = torch.tensor([
            [0.5, 0.5, 0.4, 0.4],
            [0.1, 0.1, 0.1, 0.1],
        ])
        ground_truth = torch.tensor([
            [[0.5, 0.5, 0.4, 0.4]],
            [[0.5, 0.5, 0.2, 0.2]],
        ])
        valid = torch.ones(2, 1, dtype=torch.bool)
        metrics = refexp_box_metrics(prediction, ground_truth, valid)
        self.assertAlmostEqual(metrics["acc_0.5"], 0.5)
        self.assertAlmostEqual(metrics["acc_0.75"], 0.5)
        self.assertAlmostEqual(metrics["miou"], 0.5)

    def test_oracle_uses_best_selected_query_not_top1(self):
        selected = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.4, 0.4]]])
        ground_truth = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        metrics = query_oracle_metrics(selected, ground_truth, valid)
        self.assertAlmostEqual(metrics["oracle_0.5"], 1.0)
        self.assertAlmostEqual(metrics["oracle_0.75"], 1.0)
        self.assertAlmostEqual(metrics["max_iou"], 1.0)


class ValidationPredictionMetadataTests(unittest.TestCase):
    def test_encoder_queries_can_outnumber_final_predictions(self):
        from ovha_rod.prediction_metadata import (
            get_encoder_query_boxes,
            with_encoder_query_boxes,
        )

        prediction = _StrictPrediction(length=300)
        encoder_boxes = torch.zeros(900, 4)

        updated = with_encoder_query_boxes(prediction, encoder_boxes)

        self.assertIsNot(updated, prediction)
        self.assertIsNone(get_encoder_query_boxes(prediction))
        self.assertIs(get_encoder_query_boxes(updated), encoder_boxes)
        self.assertNotIn("encoder_query_boxes", prediction.__dict__)

    def test_missing_encoder_query_metadata_returns_none(self):
        from ovha_rod.prediction_metadata import get_encoder_query_boxes

        self.assertIsNone(get_encoder_query_boxes(_StrictPrediction(length=300)))

    def test_validation_paths_share_the_metadata_helpers(self):
        head = (
            ROOT / "ovha_rod/models/dense_heads/ovha_grounding_dino_head.py"
        ).read_text()
        metric = (
            ROOT / "ovha_rod/evaluation/ovha_refexp_metric.py"
        ).read_text()

        self.assertIn("with_encoder_query_boxes(prediction, absolute)", head)
        self.assertNotIn("prediction.encoder_query_boxes = absolute", head)
        self.assertIn("get_encoder_query_boxes(prediction)", metric)


class HookContractTests(unittest.TestCase):
    def test_warmup_and_diagnostics_hooks_exist_and_compile(self):
        for relative in (
            "ovha_rod/hooks/seed_loss_warmup_hook.py",
            "ovha_rod/hooks/operator_diagnostics_hook.py",
            "ovha_rod/hooks/checkpoint_provenance_hook.py",
        ):
            path = ROOT / relative
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)
                ast.parse(path.read_text())

        diagnostics = (
            ROOT / "ovha_rod/hooks/operator_diagnostics_hook.py").read_text()
        self.assertIn("collect_scalar_diagnostics", diagnostics)
        self.assertIn('row["nonfinite_scalar_count"]', diagnostics)
        self.assertIn("allow_nan=False", diagnostics)

    def test_mmdetection_metric_is_registered_in_all_configs(self):
        metric_path = ROOT / "ovha_rod/evaluation/ovha_refexp_metric.py"
        self.assertTrue(metric_path.exists(), metric_path)
        ast.parse(metric_path.read_text())
        for config in sorted((ROOT / "configs").glob("ovha_rod_*.py")):
            with self.subTest(config=config):
                self.assertIn("type='OVHARefExpMetric'", config.read_text())


if __name__ == "__main__":
    unittest.main()
