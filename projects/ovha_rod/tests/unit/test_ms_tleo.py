from pathlib import Path
import unittest

import torch

from ovha_rod.models import operators as operator_package
from ovha_rod.models.operators.decoder_contracts import (
    DecoderResidualState,
    StructuredResidualFusion,
)
from ovha_rod.models.operators.ms_tleo import (
    MSTLEO,
    normalized_image_to_grid,
)


class MSTLEOTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)
        self.operator = MSTLEO(d_model=4, context_scale=1.5)

    @staticmethod
    def _features(*, requires_grad: bool = False):
        return (
            torch.randn(2, 4, 8, 8, requires_grad=requires_grad),
            torch.randn(2, 4, 4, 4, requires_grad=requires_grad),
        )

    @staticmethod
    def _boxes(*, requires_grad: bool = False):
        boxes = torch.tensor(
            [
                [[0.50, 0.50, 0.30, 0.20], [0.25, 0.75, 0.20, 0.25]],
                [[0.70, 0.30, 0.15, 0.35], [0.40, 0.40, 0.25, 0.25]],
            ]
        )
        return boxes.requires_grad_(requires_grad)

    def test_align_corners_false_coordinate_convention_is_explicit(self):
        coordinates = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(
            torch.equal(
                normalized_image_to_grid(coordinates),
                torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]),
            )
        )

        pixel_centres = (torch.arange(4, dtype=torch.float32) + 0.5) / 4
        x_coordinates = pixel_centres.view(1, 1, 1, 4).expand(1, 1, 4, 4)
        boxes = torch.tensor([[[0.375, 0.500, 0.200, 0.200]]])
        valid = torch.ones(1, 1, dtype=torch.bool)

        evidence = MSTLEO(d_model=1, context_scale=1.5).extract_evidence(
            (x_coordinates,), boxes, valid)

        expected = torch.tensor([[[0.375]]])
        self.assertTrue(torch.allclose(evidence.interior, expected, atol=1e-6))
        self.assertTrue(torch.allclose(evidence.boundary, expected, atol=1e-6))
        self.assertTrue(torch.allclose(evidence.context, expected, atol=1e-6))

    def test_multiscale_layouts_are_resolution_consistent_and_mean_aggregated(self):
        def affine_map(size: int, offset: float) -> torch.Tensor:
            centres = (torch.arange(size, dtype=torch.float32) + 0.5) / size
            return (centres + offset).view(1, 1, 1, size).expand(
                1, 1, size, size)

        boxes = torch.tensor([[[0.375, 0.500, 0.200, 0.200]]])
        valid = torch.ones(1, 1, dtype=torch.bool)

        evidence = MSTLEO(d_model=1, context_scale=1.5).extract_evidence(
            (affine_map(4, 0.0), affine_map(8, 2.0)), boxes, valid)

        expected = torch.tensor([[[1.375]]])
        self.assertTrue(torch.allclose(evidence.interior, expected, atol=1e-6))
        self.assertTrue(torch.allclose(evidence.boundary, expected, atol=1e-6))
        self.assertTrue(torch.allclose(evidence.context, expected, atol=1e-6))

    def test_per_level_valid_ratios_scale_padding_coordinates(self):
        def affine_xy_map(height: int, width: int) -> torch.Tensor:
            y_centres = (torch.arange(height, dtype=torch.float32) + 0.5) / height
            x_centres = (torch.arange(width, dtype=torch.float32) + 0.5) / width
            y_coordinate, x_coordinate = torch.meshgrid(
                y_centres, x_centres, indexing="ij")
            return (x_coordinate + 10.0 * y_coordinate).view(
                1, 1, height, width).expand(2, 1, height, width)

        feature_maps = (affine_xy_map(16, 16), affine_xy_map(8, 8))
        boxes = torch.tensor(
            [
                [[0.5, 0.5, 0.2, 0.2]],
                [[0.5, 0.5, 0.2, 0.2]],
            ]
        )
        valid = torch.ones(2, 1, dtype=torch.bool)
        # Ratios are ordered as (valid width, valid height). For sample 0,
        # the two level centres become (0.25, 0.375) and (0.5, 0.25), so
        # the equally averaged affine evidence is (4.0 + 3.0) / 2 = 3.5.
        valid_ratios = torch.tensor(
            [
                [[0.50, 0.75], [1.00, 0.50]],
                [[0.25, 1.00], [0.75, 0.25]],
            ]
        )

        evidence = MSTLEO(d_model=1).extract_evidence(
            feature_maps,
            boxes,
            valid,
            valid_ratios=valid_ratios,
        )

        expected = torch.tensor([[[3.500]], [[3.375]]])
        self.assertTrue(torch.allclose(evidence.interior, expected, atol=1e-5))
        self.assertTrue(torch.allclose(evidence.boundary, expected, atol=1e-5))
        self.assertTrue(torch.allclose(evidence.context, expected, atol=1e-5))

    def test_none_and_all_one_valid_ratios_preserve_legacy_behavior(self):
        feature_maps = self._features()
        boxes = self._boxes()
        valid = torch.tensor([[True, True], [True, False]])

        legacy = self.operator(feature_maps, boxes, valid)
        explicit_none = self.operator(
            feature_maps, boxes, valid, valid_ratios=None)
        all_one = self.operator(
            feature_maps,
            boxes,
            valid,
            valid_ratios=torch.ones(2, len(feature_maps), 2),
        )

        for field in (
            "query_delta",
            "box_delta",
            "score_delta",
            "gate_logits",
            "valid",
        ):
            self.assertTrue(torch.equal(getattr(legacy, field),
                                        getattr(explicit_none, field)))
            self.assertTrue(torch.equal(getattr(legacy, field),
                                        getattr(all_one, field)))
        for name in legacy.diagnostics:
            self.assertTrue(torch.equal(legacy.diagnostics[name],
                                        all_one.diagnostics[name]))

    def test_valid_ratios_shape_dtype_device_finite_and_range_fail_fast(self):
        feature_maps = self._features()
        boxes = self._boxes()
        valid = torch.ones(2, 2, dtype=torch.bool)

        invalid_cases = (
            (torch.ones(2, 2), "shape.*B,L,2"),
            (torch.ones(2, 3, 2), "shape.*B,L,2"),
            (torch.ones(2, 2, 2, dtype=torch.int64), "floating point"),
            (torch.ones(2, 2, 2, dtype=torch.float64), "same dtype"),
            (torch.full((2, 2, 2), float("nan")), "finite"),
            (torch.zeros(2, 2, 2), "greater than zero"),
            (torch.full((2, 2, 2), 1.01), "at most one"),
            (torch.ones(2, 2, 2, device="meta"), "same device"),
        )
        for ratios, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.operator(
                        feature_maps,
                        boxes,
                        valid,
                        valid_ratios=ratios,
                    )

    def test_context_ring_expands_beyond_boundary_and_interior_layouts(self):
        size = 64
        centres = (torch.arange(size, dtype=torch.float32) + 0.5) / size
        y_coordinate, x_coordinate = torch.meshgrid(
            centres, centres, indexing="ij")
        radius_squared = (
            (x_coordinate - 0.5).square() + (y_coordinate - 0.5).square()
        ).view(1, 1, size, size)
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])
        valid = torch.ones(1, 1, dtype=torch.bool)

        evidence = MSTLEO(d_model=1).extract_evidence(
            (radius_squared,), boxes, valid)

        self.assertLess(float(evidence.interior), float(evidence.boundary))
        self.assertLess(float(evidence.boundary), float(evidence.context))

    def test_masking_is_strict_and_zero_gate_is_exact_fusion_noop(self):
        valid = torch.tensor([[True, False], [False, True]])
        boxes = self._boxes()
        boxes = torch.where(valid[..., None], boxes, torch.zeros_like(boxes))

        residual = self.operator(self._features(), boxes, valid)

        self.assertTrue(torch.equal(residual.query_delta[~valid], torch.zeros(2, 4)))
        self.assertTrue(torch.equal(residual.box_delta[~valid], torch.zeros(2, 4)))
        self.assertTrue(torch.equal(residual.score_delta[~valid], torch.zeros(2)))
        self.assertTrue(torch.equal(residual.gate_logits, torch.zeros(2, 2, 3)))

        parent = DecoderResidualState(
            query=torch.randn(2, 2, 4),
            box_logits=torch.randn(2, 2, 4),
            referent_score=torch.randn(2, 2),
        )
        fused = StructuredResidualFusion()(parent, (residual,))
        self.assertTrue(torch.equal(fused.query, parent.query))
        self.assertTrue(torch.equal(fused.box_logits, parent.box_logits))
        self.assertTrue(torch.equal(fused.referent_score, parent.referent_score))

    def test_zero_padding_and_degenerate_box_policies_are_explicit(self):
        feature = torch.ones(1, 4, 4, 4)
        valid = torch.ones(1, 1, dtype=torch.bool)
        outside = torch.tensor([[[2.0, 2.0, 0.1, 0.1]]])

        evidence = self.operator.extract_evidence((feature,), outside, valid)

        self.assertTrue(torch.equal(evidence.interior, torch.zeros(1, 1, 4)))
        self.assertTrue(torch.equal(evidence.boundary, torch.zeros(1, 1, 4)))
        self.assertTrue(torch.equal(evidence.context, torch.zeros(1, 1, 4)))

        with self.assertRaisesRegex(ValueError, "positive width and height"):
            self.operator.extract_evidence(
                (feature,), torch.tensor([[[0.5, 0.5, 0.0, 0.1]]]), valid)

        invalid = torch.zeros(1, 1, dtype=torch.bool)
        evidence = self.operator.extract_evidence(
            (feature,), torch.zeros(1, 1, 4), invalid)
        self.assertTrue(torch.equal(evidence.interior, torch.zeros(1, 1, 4)))

    def test_nonfinite_and_shape_or_dtype_errors_fail_fast(self):
        features = list(self._features())
        features[0] = features[0].clone()
        features[0][0, 0, 0, 0] = float("nan")
        valid = torch.ones(2, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "feature_maps.*finite"):
            self.operator(tuple(features), self._boxes(), valid)

        boxes = self._boxes()
        boxes[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "boxes.*finite"):
            self.operator(self._features(), boxes, valid)

        with self.assertRaisesRegex(ValueError, "non-empty tuple"):
            self.operator((), self._boxes(), valid)
        with self.assertRaisesRegex(ValueError, "same channel dimension"):
            self.operator(
                (torch.zeros(2, 4, 4, 4), torch.zeros(2, 3, 2, 2)),
                self._boxes(),
                valid,
            )
        with self.assertRaisesRegex(ValueError, "positive channel and spatial"):
            self.operator(
                (torch.zeros(2, 4, 0, 4),), self._boxes(), valid)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.operator(self._features(), self._boxes(), valid.float())

    def test_forward_and_box_sampling_backward_are_finite(self):
        features = self._features(requires_grad=True)
        boxes = self._boxes(requires_grad=True)
        valid = torch.tensor([[True, True], [True, False]])

        residual = self.operator(features, boxes, valid)
        direct_loss = (
            residual.query_delta.square().mean()
            + residual.box_delta.square().mean()
            + residual.score_delta.square().mean()
        )
        parent = DecoderResidualState(
            query=torch.randn(2, 2, 4),
            box_logits=torch.randn(2, 2, 4),
            referent_score=torch.randn(2, 2),
        )
        fused = StructuredResidualFusion()(parent, (residual,))
        fusion_loss = fused.query.square().mean()
        loss = direct_loss + fusion_loss
        loss.backward()

        gradients = tuple(feature.grad for feature in features) + (
            boxes.grad,
            self.operator.query_head.weight.grad,
            self.operator.box_head.weight.grad,
            self.operator.score_head.weight.grad,
            self.operator.gate_head.weight.grad,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(float(boxes.grad[valid].abs().sum()), 0.0)
        self.assertGreater(float(self.operator.gate_head.weight.grad.abs().sum()), 0.0)

    def test_constructor_and_formal_config_default_off_contracts(self):
        with self.assertRaisesRegex(ValueError, "d_model"):
            MSTLEO(d_model=0)
        with self.assertRaisesRegex(ValueError, "context_scale"):
            MSTLEO(d_model=4, context_scale=1.0)

        config_root = Path(__file__).parents[2] / "configs"
        formal_configs = tuple(config_root.glob("ovha_rod_swin_t_5e_*.py"))
        self.assertGreater(len(formal_configs), 0)
        for config in formal_configs:
            text = config.read_text(encoding="utf-8")
            self.assertNotIn("MSTLEO", text)
            self.assertNotIn("ms_tleo", text)
        self.assertFalse(hasattr(operator_package, "MSTLEO"))


if __name__ == "__main__":
    unittest.main()
