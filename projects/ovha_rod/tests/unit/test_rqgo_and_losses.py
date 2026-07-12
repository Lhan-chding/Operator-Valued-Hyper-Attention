import unittest

import torch

from ovha_rod.models.losses import build_seed_quality_targets, quality_focal_seed_loss
from ovha_rod.models.operators.base import memory_valid_mask
from ovha_rod.models.operators.generic_seed import (
    GenericDenseSeedPredictor, matched_generic_hidden_dim)
from ovha_rod.models.operators.rqgo import RQGO
from ovha_rod.models.role_encoder import LatentRoleEncoder


class RQGOTests(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(13)
        memory = torch.randn(2, 20, 16)
        boxes = torch.rand(2, 20, 4).clamp(0.05, 0.95)
        text = torch.randn(2, 7, 16)
        text_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
        memory_mask = torch.zeros(2, 20, dtype=torch.bool)
        memory_mask[:, -3:] = True
        spatial_shapes = torch.tensor([[4, 4], [2, 2]])
        return memory, boxes, text, text_mask, memory_mask, spatial_shapes

    def test_zero_init_preserves_parent_topk_exactly(self):
        memory, boxes, text, text_mask, memory_mask, spatial_shapes = self._inputs()
        roles = LatentRoleEncoder(16, num_heads=4)(text, text_mask)
        model = RQGO(d_model=16, seed_bias_cap=2.0).eval()
        base_scores = torch.randn(2, 20).masked_fill(memory_mask, float("-inf"))

        result = model(memory, boxes, roles, spatial_shapes, memory_mask)
        parent_topk = base_scores.topk(5, dim=1).indices
        ovha_topk = (base_scores + result.seed_bias).topk(5, dim=1).indices

        self.assertTrue(torch.equal(result.seed_bias, torch.zeros_like(result.seed_bias)))
        self.assertTrue(torch.equal(parent_topk, ovha_topk))
        self.assertLessEqual(float(result.seed_bias.detach().abs().max()), 2.0)
        self.assertTrue(torch.equal(result.valid, ~memory_mask))

    def test_padding_content_cannot_change_valid_seed_logits(self):
        memory, boxes, text, text_mask, memory_mask, spatial_shapes = self._inputs()
        roles = LatentRoleEncoder(16, num_heads=4)(text, text_mask)
        model = RQGO(d_model=16)
        with torch.no_grad():
            model.seed_head[-1].weight.fill_(0.05)
        changed = memory.clone()
        changed[memory_mask] = 999.0

        first = model(memory, boxes, roles, spatial_shapes, memory_mask)
        second = model(changed, boxes, roles, spatial_shapes, memory_mask)

        self.assertTrue(torch.allclose(first.seed_bias.masked_select(first.valid), second.seed_bias.masked_select(second.valid), atol=1e-5))
        self.assertTrue(torch.equal(first.seed_bias.masked_select(~first.valid), torch.zeros_like(first.seed_bias.masked_select(~first.valid))))

    def test_seed_loss_backpropagates_to_rqgo(self):
        memory, boxes, text, text_mask, memory_mask, spatial_shapes = self._inputs()
        roles = LatentRoleEncoder(16, num_heads=4)(text, text_mask)
        model = RQGO(d_model=16)
        parent_score = torch.randn(2, 20)
        gt = torch.tensor([[[0.5, 0.5, 0.4, 0.4]], [[0.3, 0.3, 0.2, 0.2]]])
        gt_mask = torch.ones(2, 1, dtype=torch.bool)
        result = model(memory, boxes, roles, spatial_shapes, memory_mask)
        targets = build_seed_quality_targets(boxes, gt, gt_mask, gamma=1.0)
        loss = quality_focal_seed_loss(parent_score.detach() + result.seed_bias, targets, result.valid)
        loss.backward()
        grad = model.seed_head[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_generic_control_has_same_contract(self):
        memory, boxes, text, text_mask, memory_mask, _ = self._inputs()
        pooled = (text * text_mask[..., None]).sum(1) / text_mask.sum(1, keepdim=True)
        model = GenericDenseSeedPredictor(d_model=16, num_levels=2)
        level_ids = torch.tensor([0] * 16 + [1] * 4)
        result = model(memory, boxes, pooled, level_ids, memory_mask)
        self.assertEqual(tuple(result.seed_bias.shape), (2, 20))
        self.assertTrue(torch.equal(result.valid, ~memory_mask))
        self.assertTrue(torch.equal(result.seed_bias, torch.zeros_like(result.seed_bias)))

    def test_none_memory_mask_means_every_encoder_token_is_valid(self):
        memory, boxes, text, text_mask, _, spatial_shapes = self._inputs()
        roles = LatentRoleEncoder(16, num_heads=4)(text, text_mask)

        valid = memory_valid_mask(memory, None)
        rqgo = RQGO(d_model=16)(memory, boxes, roles, spatial_shapes, None)
        generic = GenericDenseSeedPredictor(
            d_model=16, num_levels=2)(
                memory, boxes, text.mean(dim=1),
                torch.tensor([0] * 16 + [1] * 4), None)

        self.assertTrue(valid.all())
        self.assertTrue(rqgo.valid.all())
        self.assertTrue(generic.valid.all())

    def test_memory_valid_mask_rejects_wrong_shape(self):
        memory, *_ = self._inputs()
        with self.assertRaisesRegex(ValueError, "shape"):
            memory_valid_mask(memory, torch.zeros(2, 19, dtype=torch.bool))

    def test_generic_control_matches_rqgo_and_role_capacity(self):
        hidden = matched_generic_hidden_dim(
            d_model=256, num_levels=4, relation_count=8, scale_count=3)
        typed_modules = (LatentRoleEncoder(256, num_heads=8), RQGO(256))
        generic = GenericDenseSeedPredictor(256, 4, hidden_dim=hidden)
        typed_count = sum(
            parameter.numel() for module in typed_modules
            for parameter in module.parameters())
        generic_count = sum(parameter.numel() for parameter in generic.parameters())
        self.assertLess(abs(generic_count - typed_count) / typed_count, 0.01)

    def test_seed_operators_are_finite_under_autocast_backward(self):
        memory, boxes, text, text_mask, memory_mask, spatial_shapes = self._inputs()
        roles = LatentRoleEncoder(16, num_heads=4)(text, text_mask)
        parent_score = torch.randn(2, 20)
        targets = torch.rand(2, 20)
        operators = (
            (RQGO(d_model=16), (memory, boxes, roles, spatial_shapes, memory_mask)),
            (
                GenericDenseSeedPredictor(d_model=16, num_levels=2),
                (memory, boxes, text.mean(dim=1),
                 torch.tensor([0] * 16 + [1] * 4), memory_mask),
            ),
        )
        for operator, inputs in operators:
            with self.subTest(operator=type(operator).__name__):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    result = operator(*inputs)
                    loss = quality_focal_seed_loss(
                        parent_score.detach() + result.seed_bias,
                        targets,
                        result.valid,
                    )
                loss.backward()
                gradients = [
                    parameter.grad for parameter in operator.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(torch.isfinite(result.seed_bias).all())
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
                self.assertLessEqual(float(result.seed_bias.detach().abs().max()), 2.0)


class SeedLossTests(unittest.TestCase):
    def test_iou_targets_are_bounded_and_stop_gradient(self):
        proposals = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]]], requires_grad=True)
        gt = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]], requires_grad=True)
        target = build_seed_quality_targets(proposals, gt, torch.ones(1, 1, dtype=torch.bool), gamma=2.0)
        self.assertFalse(target.requires_grad)
        self.assertAlmostEqual(float(target[0, 0]), 1.0, places=6)
        self.assertGreaterEqual(float(target.min()), 0.0)
        self.assertLessEqual(float(target.max()), 1.0)

    def test_empty_ground_truth_and_invalid_proposals_are_safe(self):
        proposals = torch.rand(1, 3, 4)
        target = build_seed_quality_targets(proposals, torch.empty(1, 0, 4), torch.empty(1, 0, dtype=torch.bool))
        valid = torch.tensor([[1, 0, 1]], dtype=torch.bool)
        loss = quality_focal_seed_loss(torch.zeros(1, 3), target, valid)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(target.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
