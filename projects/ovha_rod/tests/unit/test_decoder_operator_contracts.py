import unittest

import torch

from ovha_rod.models.operators.decoder_contracts import (
    DecoderOperatorResidual,
    DecoderResidualState,
    StructuredResidualFusion,
)


class DecoderOperatorResidualTests(unittest.TestCase):
    def _residual(self, *, gate_value: float = 0.0):
        return DecoderOperatorResidual(
            query_delta=torch.ones(2, 3, 8),
            box_delta=torch.full((2, 3, 4), 2.0),
            score_delta=torch.full((2, 3), 3.0),
            gate_logits=torch.full((2, 3, 3), gate_value),
            valid=torch.tensor(
                [[True, True, False], [True, False, False]],
                dtype=torch.bool,
            ),
            diagnostics={"mass": torch.tensor(1.0)},
        )

    def test_contract_is_frozen_and_validates_all_axes(self):
        residual = self._residual()
        self.assertEqual(tuple(residual.query_delta.shape), (2, 3, 8))
        with self.assertRaisesRegex(Exception, "cannot assign"):
            residual.valid = torch.ones(2, 3, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "box_delta"):
            DecoderOperatorResidual(
                query_delta=torch.zeros(2, 3, 8),
                box_delta=torch.zeros(2, 4, 4),
                score_delta=torch.zeros(2, 3),
                gate_logits=torch.zeros(2, 3, 3),
                valid=torch.ones(2, 3, dtype=torch.bool),
            )

        with self.assertRaisesRegex(ValueError, "gate_logits"):
            DecoderOperatorResidual(
                query_delta=torch.zeros(2, 3, 8),
                box_delta=torch.zeros(2, 3, 4),
                score_delta=torch.zeros(2, 3),
                gate_logits=torch.zeros(2, 3, 2),
                valid=torch.ones(2, 3, dtype=torch.bool),
            )

    def test_contract_rejects_nonfinite_values_and_non_boolean_mask(self):
        query_delta = torch.zeros(1, 2, 4)
        query_delta[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            DecoderOperatorResidual(
                query_delta=query_delta,
                box_delta=torch.zeros(1, 2, 4),
                score_delta=torch.zeros(1, 2),
                gate_logits=torch.zeros(1, 2, 3),
                valid=torch.ones(1, 2, dtype=torch.bool),
            )

        with self.assertRaisesRegex(ValueError, "boolean"):
            DecoderOperatorResidual(
                query_delta=torch.zeros(1, 2, 4),
                box_delta=torch.zeros(1, 2, 4),
                score_delta=torch.zeros(1, 2),
                gate_logits=torch.zeros(1, 2, 3),
                valid=torch.ones(1, 2),
            )


class StructuredResidualFusionTests(unittest.TestCase):
    def _parent(self):
        torch.manual_seed(23)
        return DecoderResidualState(
            query=torch.randn(2, 3, 8),
            box_logits=torch.randn(2, 3, 4),
            referent_score=torch.randn(2, 3),
        )

    def _residual(self, gate_value: float):
        return DecoderOperatorResidual(
            query_delta=torch.ones(2, 3, 8),
            box_delta=torch.full((2, 3, 4), 2.0),
            score_delta=torch.full((2, 3), 3.0),
            gate_logits=torch.full((2, 3, 3), gate_value),
            valid=torch.tensor(
                [[True, True, False], [True, False, False]],
                dtype=torch.bool,
            ),
        )

    def test_zero_gate_is_exact_parent_noop_without_input_mutation(self):
        parent = self._parent()
        query_before = parent.query.clone()
        boxes_before = parent.box_logits.clone()
        score_before = parent.referent_score.clone()

        fused = StructuredResidualFusion()(parent, (self._residual(0.0),))

        self.assertTrue(torch.equal(fused.query, query_before))
        self.assertTrue(torch.equal(fused.box_logits, boxes_before))
        self.assertTrue(torch.equal(fused.referent_score, score_before))
        self.assertTrue(torch.equal(parent.query, query_before))
        self.assertTrue(torch.equal(parent.box_logits, boxes_before))
        self.assertTrue(torch.equal(parent.referent_score, score_before))

    def test_nonzero_gate_changes_only_valid_queries(self):
        parent = self._parent()
        residual = self._residual(0.5)
        fused = StructuredResidualFusion()(parent, (residual,))
        valid = residual.valid

        self.assertGreater(
            float((fused.query - parent.query)[valid].abs().sum()), 0.0)
        self.assertGreater(
            float((fused.box_logits - parent.box_logits)[valid].abs().sum()),
            0.0,
        )
        self.assertGreater(
            float((fused.referent_score - parent.referent_score)[valid].abs().sum()),
            0.0,
        )
        self.assertTrue(torch.equal(fused.query[~valid], parent.query[~valid]))
        self.assertTrue(
            torch.equal(fused.box_logits[~valid], parent.box_logits[~valid]))
        self.assertTrue(
            torch.equal(
                fused.referent_score[~valid], parent.referent_score[~valid]))

    def test_multiple_operator_residuals_add_and_backward_is_finite(self):
        parent = self._parent()
        first = self._residual(0.25)
        second = self._residual(-0.1)
        first.query_delta.requires_grad_(True)
        first.box_delta.requires_grad_(True)
        first.score_delta.requires_grad_(True)
        first.gate_logits.requires_grad_(True)

        fused = StructuredResidualFusion()(parent, (first, second))
        loss = (
            fused.query.square().mean()
            + fused.box_logits.square().mean()
            + fused.referent_score.square().mean()
        )
        loss.backward()

        gradients = (
            first.query_delta.grad,
            first.box_delta.grad,
            first.score_delta.grad,
            first.gate_logits.grad,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_empty_operator_sequence_returns_new_parent_state(self):
        parent = self._parent()
        fused = StructuredResidualFusion()(parent, ())

        self.assertIsNot(fused, parent)
        self.assertTrue(torch.equal(fused.query, parent.query))
        self.assertTrue(torch.equal(fused.box_logits, parent.box_logits))
        self.assertTrue(torch.equal(fused.referent_score, parent.referent_score))


if __name__ == "__main__":
    unittest.main()
