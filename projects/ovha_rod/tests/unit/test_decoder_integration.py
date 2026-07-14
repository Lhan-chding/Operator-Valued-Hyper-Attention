import unittest

import torch

from ovha_rod.models.operators.decoder_contracts import DecoderOperatorResidual
from ovha_rod.models.operators.decoder_contracts import DecoderResidualState
from ovha_rod.models.operators.decoder_contracts import StructuredResidualFusion


def _integration_api():
    from ovha_rod.models.operators.decoder_integration import (
        apply_matching_query_residual,
        integrate_matching_query_state,
        stack_decoder_operator_outputs,
    )

    return (
        apply_matching_query_residual,
        integrate_matching_query_state,
        stack_decoder_operator_outputs,
    )


class DecoderIntegrationTests(unittest.TestCase):
    def _inputs(self, *, matching_queries: int = 3, dn_queries: int = 2):
        torch.manual_seed(97)
        batch_size, hidden_dim = 2, 8
        total_queries = matching_queries + dn_queries
        query = torch.randn(
            batch_size, total_queries, hidden_dim, requires_grad=True)
        box_logits = torch.randn(
            batch_size, total_queries, 4, requires_grad=True)
        query_delta = torch.ones(
            batch_size, matching_queries, hidden_dim, requires_grad=True)
        box_delta = torch.full(
            (batch_size, matching_queries, 4), 0.5, requires_grad=True)
        score_delta = torch.full(
            (batch_size, matching_queries), 0.25, requires_grad=True)
        gate_logits = torch.full(
            (batch_size, matching_queries, 3), 0.4, requires_grad=True)
        valid = torch.tensor(
            [[True, False, True], [True, True, False]], dtype=torch.bool)
        residual = DecoderOperatorResidual(
            query_delta=query_delta,
            box_delta=box_delta,
            score_delta=score_delta,
            gate_logits=gate_logits,
            valid=valid,
        )
        return query, box_logits, residual

    def test_training_dn_prefix_is_exact_and_invalid_tail_is_masked(self):
        apply_residual, _, _ = _integration_api()
        query, box_logits, residual = self._inputs()
        result = apply_residual(
            query=query,
            parent_box_logits=box_logits,
            residual=residual,
            matching_query_count=3,
        )

        self.assertTrue(torch.equal(result.query[:, :2], query[:, :2]))
        self.assertTrue(torch.equal(
            result.reference_points[:, :2], box_logits[:, :2].sigmoid()))
        invalid = ~residual.valid
        self.assertTrue(torch.equal(
            result.query[:, -3:][invalid], query[:, -3:][invalid]))
        self.assertTrue(torch.equal(
            result.reference_points[:, -3:][invalid],
            box_logits[:, -3:].sigmoid()[invalid],
        ))
        self.assertTrue(torch.equal(
            result.operator_box_delta[invalid],
            torch.zeros_like(result.operator_box_delta[invalid]),
        ))
        self.assertTrue(torch.equal(
            result.operator_referent_score[invalid],
            torch.zeros_like(result.operator_referent_score[invalid]),
        ))

    def test_eval_empty_dn_and_reference_detach_contract(self):
        apply_residual, _, _ = _integration_api()
        query, box_logits, residual = self._inputs(dn_queries=0)
        result = apply_residual(
            query=query,
            parent_box_logits=box_logits,
            residual=residual,
            matching_query_count=3,
        )

        self.assertEqual(tuple(result.query.shape), (2, 3, 8))
        self.assertEqual(tuple(result.reference_points.shape), (2, 3, 4))
        self.assertTrue(result.reference_points.requires_grad)
        self.assertFalse(result.next_reference_points.requires_grad)
        self.assertIsNone(result.next_reference_points.grad_fn)
        self.assertTrue(torch.equal(
            result.next_reference_points, result.reference_points.detach()))

    def test_final_layer_all_three_residual_channels_receive_gradients(self):
        apply_residual, _, _ = _integration_api()
        query, box_logits, residual = self._inputs(dn_queries=0)
        result = apply_residual(
            query=query,
            parent_box_logits=box_logits,
            residual=residual,
            matching_query_count=3,
        )
        loss = (
            result.query.square().mean()
            + result.reference_points.square().mean()
            + result.operator_referent_score.square().mean()
        )
        loss.backward()

        channels = (
            residual.query_delta.grad,
            residual.box_delta.grad,
            residual.score_delta.grad,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(gradient is not None for gradient in channels))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in channels))
        self.assertTrue(all(float(gradient.abs().sum()) > 0 for gradient in channels))
        self.assertIsNotNone(residual.gate_logits.grad)
        self.assertGreater(float(residual.gate_logits.grad.abs().sum()), 0.0)

    def test_shape_mask_and_matching_count_errors_fail_fast(self):
        apply_residual, _, _ = _integration_api()
        query, box_logits, residual = self._inputs()
        cases = (
            ({"query": query[:, :, :7]}, "query_delta"),
            ({"parent_box_logits": box_logits[:, :4]}, "parent_box_logits"),
            ({"matching_query_count": 4}, "matching_query_count"),
        )
        for changes, message in cases:
            arguments = dict(
                query=query,
                parent_box_logits=box_logits,
                residual=residual,
                matching_query_count=3,
            )
            arguments.update(changes)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    apply_residual(**arguments)

        with self.assertRaisesRegex(ValueError, "boolean"):
            DecoderOperatorResidual(
                query_delta=torch.zeros(2, 3, 8),
                box_delta=torch.zeros(2, 3, 4),
                score_delta=torch.zeros(2, 3),
                gate_logits=torch.zeros(2, 3, 3),
                valid=torch.ones(2, 3),
            )

    def test_optional_operator_stacks_are_none_or_layer_major(self):
        apply_residual, _, stack_outputs = _integration_api()
        self.assertEqual(stack_outputs(()), (None, None))

        query, box_logits, residual = self._inputs(dn_queries=0)
        first = apply_residual(
            query=query,
            parent_box_logits=box_logits,
            residual=residual,
            matching_query_count=3,
        )
        second = apply_residual(
            query=first.query,
            parent_box_logits=box_logits + 0.1,
            residual=residual,
            matching_query_count=3,
        )
        box_deltas, referent_scores = stack_outputs((first, second))

        self.assertEqual(tuple(box_deltas.shape), (2, 2, 3, 4))
        self.assertEqual(tuple(referent_scores.shape), (2, 2, 3))
        self.assertTrue(torch.equal(box_deltas[0], first.operator_box_delta))
        self.assertTrue(torch.equal(
            referent_scores[1], second.operator_referent_score))

    def test_pre_fused_bank_state_is_spliced_without_second_fusion(self):
        _, integrate_state, _ = _integration_api()
        query, box_logits, residual = self._inputs()
        matching_query = query[:, -3:, :]
        matching_box_logits = box_logits[:, -3:, :]
        fused_state = DecoderResidualState(
            query=matching_query + residual.query_delta,
            box_logits=matching_box_logits + residual.box_delta,
            referent_score=residual.score_delta,
        )

        result = integrate_state(
            query=query,
            parent_box_logits=box_logits,
            fused_matching_state=fused_state,
            matching_query_count=3,
        )

        self.assertTrue(torch.equal(result.query[:, :2], query[:, :2]))
        self.assertTrue(torch.equal(
            result.query[:, -3:], fused_state.query))
        torch.testing.assert_close(
            result.operator_box_delta, residual.box_delta)
        self.assertTrue(torch.equal(
            result.operator_referent_score, residual.score_delta))

    def test_rebased_reference_matches_fused_query_head_prediction(self):
        _, integrate_state, _ = _integration_api()
        query, box_logits, residual = self._inputs()
        matching_query = query[:, -3:, :] + residual.query_delta
        matching_parent_logits = box_logits[:, -3:, :]
        fused_state = DecoderResidualState(
            query=matching_query,
            box_logits=matching_parent_logits + residual.box_delta,
            referent_score=residual.score_delta,
        )
        fused_query_parent_box_logits = box_logits + 0.37
        fused_query_parent_box_logits = torch.cat(
            (box_logits[:, :2], fused_query_parent_box_logits[:, 2:]), dim=1)

        result = integrate_state(
            query=query,
            parent_box_logits=box_logits,
            fused_matching_state=fused_state,
            matching_query_count=3,
            fused_query_parent_box_logits=fused_query_parent_box_logits,
        )

        expected_logits = torch.cat(
            (
                box_logits[:, :2],
                fused_query_parent_box_logits[:, -3:] + residual.box_delta,
            ),
            dim=1,
        )
        torch.testing.assert_close(
            result.reference_points, expected_logits.sigmoid())
        torch.testing.assert_close(
            result.operator_box_delta, residual.box_delta)

    def test_recurrent_referent_score_reaches_every_layer_score_gate(self):
        batch_size, query_count, hidden_dim = 2, 3, 8
        query = torch.zeros(batch_size, query_count, hidden_dim)
        box_logits = torch.zeros(batch_size, query_count, 4)
        valid = torch.ones(batch_size, query_count, dtype=torch.bool)
        score = torch.zeros(batch_size, query_count)
        score_gates = [
            torch.full(
                (batch_size, query_count, 1),
                0.2 + layer_index * 0.1,
                requires_grad=True,
            )
            for layer_index in range(3)
        ]

        for score_gate in score_gates:
            parent = DecoderResidualState(query, box_logits, score)
            residual = DecoderOperatorResidual(
                query_delta=torch.zeros_like(query),
                box_delta=torch.zeros_like(box_logits),
                score_delta=torch.ones_like(score),
                gate_logits=torch.cat(
                    (torch.zeros_like(score_gate),
                     torch.zeros_like(score_gate), score_gate),
                    dim=-1,
                ),
                valid=valid,
            )
            score = StructuredResidualFusion()(parent, (residual,)).referent_score

        score.sum().backward()
        for layer_index, score_gate in enumerate(score_gates):
            with self.subTest(layer=layer_index):
                self.assertIsNotNone(score_gate.grad)
                self.assertTrue(torch.isfinite(score_gate.grad).all())
                self.assertGreater(float(score_gate.grad.abs().sum()), 0.0)

    def test_bank_diagnostics_are_detached_tensor_scalars(self):
        from ovha_rod.models.operators.decoder_bank import (
            DecoderOperatorBank,
            DecoderOperatorContext,
        )
        from ovha_rod.models.operators.decoder_integration import (
            summarize_decoder_bank_outputs,
        )

        torch.manual_seed(211)
        batch_size, query_count, hidden_dim = 2, 3, 8
        query = torch.randn(batch_size, query_count, hidden_dim)
        boxes = torch.rand(batch_size, query_count, 4)
        boxes[..., 2:] = boxes[..., 2:] * 0.8 + 0.1
        valid = torch.ones(batch_size, query_count, dtype=torch.bool)
        relation_role = torch.randn(batch_size, hidden_dim)
        bank = DecoderOperatorBank(
            d_model=hidden_dim,
            num_layers=2,
            router_hidden_dim=12,
            adapter_rank=4,
            enabled_operators=("qsro",),
        )
        parent = DecoderResidualState(
            query=query,
            box_logits=torch.logit(boxes),
            referent_score=torch.zeros(batch_size, query_count),
        )
        outputs = []
        memory_state = None
        for layer_index in range(2):
            output = bank(DecoderOperatorContext(
                parent=parent,
                boxes=parent.box_logits.sigmoid(),
                valid=valid,
                layer_index=layer_index,
                relation_role=relation_role,
                memory_state=memory_state,
            ))
            outputs.append(output)
            parent = output.fused
            memory_state = output.memory_state

        diagnostics = summarize_decoder_bank_outputs(outputs)
        required = {
            "decoder_memory_norm",
            "decoder_qsro_router_weight_mean",
            "decoder_qsro_gate_sigmoid_mean",
            "decoder_qsro_gate_abs_mean",
            "decoder_qsro_availability_fraction",
            "decoder_qsro_reliability_mean",
            "decoder_tq_cato_router_weight_mean",
            "decoder_ms_tleo_router_weight_mean",
        }
        self.assertTrue(required.issubset(diagnostics))
        for name, value in diagnostics.items():
            with self.subTest(name=name):
                self.assertIsInstance(value, torch.Tensor)
                self.assertEqual(value.numel(), 1)
                self.assertFalse(value.requires_grad)
                self.assertTrue(torch.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
