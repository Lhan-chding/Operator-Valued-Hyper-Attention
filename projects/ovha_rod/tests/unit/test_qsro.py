import inspect
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

import ovha_rod.models.operators as operator_exports
from ovha_rod.models.operators import qsro as qsro_module
from ovha_rod.models.operators.decoder_contracts import (
    DecoderResidualState,
    StructuredResidualFusion,
)
from ovha_rod.models.operators.qsro import QuerySpatialRelationOperator


class QuerySpatialRelationOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.d_model = 16
        self.operator = QuerySpatialRelationOperator(d_model=self.d_model)
        self.query = torch.randn(2, 4, self.d_model)
        self.boxes = torch.tensor(
            [
                [
                    [0.20, 0.25, 0.18, 0.20],
                    [0.70, 0.28, 0.16, 0.22],
                    [0.35, 0.72, 0.25, 0.18],
                    [0.82, 0.78, 0.12, 0.14],
                ],
                [
                    [0.15, 0.20, 0.10, 0.12],
                    [0.48, 0.45, 0.30, 0.24],
                    [0.75, 0.65, 0.14, 0.20],
                    [0.55, 0.85, 0.20, 0.10],
                ],
            ],
            dtype=self.query.dtype,
        )
        self.relation_role = torch.randn(2, self.d_model)
        self.valid = torch.tensor(
            [[True, True, True, False], [True, True, False, False]],
            dtype=torch.bool,
        )

    def _forward(self):
        return self.operator(
            self.query, self.boxes, self.relation_role, self.valid)

    def test_outputs_structured_residual_with_scalar_finite_diagnostics(self):
        result = self._forward()

        self.assertEqual(tuple(result.query_delta.shape), (2, 4, 16))
        self.assertEqual(tuple(result.box_delta.shape), (2, 4, 4))
        self.assertEqual(tuple(result.score_delta.shape), (2, 4))
        self.assertEqual(tuple(result.gate_logits.shape), (2, 4, 3))
        self.assertTrue(torch.equal(result.valid, self.valid))
        self.assertGreaterEqual(len(result.diagnostics), 3)
        for value in result.diagnostics.values():
            self.assertEqual(value.numel(), 1)
            self.assertTrue(torch.isfinite(value).all())

    def test_query_permutation_is_equivariant(self):
        permutation = torch.tensor([2, 0, 3, 1])
        original = self._forward()
        permuted = self.operator(
            self.query[:, permutation],
            self.boxes[:, permutation],
            self.relation_role,
            self.valid[:, permutation],
        )

        torch.testing.assert_close(
            permuted.query_delta, original.query_delta[:, permutation])
        torch.testing.assert_close(
            permuted.box_delta, original.box_delta[:, permutation])
        torch.testing.assert_close(
            permuted.score_delta, original.score_delta[:, permutation])
        torch.testing.assert_close(
            permuted.gate_logits, original.gate_logits[:, permutation])
        self.assertTrue(torch.equal(permuted.valid, self.valid[:, permutation]))
        self.assertEqual(permuted.diagnostics.keys(), original.diagnostics.keys())
        for name in original.diagnostics:
            torch.testing.assert_close(
                permuted.diagnostics[name], original.diagnostics[name])

    def test_invalid_queries_are_zero_and_cannot_influence_valid_queries(self):
        baseline = self._forward()
        changed_query = self.query.clone()
        changed_boxes = self.boxes.clone()
        changed_query[~self.valid] = 1e4
        changed_boxes[~self.valid] = torch.tensor(
            [0.99, 0.01, 0.01, 0.99], dtype=changed_boxes.dtype)

        changed = self.operator(
            changed_query, changed_boxes, self.relation_role, self.valid)

        torch.testing.assert_close(
            changed.query_delta[self.valid], baseline.query_delta[self.valid])
        torch.testing.assert_close(
            changed.box_delta[self.valid], baseline.box_delta[self.valid])
        torch.testing.assert_close(
            changed.score_delta[self.valid], baseline.score_delta[self.valid])
        self.assertEqual(
            changed.query_delta[~self.valid].detach().abs().sum().item(), 0.0)
        self.assertEqual(
            changed.box_delta[~self.valid].detach().abs().sum().item(), 0.0)
        self.assertEqual(
            changed.score_delta[~self.valid].detach().abs().sum().item(), 0.0)
        self.assertEqual(
            changed.gate_logits[~self.valid].detach().abs().sum().item(), 0.0)

    def test_single_and_zero_valid_query_samples_are_finite_and_masked(self):
        valid = torch.tensor(
            [[True, False, False, False], [False, False, False, False]],
            dtype=torch.bool,
        )
        result = self.operator(
            self.query, self.boxes, self.relation_role, valid)

        for value in (
            result.query_delta,
            result.box_delta,
            result.score_delta,
            result.gate_logits,
            *result.diagnostics.values(),
        ):
            self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(
            result.query_delta[~valid].detach().abs().sum().item(), 0.0)
        self.assertEqual(
            result.box_delta[~valid].detach().abs().sum().item(), 0.0)
        self.assertEqual(
            result.score_delta[~valid].detach().abs().sum().item(), 0.0)
        self.assertEqual(
            result.gate_logits[~valid].detach().abs().sum().item(), 0.0)

    def test_zero_initialized_gate_is_exact_structured_fusion_noop(self):
        result = self._forward()
        parent = DecoderResidualState(
            query=self.query.clone(),
            box_logits=torch.randn(2, 4, 4),
            referent_score=torch.randn(2, 4),
        )

        fused = StructuredResidualFusion()(parent, (result,))

        self.assertEqual(result.gate_logits.detach().abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(fused.query, parent.query))
        self.assertTrue(torch.equal(fused.box_logits, parent.box_logits))
        self.assertTrue(torch.equal(
            fused.referent_score, parent.referent_score))

    def test_fused_backward_is_finite_and_reaches_zero_gate_head(self):
        query = self.query.clone().requires_grad_(True)
        boxes = self.boxes.clone().requires_grad_(True)
        relation_role = self.relation_role.clone().requires_grad_(True)
        result = self.operator(query, boxes, relation_role, self.valid)
        parent = DecoderResidualState(
            query=query,
            box_logits=torch.randn(2, 4, 4, requires_grad=True),
            referent_score=torch.randn(2, 4, requires_grad=True),
        )
        fused = StructuredResidualFusion()(parent, (result,))

        loss = (
            fused.query.square().mean()
            + fused.box_logits.square().mean()
            + fused.referent_score.square().mean()
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        for parameter in self.operator.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        gate_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.operator.gate_head.parameters()
        )
        self.assertGreater(gate_gradient, 0.0)
        for value in (query.grad, boxes.grad, relation_role.grad):
            self.assertIsNotNone(value)
            self.assertTrue(torch.isfinite(value).all())

    def test_relation_role_changes_relation_distractor_contrast(self):
        positive = self._forward()
        negative = self.operator(
            self.query, self.boxes, -self.relation_role, self.valid)

        difference = (
            (positive.query_delta - negative.query_delta).abs().sum()
            + (positive.box_delta - negative.box_delta).abs().sum()
            + (positive.score_delta - negative.score_delta).abs().sum()
        )
        self.assertGreater(difference.detach().item(), 0.0)

    def test_public_boundary_rejects_invalid_shapes_values_and_dtypes(self):
        cases = (
            (
                "query",
                (self.query[:, :, :-1], self.boxes,
                 self.relation_role, self.valid),
            ),
            (
                "boxes",
                (self.query, self.boxes[:, :, :3],
                 self.relation_role, self.valid),
            ),
            (
                "relation_role",
                (self.query, self.boxes,
                 self.relation_role[:, :-1], self.valid),
            ),
            (
                "boolean",
                (self.query, self.boxes,
                 self.relation_role, self.valid.float()),
            ),
            (
                "normalized",
                (self.query, self.boxes + 1.0,
                 self.relation_role, self.valid),
            ),
            (
                "finite",
                (
                    self.query.masked_fill(
                        torch.zeros_like(self.query, dtype=torch.bool)
                        .index_fill(2, torch.tensor([0]), True),
                        float("nan"),
                    ),
                    self.boxes,
                    self.relation_role,
                    self.valid,
                ),
            ),
            (
                "dtype",
                (self.query, self.boxes.double(),
                 self.relation_role, self.valid),
            ),
        )
        for message, arguments in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.operator(*arguments)
        with self.assertRaisesRegex(ValueError, "positive"):
            QuerySpatialRelationOperator(d_model=0)

        for query, boxes, role, valid in (
            (
                self.query[:0], self.boxes[:0],
                self.relation_role[:0], self.valid[:0],
            ),
            (
                self.query[:, :0], self.boxes[:, :0],
                self.relation_role, self.valid[:, :0],
            ),
        ):
            with self.subTest(shape=tuple(query.shape)):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    self.operator(query, boxes, role, valid)

    def test_query_chunk_size_accepts_none_or_strictly_positive_integer(self):
        for chunk_size in (None, 1, 3):
            with self.subTest(chunk_size=chunk_size):
                operator = QuerySpatialRelationOperator(
                    d_model=self.d_model,
                    query_chunk_size=chunk_size,
                )
                self.assertEqual(operator.query_chunk_size, chunk_size)

        for chunk_size in (True, False, 0, -1, 1.0, "2"):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(
                    ValueError, "query_chunk_size.*positive integer"
                ):
                    QuerySpatialRelationOperator(
                        d_model=self.d_model,
                        query_chunk_size=chunk_size,
                    )

    def test_target_query_chunking_bounds_pairwise_geometry_peak(self):
        observed_target_widths = []
        dense_geometry = qsro_module._pairwise_box_geometry

        def recording_geometry(
            boxes, *, target_start=0, target_end=None
        ):
            resolved_end = boxes.shape[1] if target_end is None else target_end
            observed_target_widths.append(resolved_end - target_start)
            return dense_geometry(
                boxes,
                target_start=target_start,
                target_end=target_end,
            )

        operator = QuerySpatialRelationOperator(
            d_model=self.d_model,
            query_chunk_size=3,
        )
        operator.load_state_dict(self.operator.state_dict())
        with mock.patch.object(
            qsro_module,
            "_pairwise_box_geometry",
            side_effect=recording_geometry,
        ):
            result = operator(
                self.query,
                self.boxes,
                self.relation_role,
                self.valid,
            )

        self.assertEqual(observed_target_widths, [3, 1])
        self.assertEqual(tuple(result.query_delta.shape), (2, 4, 16))

    def test_chunked_outputs_and_gradients_match_dense_execution(self):
        dense = QuerySpatialRelationOperator(
            d_model=self.d_model,
            query_chunk_size=None,
        ).double()
        chunked = QuerySpatialRelationOperator(
            d_model=self.d_model,
            query_chunk_size=3,
        ).double()
        chunked.load_state_dict(dense.state_dict())

        dense_inputs = (
            self.query.double().clone().requires_grad_(True),
            self.boxes.double().clone().requires_grad_(True),
            self.relation_role.double().clone().requires_grad_(True),
        )
        chunked_inputs = tuple(
            value.detach().clone().requires_grad_(True)
            for value in dense_inputs
        )

        dense_result = dense(*dense_inputs, self.valid)
        chunked_result = chunked(*chunked_inputs, self.valid)
        for dense_value, chunked_value in zip(
            (
                dense_result.query_delta,
                dense_result.box_delta,
                dense_result.score_delta,
                dense_result.gate_logits,
            ),
            (
                chunked_result.query_delta,
                chunked_result.box_delta,
                chunked_result.score_delta,
                chunked_result.gate_logits,
            ),
        ):
            torch.testing.assert_close(
                chunked_value, dense_value, rtol=1e-10, atol=1e-12)
        self.assertEqual(
            chunked_result.diagnostics.keys(),
            dense_result.diagnostics.keys(),
        )
        for name, dense_value in dense_result.diagnostics.items():
            torch.testing.assert_close(
                chunked_result.diagnostics[name],
                dense_value,
                rtol=1e-10,
                atol=1e-12,
            )

        def objective(result):
            return (
                0.7 * result.query_delta.sum()
                + 0.5 * result.box_delta.sum()
                + 0.3 * result.score_delta.sum()
                + 0.2 * result.gate_logits.sum()
                + 0.1 * result.diagnostics[
                    "qsro_attention_entropy"
                ]
                + 0.05 * result.diagnostics[
                    "qsro_contrast_abs_mean"
                ]
            )

        objective(dense_result).backward()
        objective(chunked_result).backward()

        for dense_input, chunked_input in zip(dense_inputs, chunked_inputs):
            self.assertIsNotNone(dense_input.grad)
            self.assertIsNotNone(chunked_input.grad)
            torch.testing.assert_close(
                chunked_input.grad,
                dense_input.grad,
                rtol=1e-9,
                atol=1e-11,
            )
        for (dense_name, dense_parameter), (
            chunked_name,
            chunked_parameter,
        ) in zip(dense.named_parameters(), chunked.named_parameters()):
            self.assertEqual(chunked_name, dense_name)
            self.assertIsNotNone(dense_parameter.grad, msg=dense_name)
            self.assertIsNotNone(chunked_parameter.grad, msg=chunked_name)
            torch.testing.assert_close(
                chunked_parameter.grad,
                dense_parameter.grad,
                rtol=1e-9,
                atol=1e-11,
                msg=dense_name,
            )

    def test_forward_accepts_only_inference_available_inputs(self):
        parameter_names = tuple(
            inspect.signature(
                QuerySpatialRelationOperator.forward).parameters)
        self.assertEqual(
            parameter_names,
            ("self", "query", "boxes", "relation_role", "valid"),
        )

        config_root = Path(__file__).resolve().parents[2] / "configs"
        formal_configs = tuple(config_root.glob("ovha_rod_swin_t_5e_*.py"))
        self.assertGreater(len(formal_configs), 0)
        for config in formal_configs:
            source = config.read_text(encoding="utf-8").lower()
            self.assertNotIn("qsro", source)
            self.assertNotIn("queryspatialrelationoperator", source)

    def test_formal_package_import_does_not_eagerly_load_qsro(self):
        project_root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(project_root)
        command = """
import sys
import ovha_rod
assert 'ovha_rod.models.operators.qsro' not in sys.modules
from ovha_rod.models.operators import QSRO, QuerySpatialRelationOperator
assert QSRO is QuerySpatialRelationOperator
assert 'ovha_rod.models.operators.qsro' in sys.modules
"""
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    def test_lazy_public_exports_resolve_and_unknown_names_fail(self):
        self.assertIs(operator_exports.QSRO, QuerySpatialRelationOperator)
        self.assertIs(
            operator_exports.QuerySpatialRelationOperator,
            QuerySpatialRelationOperator,
        )
        with self.assertRaisesRegex(AttributeError, "missing_qsro_export"):
            getattr(operator_exports, "missing_qsro_export")


if __name__ == "__main__":
    unittest.main()
