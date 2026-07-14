import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import torch

from ovha_rod.models.operators.decoder_contracts import (
    DecoderResidualState,
    StructuredResidualFusion,
)
from ovha_rod.models.operators.tq_cato import TQCATO, TQCATOResult


ROOT = Path(__file__).resolve().parents[2]


class TQCATOTests(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(71)
        return (
            torch.randn(2, 4, 8),
            torch.randn(2, 5, 8),
            torch.tensor(
                [[True, True, False, True], [True, False, True, False]]),
            torch.tensor(
                [[True, False, True, True, False], [False, True, True, False, True]]),
        )

    def test_transport_is_row_normalized_and_masks_exactly(self):
        query, text, query_valid, text_valid = self._inputs()
        result = TQCATO(d_model=8)(query, text, query_valid, text_valid)

        self.assertEqual(tuple(result.transport.shape), (2, 5, 4))
        valid_pairs = text_valid[..., None] & query_valid[:, None, :]
        self.assertTrue(torch.equal(result.transport[~valid_pairs], torch.zeros_like(
            result.transport[~valid_pairs])))
        valid_rows = result.transport.sum(dim=-1)[text_valid]
        self.assertTrue(torch.allclose(
            valid_rows, torch.ones_like(valid_rows), atol=1e-6, rtol=0.0))
        self.assertTrue(torch.equal(result.residual.valid, query_valid))
        with self.assertRaises(FrozenInstanceError):
            result.transport = torch.zeros_like(result.transport)

    def test_every_sample_requires_valid_queries_and_text_tokens(self):
        query, text, query_valid, text_valid = self._inputs()
        query_valid[1] = False
        with self.assertRaisesRegex(ValueError, "sample 1.*valid query"):
            TQCATO(d_model=8)(query, text, query_valid, text_valid)

        query, text, query_valid, text_valid = self._inputs()
        text_valid[0] = False
        with self.assertRaisesRegex(ValueError, "sample 0.*valid text"):
            TQCATO(d_model=8)(query, text, query_valid, text_valid)

    def test_constructor_rejects_invalid_dimensions_and_temperatures(self):
        for d_model in (0, -1, 1.5, True):
            with self.subTest(d_model=d_model):
                with self.assertRaisesRegex(ValueError, "d_model"):
                    TQCATO(d_model=d_model)
        for temperature in (0.0, -1.0, float("inf"), float("nan"), "hot", True):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    TQCATO(d_model=8, temperature=temperature)

    def test_input_contract_rejects_shapes_masks_dtypes_and_nonfinite(self):
        query, text, query_valid, text_valid = self._inputs()
        operator = TQCATO(d_model=8)
        cases = (
            ((query[:, :, :7], text, query_valid, text_valid), "dimension"),
            ((query, text[:, :, :7], query_valid, text_valid), "dimension"),
            ((query, text, query_valid[:, :3], text_valid), "query_valid"),
            ((query, text, query_valid, text_valid.float()), "boolean"),
            ((query.double(), text, query_valid, text_valid), "dtype"),
        )
        for arguments, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    operator(*arguments)

        query = query.clone()
        query[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            operator(query, text, query_valid, text_valid)

    def test_input_contract_rejects_boundary_types_and_devices(self):
        query, text, query_valid, text_valid = self._inputs()
        operator = TQCATO(d_model=8)
        cases = (
            ((None, text, query_valid, text_valid), "query.*tensor"),
            ((query[0], text, query_valid, text_valid), "shape"),
            ((query, text[:1], query_valid, text_valid[:1]), "batch"),
            ((query.to(torch.int64), text, query_valid, text_valid), "floating"),
            ((query, text, query_valid, text_valid[:, :4]), "text_valid"),
            (
                (query, torch.empty(text.shape, device="meta"), query_valid, text_valid),
                "share a device",
            ),
            (
                (
                    query,
                    text,
                    torch.empty(query_valid.shape, dtype=torch.bool, device="meta"),
                    text_valid,
                ),
                "masks must share a device",
            ),
        )
        for arguments, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    operator(*arguments)

    def test_result_contract_rejects_invalid_payloads(self):
        query, text, query_valid, text_valid = self._inputs()
        result = TQCATO(d_model=8)(query, text, query_valid, text_valid)
        self.assertIsInstance(result, TQCATOResult)
        cases = (
            ({"residual": object()}, "DecoderOperatorResidual"),
            ({"transport": result.transport[:, 0]}, "shape"),
            ({"transport": result.transport[:, :, :3]}, "shapes must agree"),
            ({"transport": torch.zeros(2, 5, 4, dtype=torch.int64)}, "floating"),
            ({"transport": result.transport.double()}, "dtype"),
            (
                {"transport": torch.empty(result.transport.shape, device="meta")},
                "device",
            ),
        )
        for changes, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    replace(result, **changes)

        nonfinite = result.transport.clone()
        nonfinite[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(result, transport=nonfinite)

    def test_zero_initialized_gate_is_exact_structured_fusion_noop(self):
        query, text, query_valid, text_valid = self._inputs()
        operator = TQCATO(d_model=8)
        result = operator(query, text, query_valid, text_valid)
        parent = DecoderResidualState(
            query=query,
            box_logits=torch.randn(2, 4, 4),
            referent_score=torch.randn(2, 4),
        )

        fused = StructuredResidualFusion()(parent, (result.residual,))

        self.assertTrue(torch.count_nonzero(result.residual.gate_logits) == 0)
        self.assertTrue(torch.equal(fused.query, parent.query))
        self.assertTrue(torch.equal(fused.box_logits, parent.box_logits))
        self.assertTrue(torch.equal(fused.referent_score, parent.referent_score))
        self.assertTrue(torch.count_nonzero(result.residual.box_delta) == 0)
        self.assertTrue(torch.count_nonzero(result.residual.score_delta) == 0)

    def test_query_and_text_permutations_preserve_operator_contract(self):
        query, text, query_valid, text_valid = self._inputs()
        operator = TQCATO(d_model=8)
        original = operator(query, text, query_valid, text_valid)

        query_order = torch.tensor([2, 0, 3, 1])
        query_permuted = operator(
            query[:, query_order], text, query_valid[:, query_order], text_valid)
        self.assertTrue(torch.allclose(
            query_permuted.transport,
            original.transport[:, :, query_order], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(
            query_permuted.residual.query_delta,
            original.residual.query_delta[:, query_order], atol=1e-6, rtol=1e-6))

        text_order = torch.tensor([3, 0, 4, 1, 2])
        text_permuted = operator(
            query, text[:, text_order], query_valid, text_valid[:, text_order])
        self.assertTrue(torch.allclose(
            text_permuted.transport,
            original.transport[:, text_order], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(
            text_permuted.residual.query_delta,
            original.residual.query_delta, atol=1e-6, rtol=1e-6))

    def test_extreme_logits_and_backward_remain_finite(self):
        query, text, query_valid, text_valid = self._inputs()
        query = (query * 1e4).requires_grad_(True)
        text = (text * 1e4).requires_grad_(True)
        operator = TQCATO(d_model=8)
        with torch.no_grad():
            operator.gate_head.bias.fill_(0.25)

        result = operator(query, text, query_valid, text_valid)
        parent = DecoderResidualState(
            query=query,
            box_logits=torch.zeros(2, 4, 4),
            referent_score=torch.zeros(2, 4),
        )
        fused = StructuredResidualFusion()(parent, (result.residual,))
        loss = fused.query.square().mean()
        loss.backward()

        gradients = [query.grad, text.grad]
        gradients.extend(
            parameter.grad for parameter in operator.parameters()
            if parameter.requires_grad)
        self.assertTrue(torch.isfinite(result.transport).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_formal_configs_do_not_enable_tq_cato(self):
        for path in sorted((ROOT / "configs").glob("ovha_rod_swin_t_5e_*.py")):
            with self.subTest(config=path.name):
                source = path.read_text(encoding="utf-8").lower()
                self.assertNotIn("tq_cato", source)
                self.assertNotIn("tqcato", source)


if __name__ == "__main__":
    unittest.main()
