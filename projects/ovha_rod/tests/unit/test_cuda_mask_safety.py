import unittest
from unittest import mock

import torch

from ovha_rod.models.operators import operator_router as router_module
from ovha_rod.models.operators import tq_cato as tq_cato_module
from ovha_rod.models.operators.operator_router import OperatorRouter
from ovha_rod.models.operators.tq_cato import TQCATO


class CUDAMaskSafetyTests(unittest.TestCase):
    """Emulate CUDA's synchronization-free validation path on CPU."""

    def test_tq_cato_all_empty_masks_return_finite_zero_fallback(self):
        query = torch.randn(2, 3, 8)
        text = torch.randn(2, 4, 8)
        query_valid = torch.zeros(2, 3, dtype=torch.bool)
        text_valid = torch.zeros(2, 4, dtype=torch.bool)

        with mock.patch.object(
            tq_cato_module,
            "tensor_value_checks_enabled",
            return_value=False,
        ):
            result = TQCATO(d_model=8)(
                query, text, query_valid, text_valid)

        self.assertTrue(torch.isfinite(result.transport).all())
        self.assertEqual(int(torch.count_nonzero(result.transport)), 0)
        for value in result.residual.diagnostics.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_router_all_unavailable_returns_finite_zero_fallback(self):
        query = torch.randn(2, 3, 8)
        memory = torch.randn(2, 3, 8)
        valid = torch.ones(2, 3, dtype=torch.bool)
        unavailable = torch.zeros(3, dtype=torch.bool)
        router = OperatorRouter(
            d_model=8,
            operator_count=3,
            num_layers=6,
            hidden_dim=12,
        )

        with mock.patch.object(
            router_module,
            "tensor_value_checks_enabled",
            return_value=False,
        ):
            result = router(
                query,
                memory,
                layer_index=0,
                valid=valid,
                operator_available=unavailable,
            )

        self.assertTrue(torch.isfinite(result.weights).all())
        self.assertTrue(torch.isfinite(result.logits).all())
        self.assertEqual(int(torch.count_nonzero(result.weights)), 0)
        self.assertEqual(int(torch.count_nonzero(result.logits)), 0)


if __name__ == "__main__":
    unittest.main()
