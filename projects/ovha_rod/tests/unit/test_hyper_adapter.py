import unittest

import torch

from ovha_rod.models.operators.hyper_adapter import LowRankHyperAdapter


class LowRankHyperAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)
        self.query = torch.randn(2, 4, 8, requires_grad=True)
        self.memory = torch.randn(2, 4, 8, requires_grad=True)
        self.valid = torch.tensor(
            [[True, True, False, True], [True, False, True, False]])
        self.adapter = LowRankHyperAdapter(
            d_model=8, operator_count=3, rank=2)

    def test_zero_initialization_is_exact_neutral_and_masked(self):
        result = self.adapter(self.query, self.memory, self.valid)
        self.assertEqual(tuple(result.scale.shape), (2, 4, 3, 8))
        self.assertEqual(tuple(result.shift.shape), (2, 4, 3, 8))
        self.assertEqual(tuple(result.channel_delta.shape), (2, 4, 3, 3))
        self.assertTrue(torch.equal(result.scale, torch.zeros_like(result.scale)))
        self.assertTrue(torch.equal(result.shift, torch.zeros_like(result.shift)))
        self.assertTrue(torch.equal(
            result.channel_delta, torch.zeros_like(result.channel_delta)))

    def test_nonzero_basis_is_permutation_equivariant_and_finite(self):
        with torch.no_grad():
            self.adapter.scale_basis.fill_(0.05)
            self.adapter.shift_basis.fill_(0.02)
            self.adapter.channel_basis.fill_(0.03)
        permutation = torch.tensor([3, 1, 0, 2])
        original = self.adapter(self.query, self.memory, self.valid)
        permuted = self.adapter(
            self.query[:, permutation],
            self.memory[:, permutation],
            self.valid[:, permutation],
        )
        self.assertTrue(torch.allclose(
            permuted.scale, original.scale[:, permutation], atol=1e-6))
        self.assertTrue(torch.equal(
            original.scale[~self.valid],
            torch.zeros_like(original.scale[~self.valid]),
        ))
        loss = (
            original.scale.square().mean()
            + original.shift.square().mean()
            + original.channel_delta.square().mean()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(self.query.grad).all())

    def test_invalid_rank_shapes_and_nonfinite_inputs_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            LowRankHyperAdapter(d_model=0, operator_count=3, rank=1)
        with self.assertRaisesRegex(ValueError, "rank"):
            LowRankHyperAdapter(d_model=8, operator_count=3, rank=9)
        with self.assertRaisesRegex(ValueError, "memory"):
            self.adapter(self.query, self.memory[:, :2], self.valid)
        bad = self.memory.detach().clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.adapter(self.query, bad, self.valid)
        with self.assertRaisesRegex(ValueError, "query"):
            self.adapter(
                self.query[..., :7], self.memory[..., :7], self.valid)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.adapter(self.query, self.memory, self.valid.float())
        with self.assertRaisesRegex(ValueError, "floating"):
            self.adapter(
                self.query.to(torch.int64), self.memory.to(torch.int64),
                self.valid)
        with self.assertRaisesRegex(ValueError, "device and dtype"):
            self.adapter(self.query, self.memory.double(), self.valid)


if __name__ == "__main__":
    unittest.main()
