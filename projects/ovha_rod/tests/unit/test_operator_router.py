import unittest

import torch

from ovha_rod.models.operators.operator_router import OperatorRouter


class OperatorRouterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.query = torch.randn(2, 4, 8, requires_grad=True)
        self.memory = torch.randn(2, 4, 8, requires_grad=True)
        self.valid = torch.tensor(
            [[True, True, False, True], [True, False, True, False]])
        self.router = OperatorRouter(
            d_model=8, operator_count=3, num_layers=6, hidden_dim=12)

    def test_weights_are_normalized_masked_and_prior_conditioned(self):
        neutral = self.router(
            self.query, self.memory, layer_index=2, valid=self.valid)
        prior = torch.zeros(2, 4, 3)
        prior[..., 1] = 2.0
        conditioned = self.router(
            self.query,
            self.memory,
            layer_index=2,
            valid=self.valid,
            reliability_log_prior=prior,
        )

        self.assertEqual(tuple(neutral.weights.shape), (2, 4, 3))
        valid_weight_sums = neutral.weights[self.valid].sum(-1)
        self.assertTrue(torch.allclose(
            valid_weight_sums,
            torch.ones_like(valid_weight_sums),
        ))
        self.assertTrue(torch.equal(
            neutral.weights[~self.valid], torch.zeros_like(neutral.weights[~self.valid])))
        self.assertGreater(
            conditioned.weights[self.valid][:, 1].mean().item(),
            neutral.weights[self.valid][:, 1].mean().item(),
        )

    def test_availability_mask_and_all_unavailable_fail_fast(self):
        available = torch.tensor([True, False, True])
        result = self.router(
            self.query,
            self.memory,
            layer_index=0,
            valid=self.valid,
            operator_available=available,
        )
        self.assertTrue(torch.equal(
            result.weights[..., 1], torch.zeros_like(result.weights[..., 1])))
        with self.assertRaisesRegex(ValueError, "available"):
            self.router(
                self.query,
                self.memory,
                layer_index=0,
                valid=self.valid,
                operator_available=torch.zeros(3, dtype=torch.bool),
            )

    def test_query_permutation_equivariance_and_finite_backward(self):
        permutation = torch.tensor([2, 0, 3, 1])
        original = self.router(
            self.query, self.memory, layer_index=4, valid=self.valid)
        permuted = self.router(
            self.query[:, permutation],
            self.memory[:, permutation],
            layer_index=4,
            valid=self.valid[:, permutation],
        )
        self.assertTrue(torch.allclose(
            permuted.weights, original.weights[:, permutation], atol=1e-6))

        loss = original.weights.square().sum() + original.logits.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(self.query.grad).all())
        self.assertTrue(torch.isfinite(self.memory.grad).all())

    def test_bad_shapes_layer_and_nonfinite_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "memory"):
            self.router(
                self.query, self.memory[:, :3], layer_index=0, valid=self.valid)
        with self.assertRaisesRegex(ValueError, "layer_index"):
            self.router(
                self.query, self.memory, layer_index=6, valid=self.valid)
        bad = self.query.detach().clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.router(bad, self.memory, layer_index=0, valid=self.valid)


if __name__ == "__main__":
    unittest.main()
