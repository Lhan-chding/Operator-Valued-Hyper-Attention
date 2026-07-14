import inspect
import unittest

import torch

from ovha_rod.models.operators.rceo import RCEO


class RCEOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.query = torch.randn(2, 4, 8, requires_grad=True)
        self.boxes = torch.tensor(
            [
                [[0.5, 0.5, 0.4, 0.3], [0.1, 0.2, 0.1, 0.1],
                 [0.8, 0.8, 0.2, 0.2], [0.4, 0.6, 0.5, 0.4]],
                [[0.3, 0.2, 0.2, 0.2], [0.7, 0.7, 0.1, 0.1],
                 [0.5, 0.4, 0.3, 0.3], [0.2, 0.8, 0.2, 0.1]],
            ],
            dtype=torch.float32,
        )
        self.score = torch.randn(2, 4)
        self.valid = torch.tensor(
            [[True, True, False, True], [True, False, True, False]])
        self.rceo = RCEO(d_model=8, operator_count=3, prior_cap=2.0)

    def test_initial_prior_is_clean_neutral_bounded_and_masked(self):
        result = self.rceo(self.query, self.boxes, self.score, self.valid)
        self.assertTrue(torch.equal(
            result.log_prior[self.valid],
            torch.zeros_like(result.log_prior[self.valid]),
        ))
        self.assertTrue(torch.equal(
            result.log_prior[~self.valid],
            torch.zeros_like(result.log_prior[~self.valid]),
        ))
        self.assertTrue(torch.allclose(
            result.reliability[self.valid],
            torch.full_like(result.reliability[self.valid], 0.5),
        ))
        self.assertLessEqual(result.log_prior.abs().max().item(), 2.0)

    def test_uses_only_inference_available_inputs_and_is_equivariant(self):
        parameters = set(inspect.signature(self.rceo.forward).parameters)
        self.assertEqual(
            parameters,
            {"query", "boxes", "referent_score", "valid"},
        )
        permutation = torch.tensor([2, 0, 3, 1])
        original = self.rceo(self.query, self.boxes, self.score, self.valid)
        permuted = self.rceo(
            self.query[:, permutation],
            self.boxes[:, permutation],
            self.score[:, permutation],
            self.valid[:, permutation],
        )
        self.assertTrue(torch.allclose(
            permuted.log_prior, original.log_prior[:, permutation], atol=1e-6))

    def test_learned_prior_is_finite_centered_and_backpropagates(self):
        with torch.no_grad():
            self.rceo.output.weight.normal_(std=0.05)
        result = self.rceo(self.query, self.boxes, self.score, self.valid)
        centered = result.log_prior[self.valid].mean(dim=-1)
        self.assertTrue(torch.allclose(centered, torch.zeros_like(centered), atol=1e-6))
        loss = result.log_prior.square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(self.query.grad).all())

    def test_shape_dtype_and_nonfinite_inputs_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            RCEO(d_model=0, operator_count=3)
        with self.assertRaisesRegex(ValueError, "prior_cap"):
            RCEO(d_model=8, operator_count=3, prior_cap=0.0)
        with self.assertRaisesRegex(ValueError, "boxes"):
            self.rceo(self.query, self.boxes[:, :3], self.score, self.valid)
        with self.assertRaisesRegex(ValueError, "referent_score"):
            self.rceo(self.query, self.boxes, self.score[:, :3], self.valid)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.rceo(self.query, self.boxes, self.score, self.valid.float())
        bad = self.boxes.clone()
        bad[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.rceo(self.query, bad, self.score, self.valid)
        with self.assertRaisesRegex(ValueError, "floating"):
            self.rceo(
                self.query.to(torch.int64), self.boxes.to(torch.int64),
                self.score.to(torch.int64), self.valid)
        with self.assertRaisesRegex(ValueError, "device and dtype"):
            self.rceo(self.query, self.boxes.double(), self.score, self.valid)


if __name__ == "__main__":
    unittest.main()
