import inspect
import unittest

import torch

from ovha_rod.models.operators.rceo import RCEO


class RCEOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.valid = torch.tensor(
            [[True, True, False, True], [True, False, True, False]])
        self.operator_available = torch.tensor(
            [
                [[True, True, True], [True, False, True],
                 [False, False, False], [True, True, False]],
                [[True, True, True], [False, False, False],
                 [True, False, True], [False, False, False]],
            ]
        )
        self.text_valid = torch.tensor(
            [[True, True, True, False, False],
             [True, False, False, False, False]])
        self.valid_ratios = torch.tensor(
            [
                [[1.0, 1.0], [0.9, 0.8], [0.8, 0.7]],
                [[0.7, 0.9], [0.6, 0.8], [0.5, 0.7]],
            ],
            requires_grad=True,
        )
        self.rceo = RCEO(d_model=8, operator_count=3, prior_cap=2.0)

    def _forward(self, **overrides):
        values = {
            "operator_available": self.operator_available,
            "valid": self.valid,
            "text_valid": self.text_valid,
            "valid_ratios": self.valid_ratios,
        }
        values.update(overrides)
        return self.rceo(**values)

    def test_initial_prior_is_clean_neutral_bounded_and_masked(self):
        result = self._forward()
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

    def test_uses_explicit_quality_and_mask_evidence_and_is_equivariant(self):
        parameters = set(inspect.signature(self.rceo.forward).parameters)
        self.assertEqual(
            parameters,
            {"operator_available", "valid", "text_valid", "valid_ratios"},
        )
        permutation = torch.tensor([2, 0, 3, 1])
        original = self._forward()
        permuted = self._forward(
            operator_available=self.operator_available[:, permutation],
            valid=self.valid[:, permutation],
        )
        self.assertTrue(torch.allclose(
            permuted.log_prior, original.log_prior[:, permutation], atol=1e-6))

    def test_learned_prior_responds_to_missing_text_and_backpropagates(self):
        with torch.no_grad():
            for parameter in self.rceo.parameters():
                parameter.normal_(std=0.1)
        result = self._forward()
        centered = result.log_prior[self.valid].mean(dim=-1)
        self.assertTrue(torch.allclose(centered, torch.zeros_like(centered), atol=1e-6))
        loss = result.log_prior.square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.valid_ratios.grad)
        self.assertTrue(torch.isfinite(self.valid_ratios.grad).all())

        missing_text = self.text_valid.clone()
        missing_text[0] = False
        corrupted = self._forward(
            text_valid=missing_text,
            valid_ratios=self.valid_ratios.detach(),
        )
        self.assertFalse(torch.allclose(
            corrupted.log_prior[0, self.valid[0]],
            result.log_prior.detach()[0, self.valid[0]],
        ))

    def test_optional_missing_evidence_is_explicit_and_finite(self):
        result = self._forward(text_valid=None, valid_ratios=None)
        self.assertTrue(torch.isfinite(result.log_prior).all())
        self.assertTrue(torch.equal(
            result.log_prior[~self.valid],
            torch.zeros_like(result.log_prior[~self.valid]),
        ))

    def test_shape_dtype_and_nonfinite_inputs_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            RCEO(d_model=0, operator_count=3)
        with self.assertRaisesRegex(ValueError, "prior_cap"):
            RCEO(d_model=8, operator_count=3, prior_cap=0.0)
        with self.assertRaisesRegex(ValueError, "operator_available"):
            self._forward(operator_available=self.operator_available[:, :3])
        with self.assertRaisesRegex(ValueError, "boolean"):
            self._forward(valid=self.valid.float())
        with self.assertRaisesRegex(ValueError, "text_valid"):
            self._forward(text_valid=self.text_valid.float())
        with self.assertRaisesRegex(ValueError, "valid_ratios"):
            self._forward(valid_ratios=self.valid_ratios[:, :2])
        bad = self.valid_ratios.detach().clone()
        bad[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            self._forward(valid_ratios=bad)
        with self.assertRaisesRegex(ValueError, "floating"):
            self._forward(valid_ratios=self.valid_ratios.to(torch.int64))
        with self.assertRaisesRegex(ValueError, "range"):
            self._forward(valid_ratios=self.valid_ratios.detach() + 1.0)


if __name__ == "__main__":
    unittest.main()
