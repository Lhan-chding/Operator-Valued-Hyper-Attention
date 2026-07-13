import unittest
from pathlib import Path

import torch

from ovha_rod.gradient_accumulator import (
    accumulate_gradient_square,
    finalize_gradient_norm,
)


ROOT = Path(__file__).resolve().parents[2]


class GradientAccumulatorTests(unittest.TestCase):
    def test_accumulation_stays_tensor_native_and_detached_until_finalize(self):
        first = torch.tensor([3.0, 4.0], requires_grad=True)
        second = torch.tensor([1.0, 2.0], requires_grad=True)

        accumulated = accumulate_gradient_square(None, first)
        accumulated = accumulate_gradient_square(accumulated, second)

        self.assertIsInstance(accumulated, torch.Tensor)
        self.assertFalse(accumulated.requires_grad)
        self.assertEqual(accumulated.device, first.device)
        self.assertAlmostEqual(float(accumulated), 30.0)
        self.assertAlmostEqual(finalize_gradient_norm(accumulated), 30.0 ** 0.5)

    def test_empty_accumulator_finalizes_to_zero(self):
        self.assertEqual(finalize_gradient_norm(None), 0.0)

    def test_hook_does_not_synchronize_inside_each_parameter_callback(self):
        source = (
            ROOT / "ovha_rod/hooks/operator_diagnostics_hook.py").read_text()
        callback = source[source.index("def _record_gradient"):]

        self.assertIn("accumulate_gradient_square", callback)
        self.assertNotIn(".cpu()", callback)
        self.assertNotIn("float(", callback)
        self.assertIn("finalize_gradient_norm", source)


if __name__ == "__main__":
    unittest.main()
