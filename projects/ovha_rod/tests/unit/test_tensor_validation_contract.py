import ast
import unittest
from pathlib import Path

import torch

from ovha_rod.models.operators.tensor_validation import (
    tensor_value_checks_enabled,
)


ROOT = Path(__file__).resolve().parents[2]
OPERATORS = ROOT / "ovha_rod/models/operators"
HOT_FILES = (
    "decoder_bank.py",
    "decoder_contracts.py",
    "decoder_integration.py",
    "hyper_adapter.py",
    "ms_tleo.py",
    "operator_memory.py",
    "operator_router.py",
    "qsro.py",
    "rceo.py",
    "tq_cato.py",
)


class TensorValidationContractTests(unittest.TestCase):
    def test_cpu_contract_value_checks_remain_enabled(self):
        self.assertTrue(tensor_value_checks_enabled(torch.zeros(1)))

    def test_hot_operator_files_use_device_aware_value_guard(self):
        for filename in HOT_FILES:
            with self.subTest(filename=filename):
                source = (OPERATORS / filename).read_text(encoding="utf-8")
                ast.parse(source)
                self.assertIn("tensor_value_checks_enabled", source)


if __name__ == "__main__":
    unittest.main()
