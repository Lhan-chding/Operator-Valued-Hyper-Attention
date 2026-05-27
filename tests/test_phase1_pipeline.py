import json
import tempfile
import unittest
from pathlib import Path

from moat_ovha.config import Phase1Config
from moat_ovha.eval_meta_operator import run_evaluation
from moat_ovha.train_meta_operator import run_training


class Phase1PipelineTests(unittest.TestCase):
    def test_training_and_evaluation_write_jsonl_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = Phase1Config(
                seed=3,
                output_dir=output_dir,
                families=("fourier", "green", "separable", "mixed"),
                context_sizes=(2, 4),
                resolution=10,
                baselines=("transformer_only", "deeponet", "fno", "simple_stack"),
            )

            train_path = run_training(config)
            eval_path = run_evaluation(config)

            self.assertTrue(train_path.exists())
            self.assertTrue(eval_path.exists())
            train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
            eval_rows = [json.loads(line) for line in eval_path.read_text().splitlines()]
            self.assertGreater(len(train_rows), 0)
            self.assertGreater(len(eval_rows), 0)
            self.assertIn("relative_l2", train_rows[0])
            self.assertIn("parameter_count", eval_rows[0])


if __name__ == "__main__":
    unittest.main()
