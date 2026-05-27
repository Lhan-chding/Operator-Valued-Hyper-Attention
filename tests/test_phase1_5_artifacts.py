import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from moat_ovha_torch.config import Phase15Config


ROOT = Path(__file__).resolve().parents[1]


class Phase15ArtifactTests(unittest.TestCase):
    def test_phase15_configs_and_docs_exist(self):
        expected = [
            ROOT / "configs" / "phase1_5_cpu_smoke.json",
            ROOT / "configs" / "phase1_5_gpu_small.json",
            ROOT / "configs" / "phase1_5_gpu_main.json",
            ROOT / "configs" / "phase1_5_ablation_matrix.json",
            ROOT / "docs" / "execution_policy_mac_vs_a800.md",
            ROOT / "docs" / "phases" / "phase_1_5.md",
            ROOT / "theory_notes" / "ovha_context_identifiability.md",
            ROOT / "theory_notes" / "ovha_phase1_5_boundaries.md",
            ROOT / "requirements_torch.txt",
        ]

        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists())

    def test_cpu_smoke_config_is_small_enough_for_mac(self):
        config = Phase15Config.from_file(ROOT / "configs" / "phase1_5_cpu_smoke.json")

        self.assertLessEqual(config.batch_size, 4)
        self.assertLessEqual(config.support_points, 32)
        self.assertLessEqual(config.query_points, 32)
        self.assertLessEqual(config.steps, 50)
        self.assertEqual(config.device, "cpu")
        self.assertFalse(config.allow_metadata_inputs)

    def test_train_cli_skips_cleanly_without_torch_or_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "phase1_5_cli"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "train_torch_meta_operator.py"),
                    "--config",
                    str(ROOT / "configs" / "phase1_5_cpu_smoke.json"),
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            environment_path = output_dir / "environment.json"
            report_path = output_dir / "phase1_5_report.md"
            self.assertTrue(environment_path.exists())
            self.assertTrue(report_path.exists())
            environment = json.loads(environment_path.read_text())
            if not environment["torch_available"]:
                self.assertIn("Torch is not installed", report_path.read_text())
            else:
                self.assertTrue((output_dir / "train_metrics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
