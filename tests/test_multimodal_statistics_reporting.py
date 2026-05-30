import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MultimodalStatisticsReportingTests(unittest.TestCase):
    def test_public_summary_reports_mean_std_ci_paired_test_and_metadata(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        rows = _metric_rows()
        summary = summarize_public_results(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        validation = validate_public_summary(summary)

        self.assertTrue(validation.ok, validation.errors)
        main = summary["main_table"]["phrase_region_grounding"]["test"]["ovha_full"]
        self.assertAlmostEqual(main["mean"], 0.75)
        self.assertIn("std", main)
        self.assertIn("ci95", main)
        self.assertEqual(main["seed_count"], 3)
        self.assertIn("paired_permutation_p", summary["paired_tests"]["phrase_region_grounding"]["test"])
        self.assertIn("paired_bootstrap_ci95", summary["paired_tests"]["phrase_region_grounding"]["test"])
        self.assertEqual(len(summary["per_seed_appendix"]), len(rows))
        self.assertEqual(summary["metadata"]["parameter_count"]["ovha_full"], 1234)
        self.assertEqual(summary["metadata"]["training_steps"], 1000)
        self.assertEqual(summary["metadata"]["frozen_feature_extractor_version"]["text"], "clip-text-test")
        self.assertEqual(summary["metadata"]["hardware"]["accelerator"], "A800")

    def test_public_summary_rejects_best_seed_only_and_missing_metadata(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        rows = [_metric_rows()[0]]
        summary = summarize_public_results(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        self.assertIn("at least 3 seeds", "\n".join(validation.errors))
        self.assertIn("paired_tests", "\n".join(validation.errors))

    def test_public_summary_cli_emits_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.jsonl"
            metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in _metric_rows()))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "summarize_public_results.py"),
                    str(metrics_path),
                    "--full-model",
                    "ovha_full",
                    "--baseline-model",
                    "cross_attention_transformer",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("main_table", payload)
        self.assertTrue(payload["validation"]["ok"])


def _metric_rows():
    rows = []
    for seed, full_score, baseline_score in ((11, 0.70, 0.65), (12, 0.75, 0.70), (13, 0.80, 0.72)):
        for model, score in (("ovha_full", full_score), ("cross_attention_transformer", baseline_score)):
            rows.append(
                {
                    "task": "phrase_region_grounding",
                    "dataset": "refcoco",
                    "split": "test",
                    "model": model,
                    "seed": seed,
                    "score": score,
                    "higher_is_better": True,
                    "parameter_count": 1234 if model == "ovha_full" else 1000,
                    "training_steps": 1000,
                    "frozen_feature_extractor_version": {"text": "clip-text-test", "region": "clip-region-test"},
                    "hardware": {"accelerator": "A800", "wall_clock_hours": 1.5},
                    "raw_metric_path": f"outputs/mock/{model}/seed_{seed}.jsonl",
                }
            )
    return rows


if __name__ == "__main__":
    unittest.main()
