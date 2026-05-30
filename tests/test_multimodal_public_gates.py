import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MultimodalPublicGateTests(unittest.TestCase):
    def test_region_text_gate_requires_full_baseline_and_cato_evidence(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        report = evaluate_region_text_gate(
            statistics_summary=_summary("phrase_region_grounding", "test", full=0.80, baseline=0.72),
            diagnostics_rows=[
                _diagnostic("clean", {"CATO": 0.55, "TLEO": 0.2, "SPO": 0.15, "LRIO": 0.1}, cato_entropy=0.30),
                _diagnostic("no_cato", {"CATO": 0.0, "TLEO": 0.4, "SPO": 0.4, "LRIO": 0.2}, cato_entropy=0.90),
            ],
            no_cato_score=0.70,
            task="phrase_region_grounding",
            split="test",
        )

        self.assertTrue(report["passed"], report["reasons"])
        self.assertTrue(report["checks"]["full_beats_same_feature_baseline"]["passed"])
        self.assertTrue(report["checks"]["no_cato_drops"]["passed"])
        self.assertTrue(report["checks"]["cato_router_load_high"]["passed"])
        self.assertTrue(report["checks"]["alignment_entropy_improves"]["passed"])

    def test_sentiment_gate_requires_lrio_spo_rceo_and_robustness(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_sentiment_gate

        report = evaluate_sentiment_gate(
            statistics_summary=_summary("sentiment_emotion", "test", full=0.76, baseline=0.74),
            diagnostics_rows=[
                {
                    "setting": "clean",
                    "router_load_by_candidate": {"TLEO": 0.1, "SPO": 0.35, "LRIO": 0.40, "CATO": 0.15},
                    "adapter_params": {"LRIO_rank_entropy": 0.6, "SPO_prototype_entropy": 0.5},
                }
            ],
            ablation_scores={"ovha_no_lrio": 0.70, "ovha_no_spo": 0.71, "ovha_no_rceo": 0.68},
            robustness_summary={
                "full_drop_less_than_baseline": True,
                "rceo_reliability_monotonic": True,
                "required_ablation_degradation": {"passed": True, "reasons": []},
                "operator_load_shift": {"LRIO": -0.20, "SPO": 0.15},
            },
            task="sentiment_emotion",
            split="test",
        )

        self.assertTrue(report["passed"], report["reasons"])
        self.assertTrue(report["checks"]["no_lrio_drops"]["passed"])
        self.assertTrue(report["checks"]["no_rceo_drops"]["passed"])
        self.assertTrue(report["checks"]["robustness_passes"]["passed"])

    def test_public_gate_blocks_when_delta_or_ablation_missing(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        report = evaluate_region_text_gate(
            statistics_summary=_summary("phrase_region_grounding", "test", full=0.70, baseline=0.72),
            diagnostics_rows=[],
            no_cato_score=None,
            task="phrase_region_grounding",
            split="test",
        )

        self.assertFalse(report["passed"])
        self.assertIn("full model does not beat same-feature baseline", "\n".join(report["reasons"]))
        self.assertIn("no-CATO ablation score missing", "\n".join(report["reasons"]))

    def test_public_gate_requires_three_seed_paired_statistical_evidence(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        report = evaluate_region_text_gate(
            statistics_summary=_summary(
                "phrase_region_grounding",
                "test",
                full=0.80,
                baseline=0.72,
                seed_count=1,
                common_seed_count=1,
                include_bootstrap=False,
            ),
            diagnostics_rows=[
                _diagnostic("clean", {"CATO": 0.55, "TLEO": 0.2, "SPO": 0.15, "LRIO": 0.1}, cato_entropy=0.30),
                _diagnostic("no_cato", {"CATO": 0.0, "TLEO": 0.4, "SPO": 0.4, "LRIO": 0.2}, cato_entropy=0.90),
            ],
            no_cato_score=0.70,
            task="phrase_region_grounding",
            split="test",
        )

        self.assertFalse(report["passed"])
        joined = "\n".join(report["reasons"])
        self.assertIn("full and baseline comparison requires at least 3 seeds", joined)
        self.assertIn("paired comparison requires at least 3 common seeds", joined)
        self.assertIn("paired comparison missing paired_bootstrap_ci95", joined)

    def test_public_gate_cli_emits_go_no_go_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats_path = root / "stats.json"
            diagnostics_path = root / "diagnostics.jsonl"
            stats_path.write_text(json.dumps(_summary("phrase_region_grounding", "test", full=0.80, baseline=0.72)))
            diagnostics_path.write_text(
                json.dumps(_diagnostic("clean", {"CATO": 0.55, "TLEO": 0.2, "SPO": 0.15, "LRIO": 0.1}, cato_entropy=0.30))
                + "\n"
                + json.dumps(_diagnostic("no_cato", {"CATO": 0.0, "TLEO": 0.4, "SPO": 0.4, "LRIO": 0.2}, cato_entropy=0.90))
                + "\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "evaluate_public_gates.py"),
                    "region_text",
                    str(stats_path),
                    str(diagnostics_path),
                    "--task",
                    "phrase_region_grounding",
                    "--split",
                    "test",
                    "--no-cato-score",
                    "0.70",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["passed"], payload["reasons"])


def _summary(
    task: str,
    split: str,
    *,
    full: float,
    baseline: float,
    seed_count: int = 3,
    common_seed_count: int = 3,
    include_bootstrap: bool = True,
) -> dict[str, object]:
    paired = {"common_seed_count": common_seed_count, "paired_permutation_p": 0.25}
    if include_bootstrap:
        paired["paired_bootstrap_ci95"] = [0.01, 0.12]
    return {
        "main_table": {
            task: {
                split: {
                    "ovha_full": {"mean": full, "seed_count": seed_count, "higher_is_better": True},
                    "cross_attention_transformer": {"mean": baseline, "seed_count": seed_count, "higher_is_better": True},
                }
            }
        },
        "paired_tests": {task: {split: paired}},
    }


def _diagnostic(setting: str, loads: dict[str, float], *, cato_entropy: float) -> dict[str, object]:
    return {
        "setting": setting,
        "router_load_by_candidate": loads,
        "candidate_diagnostics": {"CATO": {"alignment_entropy": cato_entropy}},
    }


if __name__ == "__main__":
    unittest.main()
