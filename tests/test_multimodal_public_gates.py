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
                _diagnostic(
                    "clean",
                    {"CATO": 0.55, "TLEO": 0.2, "SPO": 0.15, "LRIO": 0.1},
                    cato_entropy=0.30,
                    grounding_accuracy=0.74,
                    top_alignment_accuracy=0.72,
                    rceo_reliability=0.90,
                ),
                _diagnostic(
                    "no_cato",
                    {"CATO": 0.0, "TLEO": 0.4, "SPO": 0.4, "LRIO": 0.2},
                    cato_entropy=0.90,
                    grounding_accuracy=0.52,
                    top_alignment_accuracy=0.20,
                ),
                _diagnostic(
                    "corrupted_visual",
                    {"CATO": 0.28, "TLEO": 0.30, "SPO": 0.32, "LRIO": 0.10},
                    cato_entropy=0.58,
                    grounding_accuracy=0.61,
                    top_alignment_accuracy=0.56,
                    rceo_reliability=0.55,
                    rceo_corruption_response=0.35,
                ),
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
        self.assertTrue(report["checks"]["cato_top_alignment_accuracy_high"]["passed"])
        self.assertTrue(report["checks"]["grounding_accuracy_improves_with_entropy"]["passed"])
        self.assertTrue(report["checks"]["rceo_visual_stress_router_shift"]["passed"])

    def test_sentiment_gate_requires_lrio_spo_rceo_and_robustness(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_sentiment_gate

        report = evaluate_sentiment_gate(
            statistics_summary=_summary("sentiment_emotion", "test", full=0.76, baseline=0.74),
            diagnostics_rows=[
                {
                    "setting": "clean",
                    "router_load_by_candidate": {"TLEO": 0.1, "SPO": 0.35, "LRIO": 0.40, "CATO": 0.15},
                    "adapter_params": {"LRIO_rank_entropy": 0.6, "SPO_temperature": 1.0},
                    "candidate_diagnostics": {
                        "LRIO": {"rank_entropy": 0.6, "rank_top_k": [0, 1], "pair_interaction_strength": 0.4},
                        "SPO": {
                            "prototype_entropy": 0.5,
                            "top_prototype_by_class": {"negative": 1, "positive": 4},
                        },
                    },
                }
            ],
            ablation_scores={"ovha_no_lrio": 0.70, "ovha_no_spo": 0.71, "ovha_no_rceo": 0.68},
            robustness_summary={
                "full_drop_less_than_baseline": True,
                "rceo_reliability_monotonic": True,
                "rceo_reliability_calibration": {"ece": 0.05, "bin_count": 5},
                "required_stress_coverage": {"passed": True, "reasons": []},
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
        self.assertTrue(report["checks"]["lrio_rank_entropy_present"]["passed"])
        self.assertTrue(report["checks"]["spo_top_prototype_differentiates"]["passed"])
        self.assertTrue(report["checks"]["rceo_reliability_calibrated"]["passed"])

    def test_sentiment_gate_respects_lower_is_better_mae_metrics(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_sentiment_gate

        report = evaluate_sentiment_gate(
            statistics_summary=_summary(
                "sentiment_emotion",
                "test",
                full=0.41,
                baseline=0.48,
                higher_is_better=False,
            ),
            diagnostics_rows=[
                {
                    "setting": "clean",
                    "router_load_by_candidate": {"TLEO": 0.1, "SPO": 0.35, "LRIO": 0.40, "CATO": 0.15},
                    "candidate_diagnostics": {
                        "LRIO": {"rank_entropy": 0.6},
                        "SPO": {
                            "prototype_entropy": 0.5,
                            "top_prototype_by_class": {"negative": 1, "positive": 4},
                        },
                    },
                }
            ],
            ablation_scores={"ovha_no_lrio": 0.52, "ovha_no_spo": 0.50, "ovha_no_rceo": 0.54},
            robustness_summary={
                "full_drop_less_than_baseline": True,
                "rceo_reliability_monotonic": True,
                "rceo_reliability_calibration": {"ece": 0.05, "bin_count": 5},
                "required_stress_coverage": {"passed": True, "reasons": []},
                "required_ablation_degradation": {"passed": True, "reasons": []},
            },
            task="sentiment_emotion",
            split="test",
        )

        self.assertTrue(report["checks"]["full_beats_same_feature_baseline"]["passed"], report["reasons"])
        self.assertTrue(report["checks"]["no_lrio_drops"]["passed"], report["reasons"])
        self.assertGreater(report["checks"]["full_beats_same_feature_baseline"]["value"], 0.0)

    def test_sentiment_gate_rejects_missing_plan_diagnostics_and_calibration(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_sentiment_gate

        report = evaluate_sentiment_gate(
            statistics_summary=_summary("sentiment_emotion", "test", full=0.76, baseline=0.74),
            diagnostics_rows=[
                {
                    "setting": "clean",
                    "router_load_by_candidate": {"TLEO": 0.1, "SPO": 0.35, "LRIO": 0.40, "CATO": 0.15},
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

        self.assertFalse(report["passed"])
        joined = "\n".join(report["reasons"])
        self.assertIn("LRIO rank entropy diagnostic missing", joined)
        self.assertIn("SPO prototype entropy diagnostic missing", joined)
        self.assertIn("SPO top prototype differentiation missing", joined)
        self.assertIn("RCEO reliability calibration missing", joined)

    def test_sentiment_gate_rejects_robustness_without_stress_family_coverage(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_sentiment_gate

        report = evaluate_sentiment_gate(
            statistics_summary=_summary("sentiment_emotion", "test", full=0.76, baseline=0.74),
            diagnostics_rows=[
                {
                    "setting": "clean",
                    "router_load_by_candidate": {"TLEO": 0.1, "SPO": 0.35, "LRIO": 0.40, "CATO": 0.15},
                    "candidate_diagnostics": {
                        "LRIO": {"rank_entropy": 0.6},
                        "SPO": {
                            "prototype_entropy": 0.5,
                            "top_prototype_by_class": {"negative": 1, "positive": 4},
                        },
                    },
                }
            ],
            ablation_scores={"ovha_no_lrio": 0.70, "ovha_no_spo": 0.71, "ovha_no_rceo": 0.68},
            robustness_summary={
                "full_drop_less_than_baseline": True,
                "rceo_reliability_monotonic": True,
                "rceo_reliability_calibration": {"ece": 0.05, "bin_count": 5},
                "required_ablation_degradation": {"passed": True, "reasons": []},
                "operator_load_shift": {"LRIO": -0.20, "SPO": 0.15},
            },
            task="sentiment_emotion",
            split="test",
        )

        self.assertFalse(report["passed"])
        self.assertIn("robustness stress family coverage missing", "\n".join(report["reasons"]))

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

    def test_region_text_gate_rejects_missing_alignment_accuracy_and_visual_stress(self):
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

        self.assertFalse(report["passed"])
        joined = "\n".join(report["reasons"])
        self.assertIn("CATO top alignment accuracy diagnostic missing", joined)
        self.assertIn("grounding accuracy must improve as CATO alignment entropy decreases", joined)
        self.assertIn("RCEO visual stress router shift missing", joined)

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

    def test_public_gate_rejects_non_boolean_metric_direction_summary(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        summary = _summary("phrase_region_grounding", "test", full=0.80, baseline=0.72)
        summary["main_table"]["phrase_region_grounding"]["test"]["ovha_full"]["higher_is_better"] = "true"

        report = evaluate_region_text_gate(
            statistics_summary=summary,
            diagnostics_rows=_passing_region_text_diagnostics(),
            no_cato_score=0.70,
            task="phrase_region_grounding",
            split="test",
        )

        self.assertFalse(report["passed"])
        self.assertIn("ovha_full higher_is_better must be boolean", "\n".join(report["reasons"]))

    def test_public_gate_rejects_mixed_metric_direction_summary(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        summary = _summary("phrase_region_grounding", "test", full=0.80, baseline=0.72)
        summary["main_table"]["phrase_region_grounding"]["test"]["cross_attention_transformer"]["higher_is_better"] = False

        report = evaluate_region_text_gate(
            statistics_summary=summary,
            diagnostics_rows=_passing_region_text_diagnostics(),
            no_cato_score=0.70,
            task="phrase_region_grounding",
            split="test",
        )

        self.assertFalse(report["passed"])
        self.assertIn("statistics summary higher_is_better must be consistent across models", "\n".join(report["reasons"]))

    def test_public_gate_requires_paired_metric_direction_and_mean_delta(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        summary = _summary("phrase_region_grounding", "test", full=0.80, baseline=0.72)
        paired = summary["paired_tests"]["phrase_region_grounding"]["test"]
        paired.pop("metric_direction")
        paired.pop("mean_delta")

        report = evaluate_region_text_gate(
            statistics_summary=summary,
            diagnostics_rows=_passing_region_text_diagnostics(),
            no_cato_score=0.70,
            task="phrase_region_grounding",
            split="test",
        )

        self.assertFalse(report["passed"])
        joined = "\n".join(report["reasons"])
        self.assertIn("paired comparison missing metric_direction", joined)
        self.assertIn("paired comparison missing mean_delta", joined)

    def test_public_gate_rejects_paired_mean_delta_without_positive_improvement(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        summary = _summary("phrase_region_grounding", "test", full=0.80, baseline=0.72)
        summary["paired_tests"]["phrase_region_grounding"]["test"]["mean_delta"] = -0.08

        report = evaluate_region_text_gate(
            statistics_summary=summary,
            diagnostics_rows=_passing_region_text_diagnostics(),
            no_cato_score=0.70,
            task="phrase_region_grounding",
            split="test",
        )

        self.assertFalse(report["passed"])
        self.assertIn(
            "paired comparison mean_delta must be positive for claimed improvement",
            "\n".join(report["reasons"]),
        )

    def test_public_gate_rejects_bootstrap_ci_crossing_zero_for_main_delta(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        summary = _summary("phrase_region_grounding", "test", full=0.80, baseline=0.72)
        summary["paired_tests"]["phrase_region_grounding"]["test"]["paired_bootstrap_ci95"] = [-0.01, 0.12]

        report = evaluate_region_text_gate(
            statistics_summary=summary,
            diagnostics_rows=_passing_region_text_diagnostics(),
            no_cato_score=0.70,
            task="phrase_region_grounding",
            split="test",
        )

        self.assertFalse(report["passed"])
        self.assertIn(
            "paired comparison bootstrap CI must be strictly positive for claimed improvement",
            "\n".join(report["reasons"]),
        )

    def test_public_gate_requires_topconf_reporting_metadata_not_only_mean(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        report = evaluate_region_text_gate(
            statistics_summary=_summary(
                "phrase_region_grounding",
                "test",
                full=0.80,
                baseline=0.72,
                include_reporting_metadata=False,
            ),
            diagnostics_rows=[
                _diagnostic(
                    "clean",
                    {"CATO": 0.55, "TLEO": 0.2, "SPO": 0.15, "LRIO": 0.1},
                    cato_entropy=0.30,
                    grounding_accuracy=0.74,
                    top_alignment_accuracy=0.72,
                    rceo_reliability=0.90,
                ),
                _diagnostic(
                    "no_cato",
                    {"CATO": 0.0, "TLEO": 0.4, "SPO": 0.4, "LRIO": 0.2},
                    cato_entropy=0.90,
                    grounding_accuracy=0.52,
                    top_alignment_accuracy=0.20,
                ),
                _diagnostic(
                    "corrupted_visual",
                    {"CATO": 0.28, "TLEO": 0.30, "SPO": 0.32, "LRIO": 0.10},
                    cato_entropy=0.58,
                    grounding_accuracy=0.61,
                    top_alignment_accuracy=0.56,
                    rceo_reliability=0.55,
                    rceo_corruption_response=0.35,
                ),
            ],
            no_cato_score=0.70,
            task="phrase_region_grounding",
            split="test",
        )

        self.assertFalse(report["passed"])
        joined = "\n".join(report["reasons"])
        self.assertIn("ovha_full main table missing std", joined)
        self.assertIn("cross_attention_transformer main table missing ci95", joined)
        self.assertIn("reporting metadata missing parameter_count", joined)
        self.assertIn("reporting metadata missing per_seed_table", joined)

    def test_region_text_gate_requires_complete_same_feature_baseline_defense_table(self):
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        report = evaluate_region_text_gate(
            statistics_summary=_summary(
                "phrase_region_grounding",
                "test",
                full=0.80,
                baseline=0.72,
                include_required_baselines=False,
            ),
            diagnostics_rows=[
                _diagnostic(
                    "clean",
                    {"CATO": 0.55, "TLEO": 0.2, "SPO": 0.15, "LRIO": 0.1},
                    cato_entropy=0.30,
                    grounding_accuracy=0.74,
                    top_alignment_accuracy=0.72,
                    rceo_reliability=0.90,
                ),
                _diagnostic(
                    "no_cato",
                    {"CATO": 0.0, "TLEO": 0.4, "SPO": 0.4, "LRIO": 0.2},
                    cato_entropy=0.90,
                    grounding_accuracy=0.52,
                    top_alignment_accuracy=0.20,
                ),
                _diagnostic(
                    "corrupted_visual",
                    {"CATO": 0.28, "TLEO": 0.30, "SPO": 0.32, "LRIO": 0.10},
                    cato_entropy=0.58,
                    grounding_accuracy=0.61,
                    top_alignment_accuracy=0.56,
                    rceo_reliability=0.55,
                    rceo_corruption_response=0.35,
                ),
            ],
            no_cato_score=0.70,
            task="phrase_region_grounding",
            split="test",
        )

        self.assertFalse(report["passed"])
        joined = "\n".join(report["reasons"])
        self.assertIn("statistics summary missing required same-feature baseline: text_only", joined)
        self.assertIn("statistics summary missing required same-feature baseline: region_only", joined)
        self.assertIn("statistics summary missing required same-feature baseline: ovha_no_evidence_router", joined)

    def test_public_gate_cli_emits_go_no_go_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats_path = root / "stats.json"
            diagnostics_path = root / "diagnostics.jsonl"
            stats_path.write_text(json.dumps(_summary("phrase_region_grounding", "test", full=0.80, baseline=0.72)))
            diagnostics_path.write_text(
                json.dumps(
                    _diagnostic(
                        "clean",
                        {"CATO": 0.55, "TLEO": 0.2, "SPO": 0.15, "LRIO": 0.1},
                        cato_entropy=0.30,
                        grounding_accuracy=0.74,
                        top_alignment_accuracy=0.72,
                        rceo_reliability=0.90,
                    )
                )
                + "\n"
                + json.dumps(
                    _diagnostic(
                        "no_cato",
                        {"CATO": 0.0, "TLEO": 0.4, "SPO": 0.4, "LRIO": 0.2},
                        cato_entropy=0.90,
                        grounding_accuracy=0.52,
                        top_alignment_accuracy=0.20,
                    )
                )
                + "\n"
                + json.dumps(
                    _diagnostic(
                        "corrupted_visual",
                        {"CATO": 0.28, "TLEO": 0.30, "SPO": 0.32, "LRIO": 0.10},
                        cato_entropy=0.58,
                        grounding_accuracy=0.61,
                        top_alignment_accuracy=0.56,
                        rceo_reliability=0.55,
                        rceo_corruption_response=0.35,
                    )
                )
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
    higher_is_better: bool = True,
    seed_count: int = 3,
    common_seed_count: int = 3,
    include_bootstrap: bool = True,
    include_reporting_metadata: bool = True,
    include_required_baselines: bool = True,
) -> dict[str, object]:
    mean_delta = (full - baseline) if higher_is_better else (baseline - full)
    paired = {
        "common_seed_count": common_seed_count,
        "mean_delta": mean_delta,
        "metric_direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "paired_permutation_p": 0.25,
    }
    if include_bootstrap:
        paired["paired_bootstrap_ci95"] = [0.01, 0.12]
    models = ["ovha_full", "cross_attention_transformer"]
    if include_required_baselines:
        for model in _required_baselines_for_task(task):
            if model not in models:
                models.append(model)
    non_full_score = (
        lambda index: baseline - 0.01 * index
        if higher_is_better
        else baseline + 0.01 * index
    )
    model_scores = {
        model: (full if model == "ovha_full" else non_full_score(index))
        for index, model in enumerate(models)
    }
    model_rows = {
        model: _model_summary_row(score, seed_count, include_reporting_metadata, higher_is_better)
        for model, score in model_scores.items()
    }
    summary: dict[str, object] = {
        "main_table": {
            task: {
                split: model_rows
            }
        },
        "paired_tests": {task: {split: paired}},
    }
    if include_reporting_metadata:
        summary["reporting_metadata"] = {
            "parameter_count": {model: 120000 + index for index, model in enumerate(models)},
            "training_steps": {model: 1000 for model in models},
            "frozen_feature_versions": {"text": "frozen-text-v1", "region": "frozen-region-v1"},
            "hardware": "unit-test-cpu",
            "wall_clock_summary": {model: "10m" for model in models},
            "per_seed_table": [
                {"model": model, "seed": seed, "score": score + (seed - 2) * 0.01}
                for model, score in model_scores.items()
                for seed in range(1, seed_count + 1)
            ],
        }
    return summary


def _model_summary_row(
    score: float,
    seed_count: int,
    include_reporting_metadata: bool,
    higher_is_better: bool,
) -> dict[str, object]:
    row: dict[str, object] = {"mean": score, "seed_count": seed_count, "higher_is_better": higher_is_better}
    if include_reporting_metadata:
        row.update(
            {
                "std": 0.01,
                "ci95": [score - 0.01, score + 0.01],
                "per_seed_scores": [score + (seed - 2) * 0.01 for seed in range(1, seed_count + 1)],
            }
        )
    return row


def _passing_region_text_diagnostics() -> list[dict[str, object]]:
    return [
        _diagnostic(
            "clean",
            {"CATO": 0.55, "TLEO": 0.2, "SPO": 0.15, "LRIO": 0.1},
            cato_entropy=0.30,
            grounding_accuracy=0.74,
            top_alignment_accuracy=0.72,
            rceo_reliability=0.90,
        ),
        _diagnostic(
            "no_cato",
            {"CATO": 0.0, "TLEO": 0.4, "SPO": 0.4, "LRIO": 0.2},
            cato_entropy=0.90,
            grounding_accuracy=0.52,
            top_alignment_accuracy=0.20,
        ),
        _diagnostic(
            "corrupted_visual",
            {"CATO": 0.28, "TLEO": 0.30, "SPO": 0.32, "LRIO": 0.10},
            cato_entropy=0.58,
            grounding_accuracy=0.61,
            top_alignment_accuracy=0.56,
            rceo_reliability=0.55,
            rceo_corruption_response=0.35,
        ),
    ]


def _required_baselines_for_task(task: str) -> tuple[str, ...]:
    if task == "phrase_region_grounding":
        return (
            "text_only",
            "region_only",
            "concat_fusion",
            "cross_attention_transformer",
            "modality_expert_moe",
            "clip_style_region_text_retrieval",
            "cato_only",
            "ovha_no_cato",
            "ovha_no_rceo",
            "ovha_no_evidence_router",
        )
    if task == "sentiment_emotion":
        return (
            "concat_fusion",
            "tfn_lmf",
            "mult_style_crossmodal_transformer",
            "misa_shared_private",
            "modality_expert_moe",
            "quality_aware_fusion",
            "ovha_no_lrio",
            "ovha_no_spo",
            "ovha_no_rceo",
            "ovha_no_evidence_router",
        )
    return ()


def _diagnostic(
    setting: str,
    loads: dict[str, float],
    *,
    cato_entropy: float,
    grounding_accuracy: float | None = None,
    top_alignment_accuracy: float | None = None,
    rceo_reliability: float | None = None,
    rceo_corruption_response: float | None = None,
) -> dict[str, object]:
    cato = {"alignment_entropy": cato_entropy}
    if grounding_accuracy is not None:
        cato["grounding_accuracy"] = grounding_accuracy
    if top_alignment_accuracy is not None:
        cato["top_alignment_accuracy"] = top_alignment_accuracy
    candidate_diagnostics: dict[str, object] = {"CATO": cato}
    if rceo_corruption_response is not None:
        candidate_diagnostics["RCEO"] = {"corruption_response": rceo_corruption_response}
    row: dict[str, object] = {
        "setting": setting,
        "router_load_by_candidate": loads,
        "candidate_diagnostics": candidate_diagnostics,
    }
    if rceo_reliability is not None:
        row["rceo_reliability"] = rceo_reliability
    return row


if __name__ == "__main__":
    unittest.main()
