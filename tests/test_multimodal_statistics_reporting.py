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
        self.assertEqual(summary["reporting_metadata"]["parameter_count"]["text_only"], 1000)
        self.assertEqual(summary["reporting_metadata"]["training_steps"]["ovha_no_cato"], 1000)
        self.assertTrue(summary["reporting_metadata"]["per_seed_table"])

    def test_public_summary_reports_lower_is_better_paired_delta_as_positive_improvement(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        rows = _lower_is_better_sentiment_rows()
        summary = summarize_public_results(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        validation = validate_public_summary(summary)

        self.assertTrue(validation.ok, validation.errors)
        paired = summary["paired_tests"]["sentiment_emotion"]["test"]
        self.assertEqual(paired["metric_direction"], "lower_is_better")
        self.assertAlmostEqual(paired["mean_delta"], 0.07)
        self.assertGreater(paired["paired_bootstrap_ci95"][0], 0.0)

    def test_public_summary_rejects_paired_delta_that_disagrees_with_metric_direction(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        summary = summarize_public_results(
            _lower_is_better_sentiment_rows(),
            full_model="ovha_full",
            baseline_model="cross_attention_transformer",
        )
        paired = summary["paired_tests"]["sentiment_emotion"]["test"]
        paired["mean_delta"] = -0.07
        paired["paired_bootstrap_ci95"] = [-0.08, -0.06]

        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        joined = "\n".join(validation.errors)
        self.assertIn("paired_tests mean_delta disagrees with metric_direction", joined)
        self.assertIn("paired_tests bootstrap CI disagrees with metric_direction", joined)

    def test_public_summary_rejects_common_seed_count_without_appendix_evidence(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        summary = summarize_public_results(
            _lower_is_better_sentiment_rows(),
            full_model="ovha_full",
            baseline_model="cross_attention_transformer",
        )
        summary["per_seed_appendix"] = [
            row
            for row in summary["per_seed_appendix"]
            if not (
                row["task"] == "sentiment_emotion"
                and row["split"] == "test"
                and row["model"] == "cross_attention_transformer"
                and row["seed"] == 13
            )
        ]

        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        self.assertIn(
            "sentiment_emotion/test paired_tests common_seed_count disagrees with per_seed_appendix",
            "\n".join(validation.errors),
        )

    def test_public_summary_rejects_main_table_seed_count_without_appendix_evidence(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        summary = summarize_public_results(
            _metric_rows(),
            full_model="ovha_full",
            baseline_model="cross_attention_transformer",
        )
        summary["per_seed_appendix"] = [
            row
            for row in summary["per_seed_appendix"]
            if not (
                row["task"] == "phrase_region_grounding"
                and row["split"] == "test"
                and row["model"] == "text_only"
                and row["seed"] == 13
            )
        ]

        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        self.assertIn(
            "phrase_region_grounding/test/text_only seed_count disagrees with per_seed_appendix",
            "\n".join(validation.errors),
        )

    def test_public_summary_rejects_report_metadata_per_seed_table_mismatch(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        summary = summarize_public_results(
            _metric_rows(),
            full_model="ovha_full",
            baseline_model="cross_attention_transformer",
        )
        summary["reporting_metadata"]["per_seed_table"] = [
            row
            for row in summary["reporting_metadata"]["per_seed_table"]
            if not (
                row["task"] == "phrase_region_grounding"
                and row["split"] == "test"
                and row["model"] == "text_only"
                and row["seed"] == 13
            )
        ]

        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        self.assertIn(
            "reporting_metadata per_seed_table must match per_seed_appendix",
            "\n".join(validation.errors),
        )

    def test_public_summary_rejects_best_seed_only_and_missing_metadata(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        rows = [_metric_rows()[0]]
        summary = summarize_public_results(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        self.assertIn("at least 3 seeds", "\n".join(validation.errors))
        self.assertIn("paired_tests", "\n".join(validation.errors))

    def test_public_summary_marks_weak_and_pseudo_labels_not_ground_truth(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        rows = _metric_rows()
        for row in rows:
            row["label_provenance"] = {
                "supervision_type": "pseudo",
                "source": "cross_modal_disagreement_v0",
                "must_report_as": "pseudo",
            }
        summary = summarize_public_results(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        validation = validate_public_summary(summary)

        self.assertTrue(validation.ok, validation.errors)
        self.assertEqual(summary["metadata"]["label_provenance"]["pseudo"]["count"], len(rows))
        self.assertIn("cross_modal_disagreement_v0", summary["metadata"]["label_provenance"]["pseudo"]["sources"])

    def test_public_summary_rejects_pseudo_labels_reported_as_ground_truth(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        rows = _metric_rows()
        for row in rows:
            row["pseudo_label_source"] = "cross_modal_disagreement_v0"
            row["label_provenance"] = {
                "supervision_type": "ground_truth",
                "source": "cross_modal_disagreement_v0",
                "must_report_as": "ground_truth",
            }
        summary = summarize_public_results(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        joined = "\n".join(validation.errors)
        self.assertIn("pseudo labels must be reported as pseudo, not ground_truth", joined)

    def test_public_summary_rejects_final_summary_without_raw_metric_paths(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        rows = _metric_rows()
        for row in rows:
            row.pop("raw_metric_path")
        summary = summarize_public_results(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        joined = "\n".join(validation.errors)
        self.assertIn("raw_metric_path", joined)
        self.assertIn("only final summary without raw metrics", joined)

    def test_public_summary_requires_wall_clock_and_parameter_count_per_model(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        rows = _metric_rows()
        for row in rows:
            row["hardware"] = {"accelerator": "A800"}
            if row["model"] == "cross_attention_transformer":
                row.pop("parameter_count")
        summary = summarize_public_results(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        joined = "\n".join(validation.errors)
        self.assertIn("metadata.hardware missing wall_clock_hours", joined)
        self.assertIn("parameter_count missing for model: cross_attention_transformer", joined)

    def test_public_summary_rejects_main_delta_against_weak_baseline_only(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        rows = _metric_rows()
        for row in rows:
            if row["model"] == "cross_attention_transformer":
                row["baseline_strength"] = "weak"
        summary = summarize_public_results(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        joined = "\n".join(validation.errors)
        self.assertIn("cannot report only improvement over weak baseline", joined)
        self.assertIn("cross_attention_transformer", joined)

    def test_public_summary_rejects_incomplete_same_feature_baseline_table(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        rows = _metric_rows_for_models(("ovha_full", "cross_attention_transformer"))
        summary = summarize_public_results(rows, full_model="ovha_full", baseline_model="cross_attention_transformer")
        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        joined = "\n".join(validation.errors)
        self.assertIn("statistics summary missing required same-feature baseline: text_only", joined)
        self.assertIn("statistics summary missing required same-feature baseline: region_only", joined)
        self.assertIn("statistics summary missing required same-feature baseline: ovha_no_evidence_router", joined)

    def test_public_summary_rejects_missing_report_facing_metadata(self):
        from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary

        summary = summarize_public_results(
            _metric_rows(),
            full_model="ovha_full",
            baseline_model="cross_attention_transformer",
        )
        summary.pop("reporting_metadata")

        validation = validate_public_summary(summary)

        self.assertFalse(validation.ok)
        joined = "\n".join(validation.errors)
        self.assertIn("reporting_metadata missing parameter_count", joined)
        self.assertIn("reporting_metadata missing training_steps", joined)
        self.assertIn("reporting_metadata missing per_seed_table", joined)

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
    from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task

    return _metric_rows_for_models(("ovha_full", *baseline_names_for_task("phrase_region_grounding")))


def _metric_rows_for_models(models):
    rows = []
    scores_by_model = {
        "ovha_full": (0.70, 0.75, 0.80),
        "cross_attention_transformer": (0.65, 0.70, 0.72),
        "text_only": (0.58, 0.59, 0.60),
        "region_only": (0.56, 0.57, 0.59),
        "concat_fusion": (0.61, 0.63, 0.64),
        "modality_expert_moe": (0.62, 0.64, 0.65),
        "clip_style_region_text_retrieval": (0.60, 0.62, 0.63),
        "cato_only": (0.63, 0.66, 0.67),
        "ovha_no_cato": (0.60, 0.62, 0.64),
        "ovha_no_rceo": (0.64, 0.66, 0.68),
        "ovha_no_evidence_router": (0.61, 0.62, 0.63),
    }
    for seed, full_score, baseline_score in ((11, 0.70, 0.65), (12, 0.75, 0.70), (13, 0.80, 0.72)):
        seed_index = (11, 12, 13).index(seed)
        for model in models:
            if model == "ovha_full":
                score = full_score
            elif model == "cross_attention_transformer":
                score = baseline_score
            else:
                score = scores_by_model[model][seed_index]
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


def _lower_is_better_sentiment_rows():
    from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task

    scores_by_model = {
        "ovha_full": (0.41, 0.40, 0.42),
        "cross_attention_transformer": (0.48, 0.47, 0.49),
        "concat_fusion": (0.52, 0.51, 0.53),
        "tfn_lmf": (0.50, 0.49, 0.51),
        "mult_style_crossmodal_transformer": (0.49, 0.48, 0.50),
        "misa_shared_private": (0.51, 0.50, 0.52),
        "modality_expert_moe": (0.53, 0.52, 0.54),
        "quality_aware_fusion": (0.47, 0.46, 0.48),
        "ovha_no_lrio": (0.54, 0.53, 0.55),
        "ovha_no_spo": (0.50, 0.49, 0.51),
        "ovha_no_rceo": (0.55, 0.54, 0.56),
        "ovha_no_evidence_router": (0.56, 0.55, 0.57),
    }
    rows = []
    models = ("ovha_full", "cross_attention_transformer", *baseline_names_for_task("sentiment_emotion"))
    for seed_index, seed in enumerate((11, 12, 13)):
        for model in dict.fromkeys(models):
            rows.append(
                {
                    "task": "sentiment_emotion",
                    "dataset": "cmu_mosei",
                    "split": "test",
                    "model": model,
                    "seed": seed,
                    "score": scores_by_model[model][seed_index],
                    "higher_is_better": False,
                    "parameter_count": 1234 if model == "ovha_full" else 1000,
                    "training_steps": 1000,
                    "frozen_feature_extractor_version": {
                        "text": "frozen-text-test",
                        "audio": "frozen-audio-test",
                        "vision": "frozen-visual-test",
                    },
                    "hardware": {"accelerator": "A800", "wall_clock_hours": 1.5},
                    "raw_metric_path": f"outputs/mock/{model}/seed_{seed}.jsonl",
                }
            )
    return rows


if __name__ == "__main__":
    unittest.main()
