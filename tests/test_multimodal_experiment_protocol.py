import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MultimodalExperimentProtocolTests(unittest.TestCase):
    def test_required_multimodal_configs_and_scripts_exist(self):
        expected = [
            ROOT / "configs" / "multimodal_controlled_v1_smoke.json",
            ROOT / "configs" / "multimodal_refcoco_public_smoke.json",
            ROOT / "configs" / "multimodal_cmu_mosei_public_smoke.json",
            ROOT / "configs" / "multimodal_robustness_smoke.json",
            ROOT / "moat_ovha_torch" / "config_multimodal.py",
            ROOT / "moat_ovha_torch" / "models" / "multimodal" / "baselines.py",
            ROOT / "moat_ovha_torch" / "eval" / "multimodal_diagnostics.py",
            ROOT / "scripts" / "multimodal" / "run_public_smoke.py",
            ROOT / "scripts" / "multimodal" / "summarize_diagnostics.py",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)

    def test_config_parser_enforces_multiseed_and_training_stages(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_controlled_v1_smoke.json")

        self.assertGreaterEqual(len(config.seeds), 3)
        self.assertEqual(config.training_stages, ("T0", "T1", "T2", "T3", "T4"))
        self.assertIn("TLEO", config.candidate_names)
        self.assertIn("CATO", config.candidate_names)
        self.assertTrue(config.enforce_same_features_for_baselines)
        self.assertTrue(config.fail_on_missing_cache_artifact)

        with self.assertRaisesRegex(ValueError, "at least 3 seeds"):
            MultimodalExperimentConfig.from_mapping({"name": "bad", "seeds": [1], "dataset_name": "controlled_multimodal"})

    def test_same_feature_baseline_registry_matches_plan(self):
        from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task, external_reference_names_for_task

        region = set(baseline_names_for_task("phrase_region_grounding"))
        self.assertTrue(
            {
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
            }.issubset(region)
        )

        sentiment = set(baseline_names_for_task("sentiment_emotion"))
        self.assertTrue(
            {
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
            }.issubset(sentiment)
        )
        self.assertEqual(
            set(external_reference_names_for_task("phrase_region_grounding")),
            {"MDETR", "GLIP", "GroundingDINO"},
        )

    def test_config_parser_requires_complete_same_feature_baseline_set(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        valid = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_refcoco_public_smoke.json")
        self.assertIn("cross_attention_transformer", valid.baseline_names)
        self.assertIn("ovha_no_cato", valid.baseline_names)

        invalid = {
            "name": "bad_refcoco",
            "dataset_name": "refcoco",
            "task_type": "phrase_region_grounding",
            "seeds": [1, 2, 3],
            "training_stages": ["T0", "T5"],
            "candidate_names": ["TLEO", "SPO", "LRIO", "CATO"],
            "baseline_names": ["text_only", "cross_attention_transformer"],
            "eval_episode_count": 16,
        }
        with self.assertRaisesRegex(ValueError, "missing required same-feature baselines"):
            MultimodalExperimentConfig.from_mapping(invalid)

    def test_config_parser_rejects_wrong_stage_sequence_for_task(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task

        invalid = {
            "name": "bad_refcoco_stages",
            "dataset_name": "refcoco",
            "task_type": "phrase_region_grounding",
            "seeds": [1, 2, 3],
            "training_stages": ["T0", "T1", "T5"],
            "candidate_names": ["TLEO", "SPO", "LRIO", "CATO"],
            "baseline_names": list(baseline_names_for_task("phrase_region_grounding")),
            "eval_episode_count": 16,
        }

        with self.assertRaisesRegex(ValueError, "training_stages for phrase_region_grounding must be"):
            MultimodalExperimentConfig.from_mapping(invalid)

    def test_config_parser_rejects_external_references_as_same_feature_baselines(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task

        invalid = {
            "name": "bad_external",
            "dataset_name": "refcoco",
            "task_type": "phrase_region_grounding",
            "seeds": [1, 2, 3],
            "training_stages": ["T0", "T5"],
            "candidate_names": ["TLEO", "SPO", "LRIO", "CATO"],
            "baseline_names": list(baseline_names_for_task("phrase_region_grounding")) + ["GroundingDINO"],
            "eval_episode_count": 16,
        }
        with self.assertRaisesRegex(ValueError, "external references must not be listed as same-feature baselines"):
            MultimodalExperimentConfig.from_mapping(invalid)

    def test_public_smoke_runner_fails_fast_on_missing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                "--cache-root",
                str(Path(tmp) / "missing-cache"),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("data_card.json", "\n".join(payload["errors"]))
        self.assertIn("fail-fast", payload["policy"])

    def test_public_entry_requires_controlled_go_no_go_report(self):
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        failed_controlled = {
            "go_no_go": {"controlled_multimodal_passed": False},
            "gate_table": {},
        }
        region_report = validate_public_entry_requirements("phrase_region_grounding", failed_controlled)

        self.assertFalse(region_report.ok)
        self.assertIn("controlled_multimodal_passed", "\n".join(region_report.errors))

        missing_cato = {
            "go_no_go": {"controlled_multimodal_passed": True},
            "gate_table": {
                "Stackability": {"passed": True},
                "CATO collapse": {"passed": False},
            },
        }
        region_report = validate_public_entry_requirements("phrase_region_grounding", missing_cato)

        self.assertFalse(region_report.ok)
        self.assertIn("region-text public entry requires CATO collapse", "\n".join(region_report.errors))

        missing_sentiment_gates = {
            "go_no_go": {"controlled_multimodal_passed": True},
            "gate_table": {
                "LRIO collapse": {"passed": True},
                "SPO collapse": {"passed": False},
                "RCEO gate": {"passed": True},
            },
        }
        sentiment_report = validate_public_entry_requirements("sentiment_emotion", missing_sentiment_gates)

        self.assertFalse(sentiment_report.ok)
        self.assertIn("sentiment/emotion public entry requires SPO collapse", "\n".join(sentiment_report.errors))

    def test_diagnostics_schema_requires_plan_keys(self):
        from moat_ovha_torch.eval.multimodal_diagnostics import required_diagnostic_keys, validate_diagnostic_row

        required = required_diagnostic_keys()
        for key in (
            "router_entropy",
            "router_load_by_candidate",
            "router_logit_parts",
            "candidate_loss",
            "adapter_params",
            "memory_slot_norm",
            "stackability_passed",
        ):
            self.assertIn(key, required)

        report = validate_diagnostic_row({"router_entropy": 1.0})
        self.assertFalse(report.ok)
        self.assertIn("router_load_by_candidate", "\n".join(report.errors))


if __name__ == "__main__":
    unittest.main()
