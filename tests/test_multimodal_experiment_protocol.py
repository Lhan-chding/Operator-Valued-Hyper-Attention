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

    def test_config_parser_validates_embedded_training_loss_protocol(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_refcoco_public_smoke.json")

        self.assertIn("T5", config.losses_by_stage)
        self.assertIn("public_alignment_ce", config.losses_by_stage["T5"])
        self.assertIn("CATO", config.adapter_params_by_candidate)

        base_public = json.loads((ROOT / "configs" / "multimodal_refcoco_public_smoke.json").read_text())
        missing_losses = {key: value for key, value in base_public.items() if key != "losses_by_stage"}
        with self.assertRaisesRegex(ValueError, "losses_by_stage must be explicit"):
            MultimodalExperimentConfig.from_mapping(missing_losses)

        missing_adapter_params = {
            key: value for key, value in base_public.items() if key != "adapter_params_by_candidate"
        }
        with self.assertRaisesRegex(ValueError, "adapter_params_by_candidate must be explicit"):
            MultimodalExperimentConfig.from_mapping(missing_adapter_params)

        invalid = {
            "name": "bad_public_loss",
            "dataset_name": "refcoco",
            "task_type": "phrase_region_grounding",
            "seeds": [1, 2, 3],
            "training_stages": ["T0", "T5"],
            "candidate_names": ["TLEO", "SPO", "LRIO", "CATO"],
            "baseline_names": [
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
            ],
            "eval_episode_count": 16,
            "losses_by_stage": {
                "T0": ["cache_validation"],
                "T5": ["task_loss", "candidate_individual_loss", "true_alignment_ce"],
            },
            "adapter_params_by_candidate": _valid_adapter_params(),
        }
        with self.assertRaisesRegex(ValueError, "hidden loss is controlled-only"):
            MultimodalExperimentConfig.from_mapping(invalid)

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

    def test_controlled_baselines_include_router_decomposition_ablations(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task

        required = {
            "no_evidence_router",
            "no_reliability_prior",
            "memory_only_router",
            "evidence_only_router",
        }
        registry = set(baseline_names_for_task("controlled_multimodal"))
        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_controlled_v1_smoke.json")

        self.assertTrue(required.issubset(registry), sorted(required - registry))
        self.assertTrue(required.issubset(set(config.baseline_names)), sorted(required - set(config.baseline_names)))

    def test_robustness_config_requires_complete_step6_stress_family_coverage(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_robustness_smoke.json")

        self.assertIn("missing_audio", config.robustness_corruptions)
        self.assertIn("audio_noise", config.robustness_corruptions)

        invalid = {
            "name": "bad_robustness",
            "dataset_name": "refcoco",
            "task_type": "phrase_region_grounding",
            "seeds": [1, 2, 3],
            "training_stages": ["T0", "T6"],
            "candidate_names": ["TLEO", "SPO", "LRIO", "CATO"],
            "baseline_names": [
                "concat_fusion",
                "cross_attention_transformer",
                "modality_expert_moe",
                "quality_aware_fusion",
                "ovha_no_rceo",
                "ovha_no_evidence_router",
            ],
            "eval_episode_count": 16,
            "robustness_corruptions": ["missing_text", "missing_vision", "image_blur"],
            "losses_by_stage": {"T0": ["cache_validation"], "T6": ["robustness_evaluation_only"]},
            "adapter_params_by_candidate": _valid_adapter_params(),
        }
        with self.assertRaisesRegex(ValueError, "missing required robustness stress families"):
            MultimodalExperimentConfig.from_mapping(invalid)

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
            "losses_by_stage": {
                "T0": ["cache_validation"],
                "T5": ["task_loss", "public_alignment_ce", "candidate_individual_loss"],
            },
            "adapter_params_by_candidate": _valid_adapter_params(),
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
            "losses_by_stage": {
                "T0": ["cache_validation"],
                "T5": ["task_loss", "public_alignment_ce", "candidate_individual_loss"],
            },
            "adapter_params_by_candidate": _valid_adapter_params(),
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
            "losses_by_stage": {
                "T0": ["cache_validation"],
                "T5": ["task_loss", "public_alignment_ce", "candidate_individual_loss"],
            },
            "adapter_params_by_candidate": _valid_adapter_params(),
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

        missing_cato_diagnostics = {
            "go_no_go": {"controlled_multimodal_passed": True},
            "gate_table": {
                "Stackability": {"passed": True},
                "CATO collapse": {"passed": True},
            },
        }
        region_report = validate_public_entry_requirements("phrase_region_grounding", missing_cato_diagnostics)

        self.assertFalse(region_report.ok)
        self.assertIn("region-text public entry requires CATO alignment diagnostics", "\n".join(region_report.errors))

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

    def test_sentiment_public_entry_requires_lrio_and_rceo_ablation_degradation(self):
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        missing_ablation_gates = {
            "go_no_go": {"controlled_multimodal_passed": True},
            "gate_table": {
                "LRIO collapse": {"passed": True},
                "SPO collapse": {"passed": True},
                "RCEO gate": {"passed": True},
            },
        }

        blocked = validate_public_entry_requirements("sentiment_emotion", missing_ablation_gates)

        self.assertFalse(blocked.ok)
        joined = "\n".join(blocked.errors)
        self.assertIn("sentiment/emotion public entry requires no-LRIO ablation degradation", joined)
        self.assertIn("sentiment/emotion public entry requires no-RCEO ablation degradation", joined)

        passed_ablation_gates = _complete_controlled_public_entry_report()

        allowed = validate_public_entry_requirements("sentiment_emotion", passed_ablation_gates)

        self.assertTrue(allowed.ok, allowed.errors)

    def test_public_entry_rejects_oracle_smoke_controlled_payload(self):
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        oracle_smoke_payload = {
            "mode": "oracle_smoke_only",
            "controlled_report": {
                "go_no_go": {"controlled_multimodal_passed": True},
                "gate_table": {
                    "Stackability": {"passed": True},
                    "CATO collapse": {"passed": True},
                    "CATO alignment diagnostics": {"passed": True},
                },
            },
        }

        report = validate_public_entry_requirements("phrase_region_grounding", oracle_smoke_payload)

        self.assertFalse(report.ok)
        self.assertIn("oracle_smoke_only is not valid public-entry evidence", "\n".join(report.errors))

        trained_controlled_payload = {
            "mode": "trained_controlled_report",
            "controlled_report": _complete_controlled_public_entry_report(),
        }

        report = validate_public_entry_requirements("phrase_region_grounding", trained_controlled_payload)

        self.assertTrue(report.ok, report.errors)

    def test_public_entry_rejects_minimal_controlled_gate_stub_without_full_report(self):
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        minimal_stub = {
            "go_no_go": {"controlled_multimodal_passed": True, "enter_public_multimodal": True},
            "gate_table": {
                "Stackability": {"passed": True},
                "CATO collapse": {"passed": True},
                "CATO alignment diagnostics": {"passed": True},
            },
        }

        report = validate_public_entry_requirements("phrase_region_grounding", minimal_stub)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("controlled report must include full oracle_matrix_cells", joined)
        self.assertIn("controlled report missing controlled family: mixed_relation_operator", joined)
        self.assertIn("controlled report missing required gate: Router gate", joined)

    def test_public_entry_rejects_controlled_report_with_unknown_family(self):
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        report_payload = _complete_controlled_public_entry_report()
        report_payload["families"] = {
            **report_payload["families"],
            "unplanned_relation_operator": {},
        }

        report = validate_public_entry_requirements("phrase_region_grounding", report_payload)

        self.assertFalse(report.ok)
        self.assertIn(
            "controlled report contains unknown controlled family: unplanned_relation_operator",
            "\n".join(report.errors),
        )

    def test_public_entry_rejects_family_rows_without_oracle_evidence(self):
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        report_payload = _complete_controlled_public_entry_report()
        report_payload["families"]["tleo_local_evidence"] = {}

        report = validate_public_entry_requirements("phrase_region_grounding", report_payload)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("controlled report family tleo_local_evidence missing oracle_matrix", joined)
        self.assertIn("controlled report family tleo_local_evidence missing oracle gap evidence: TLEO_oracle_gap", joined)

    def test_public_entry_rejects_contradictory_go_no_go_reasons(self):
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        report_payload = _complete_controlled_public_entry_report()
        report_payload["go_no_go"] = {
            "controlled_multimodal_passed": True,
            "enter_public_multimodal": True,
            "reasons": ["gate failed: Router gate"],
        }

        report = validate_public_entry_requirements("phrase_region_grounding", report_payload)

        self.assertFalse(report.ok)
        self.assertIn(
            "controlled report go_no_go.reasons must be empty when public entry flags are true",
            "\n".join(report.errors),
        )

    def test_public_entry_rejects_controlled_report_without_artifact_provenance(self):
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        report_payload = _complete_controlled_public_entry_report()
        report_payload.pop("evidence_artifacts", None)

        report = validate_public_entry_requirements("phrase_region_grounding", report_payload)

        self.assertFalse(report.ok)
        self.assertIn("controlled report evidence_artifacts is required", "\n".join(report.errors))

    def test_public_smoke_runner_requires_controlled_report_after_cache_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                "--cache-root",
                str(cache_root),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("controlled go/no-go report is required", "\n".join(payload["errors"]))

    def test_topconf_main_entry_requires_validated_data_caches(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(),
                region_gate_report=_passing_region_text_public_gate_report(),
                sentiment_gate_report=_passing_sentiment_public_gate_report(),
                cache_targets={
                    "refcoco": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "refcoco", "v0.1"),
                        splits=("val", "test"),
                    ),
                    "cmu_mosei": CacheValidationTarget(
                        layout=MultimodalCacheLayout(tmp_path / "missing-cache", "cmu_mosei", "v0.1"),
                        splits=("val", "test"),
                    ),
                },
            )

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("data cache validation failed for cmu_mosei", joined)
        self.assertIn("missing required cache artifact: data_card.json", joined)

    def test_topconf_main_entry_accepts_controlled_public_gates_and_valid_cache(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(),
                region_gate_report=_passing_region_text_public_gate_report(),
                sentiment_gate_report=_passing_sentiment_public_gate_report(),
                cache_targets={
                    "refcoco": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "refcoco", "v0.1"),
                        splits=("val", "test"),
                    ),
                    "cmu_mosei": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1"),
                        splits=("val", "test"),
                    ),
                },
            )

        self.assertTrue(report.ok, report.errors)

    def test_topconf_main_entry_requires_region_and_sentiment_cache_coverage(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            _write_valid_refcoco_public_cache(cache_root)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(),
                region_gate_report=_passing_region_text_public_gate_report(),
                sentiment_gate_report=_passing_sentiment_public_gate_report(),
                cache_targets={
                    "refcoco": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "refcoco", "v0.1"),
                        splits=("val", "test"),
                    ),
                },
            )

        self.assertFalse(report.ok)
        self.assertIn(
            "top-conference main experiments require at least one sentiment/emotion data cache",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_rejects_wrong_or_failed_public_gate_reports(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            region_report = {
                **_passing_region_text_public_gate_report(),
                "name": "sentiment_emotion_public",
                "checks": {
                    **_passing_region_text_public_gate_report()["checks"],
                    "full_beats_required_strong_baselines": {"passed": False},
                },
            }

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(),
                region_gate_report=region_report,
                sentiment_gate_report=_passing_sentiment_public_gate_report(),
                cache_targets={
                    "refcoco": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "refcoco", "v0.1"),
                        splits=("val", "test"),
                    ),
                    "cmu_mosei": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1"),
                        splits=("val", "test"),
                    ),
                },
            )

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("region_text_public gate report name mismatch", joined)
        self.assertIn("region_text_public gate contains failed check: full_beats_required_strong_baselines", joined)

    def test_topconf_main_entry_rejects_truncated_public_gate_reports(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            region_report = _passing_region_text_public_gate_report()
            sentiment_report = _passing_sentiment_public_gate_report()
            region_report["checks"].pop("cato_top_alignment_accuracy_high")
            sentiment_report["checks"].pop("no_rceo_drops")

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(),
                region_gate_report=region_report,
                sentiment_gate_report=sentiment_report,
                cache_targets={
                    "refcoco": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "refcoco", "v0.1"),
                        splits=("val", "test"),
                    ),
                    "cmu_mosei": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1"),
                        splits=("val", "test"),
                    ),
                },
            )

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("region_text_public required check did not pass: cato_top_alignment_accuracy_high", joined)
        self.assertIn("sentiment_emotion_public required check did not pass: no_rceo_drops", joined)

    def test_topconf_main_entry_rejects_contradictory_public_gate_reasons(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            region_report = {**_passing_region_text_public_gate_report(), "reasons": "hidden failure"}
            sentiment_report = _passing_sentiment_public_gate_report()
            sentiment_report["checks"]["robustness_passes"] = {
                "passed": True,
                "reason": "robustness summary failed",
            }

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(),
                region_gate_report=region_report,
                sentiment_gate_report=sentiment_report,
                cache_targets={
                    "refcoco": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "refcoco", "v0.1"),
                        splits=("val", "test"),
                    ),
                    "cmu_mosei": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1"),
                        splits=("val", "test"),
                    ),
                },
            )

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("region_text_public gate reasons must be an empty list when passed is true", joined)
        self.assertIn(
            "sentiment_emotion_public gate check robustness_passes reason must be empty when passed is true",
            joined,
        )

    def test_topconf_main_entry_rejects_public_gates_without_artifact_provenance(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            region_report = _passing_region_text_public_gate_report()
            region_report.pop("evidence_artifacts", None)
            sentiment_report = _passing_sentiment_public_gate_report()
            sentiment_report["evidence_artifacts"] = {
                "task": "sentiment_emotion",
                "split": "test",
                "generated_by": "scripts/multimodal/evaluate_public_gates.py",
                "statistics_summary": {"path": "artifacts/sentiment_stats.json", "sha256": "a" * 64},
                "robustness_summary": {"path": "artifacts/sentiment_robustness.json", "sha256": "b" * 64},
                "raw_metrics": [],
            }

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(),
                region_gate_report=region_report,
                sentiment_gate_report=sentiment_report,
                cache_targets={
                    "refcoco": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "refcoco", "v0.1"),
                        splits=("val", "test"),
                    ),
                    "cmu_mosei": CacheValidationTarget(
                        layout=MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1"),
                        splits=("val", "test"),
                    ),
                },
            )

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("region_text_public gate evidence_artifacts is required", joined)
        self.assertIn("sentiment_emotion_public gate evidence_artifacts missing artifact: diagnostics", joined)
        self.assertIn("sentiment_emotion_public gate evidence_artifacts raw_metrics must be a non-empty list", joined)

    def test_diagnostics_schema_requires_plan_keys(self):
        from moat_ovha_torch.eval.multimodal_diagnostics import required_diagnostic_keys, validate_diagnostic_row

        required = required_diagnostic_keys()
        for key in (
            "router_entropy",
            "router_load_by_candidate",
            "router_memory_logit_norm",
            "router_evidence_logit_norm",
            "router_reliability_logit_norm",
            "router_logit_parts",
            "candidate_loss",
            "adapter_params",
            "memory_slot_norm",
            "candidate_diagnostics",
            "stackability_passed",
        ):
            self.assertIn(key, required)

        report = validate_diagnostic_row({"router_entropy": 1.0})
        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("router_load_by_candidate", joined)
        self.assertIn("missing diagnostic key: router_memory_logit_norm", joined)
        self.assertIn("missing diagnostic key: router_evidence_logit_norm", joined)
        self.assertIn("missing diagnostic key: router_reliability_logit_norm", joined)

    def test_diagnostics_schema_rejects_missing_nested_plan_fields(self):
        from moat_ovha_torch.eval.multimodal_diagnostics import validate_diagnostic_row

        row = {
            "router_entropy": 1.0,
            "router_load_by_candidate": {"TLEO": 0.25, "SPO": 0.25, "LRIO": 0.25, "CATO": 0.25},
            "router_logit_parts": {"memory": 0.1, "evidence": 0.2},
            "candidate_loss": {"TLEO": 0.1, "SPO": 0.2, "LRIO": 0.3},
            "adapter_params": {
                "TLEO_lengthscale": 0.5,
                "SPO_temperature": 1.0,
                "CATO_alignment_temperature": 0.7,
            },
            "memory_slot_norm": {"SPO": 1.0, "LRIO": 1.0, "CATO": 1.0},
            "stackability_passed": True,
        }

        report = validate_diagnostic_row(row)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("router_logit_parts missing reliability", joined)
        self.assertIn("candidate_loss missing CATO", joined)
        self.assertIn("adapter_params missing LRIO_rank_entropy", joined)
        self.assertIn("memory_slot_norm missing TLEO", joined)

    def test_diagnostics_schema_rejects_missing_candidate_specific_plan_fields(self):
        from moat_ovha_torch.eval.multimodal_diagnostics import validate_diagnostic_row

        row = {
            "router_entropy": 1.0,
            "router_memory_logit_norm": 0.1,
            "router_evidence_logit_norm": 0.2,
            "router_reliability_logit_norm": 0.3,
            "router_load_by_candidate": {"TLEO": 0.25, "SPO": 0.25, "LRIO": 0.25, "CATO": 0.25},
            "router_logit_parts": {"memory": 0.1, "evidence": 0.2, "reliability": 0.3},
            "candidate_loss": {"TLEO": 0.1, "SPO": 0.2, "LRIO": 0.3, "CATO": 0.4},
            "adapter_params": {
                "TLEO_lengthscale": 0.5,
                "SPO_temperature": 1.0,
                "LRIO_rank_entropy": 0.6,
                "CATO_alignment_temperature": 0.7,
            },
            "memory_slot_norm": {"TLEO": 1.0, "SPO": 1.0, "LRIO": 1.0, "CATO": 1.0},
            "stackability_passed": True,
            "candidate_diagnostics": {
                "TLEO": {"lengthscale": 0.5, "local_entropy": 0.2, "candidate_loss": 0.1},
                "SPO": {"prototype_entropy": 0.4, "prototype_temperature": 1.0, "candidate_loss": 0.2},
                "LRIO": {"rank_entropy": 0.6, "candidate_loss": 0.3},
                "CATO": {"alignment_entropy": 0.3, "candidate_loss": 0.4},
                "RCEO": {"modality_reliability": 0.9, "reliability_bias_norm": 0.1, "corruption_response": 0.2},
            },
        }

        report = validate_diagnostic_row(row)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("candidate_diagnostics.TLEO missing local_window_size", joined)
        self.assertIn("candidate_diagnostics.SPO missing top_prototype", joined)
        self.assertIn("candidate_diagnostics.LRIO missing rank_top_k", joined)
        self.assertIn("candidate_diagnostics.LRIO missing pair_interaction_strength", joined)
        self.assertIn("candidate_diagnostics.CATO missing top_k_alignment", joined)
        self.assertIn("candidate_diagnostics.CATO missing transport_marginal_error", joined)

        row["candidate_diagnostics"]["TLEO"]["local_window_size"] = 5
        row["candidate_diagnostics"]["SPO"]["top_prototype"] = 2
        row["candidate_diagnostics"]["LRIO"]["rank_top_k"] = [0, 1]
        row["candidate_diagnostics"]["LRIO"]["pair_interaction_strength"] = 0.8
        row["candidate_diagnostics"]["CATO"]["top_k_alignment"] = [0, 2]
        row["candidate_diagnostics"]["CATO"]["transport_marginal_error"] = 0.03

        complete_report = validate_diagnostic_row(row)

        self.assertTrue(complete_report.ok, complete_report.errors)

    def test_diagnostics_schema_rejects_non_finite_and_non_probability_values(self):
        from moat_ovha_torch.eval.multimodal_diagnostics import validate_diagnostic_row

        row = _complete_diagnostic_row()
        row["router_entropy"] = "nan"
        row["router_memory_logit_norm"] = -0.1
        row["router_evidence_logit_norm"] = "inf"
        row["router_reliability_logit_norm"] = "bad"
        row["router_load_by_candidate"] = {"TLEO": -0.1, "SPO": 0.4, "LRIO": 0.4, "CATO": 0.4}
        row["router_logit_parts"]["reliability"] = "inf"
        row["candidate_loss"]["LRIO"] = "nan"
        row["adapter_params"]["CATO_alignment_temperature"] = "bad"
        row["memory_slot_norm"]["SPO"] = -1.0
        row["candidate_diagnostics"]["RCEO"]["modality_reliability"] = 1.4
        row["stackability_passed"] = "true"

        report = validate_diagnostic_row(row)

        self.assertFalse(report.ok)
        joined = "\n".join(report.errors)
        self.assertIn("router_entropy must be finite", joined)
        self.assertIn("router_memory_logit_norm must be finite non-negative", joined)
        self.assertIn("router_evidence_logit_norm must be finite non-negative", joined)
        self.assertIn("router_reliability_logit_norm must be finite non-negative", joined)
        self.assertIn("router_load_by_candidate.TLEO must be a finite probability", joined)
        self.assertIn("router_load_by_candidate values must sum to 1", joined)
        self.assertIn("router_logit_parts.reliability must be finite", joined)
        self.assertIn("candidate_loss.LRIO must be finite non-negative", joined)
        self.assertIn("adapter_params.CATO_alignment_temperature must be finite", joined)
        self.assertIn("memory_slot_norm.SPO must be finite non-negative", joined)
        self.assertIn("candidate_diagnostics.RCEO.modality_reliability must be a finite probability", joined)
        self.assertIn("stackability_passed must be true", joined)

def _write_valid_refcoco_public_cache(cache_root: Path) -> None:
    from moat_ovha_torch.data.multimodal.cache_schema import (
        MultimodalCacheLayout,
        default_data_card,
        file_sha256,
        required_cache_files,
    )

    layout = MultimodalCacheLayout(cache_root, "refcoco", "v0.1")
    root = layout.root
    for folder in ("masks", "positions", "provenance", "supervision", "token_fields"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "data_card.json").write_text(
        json.dumps(default_data_card("refcoco", "v0.1", ["text", "region"], ["phrase_region_grounding"]), sort_keys=True) + "\n"
    )
    (root / "splits.json").write_text(json.dumps({"val": ["val-source"], "test": ["test-source"]}, sort_keys=True) + "\n")
    (root / "samples.parquet").write_text("placeholder samples\n")
    (root / "provenance" / "feature_versions.json").write_text(
        json.dumps(
            {
                "text": "clip-text-test",
                "region": "clip-region-test",
                "baselines": {
                    "cross_attention_transformer": {"text": "clip-text-test", "region": "clip-region-test"},
                    "ovha_full": {"text": "clip-text-test", "region": "clip-region-test"},
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    (root / "provenance" / "pseudo_label_versions.json").write_text(
        json.dumps({"generated_from_splits": ["train"], "version": "test-pseudo-v1"}, sort_keys=True) + "\n"
    )
    for split in ("val", "test"):
        (root / "provenance" / f"source_ids_{split}.txt").write_text(f"{split}-source\n")
        (root / "provenance" / f"sample_records_{split}.jsonl").write_text(
            json.dumps(
                {
                    "source_id": f"{split}-source",
                    "split": split,
                    "original_split": split,
                    "raw_ref": f"raw://{split}-source",
                    "license_tag": "test-license",
                    "preprocessing_version": "preprocess-v1",
                    "image_id": f"image-{split}-source",
                    "caption_id": f"caption-{split}-source",
                    "phrase_span": {"start": 0, "end": 2},
                    "region_box": [0.0, 0.0, 1.0, 1.0],
                    "candidate_region_source": "annotated_boxes",
                    "box_coordinate_convention": "xyxy_normalized",
                },
                sort_keys=True,
            )
            + "\n"
        )
        (root / "provenance" / f"failed_samples_{split}.jsonl").write_text("")
        (root / "supervision" / f"task_labels_{split}.npy").write_text("placeholder labels\n")
        (root / "supervision" / f"alignment_pairs_{split}.parquet").write_text("placeholder alignment pairs\n")
        (root / "supervision" / f"bbox_targets_{split}.npy").write_text("placeholder boxes\n")
        (root / "supervision" / f"region_targets_{split}.npy").write_text("placeholder regions\n")
        (root / "supervision" / f"corruption_{split}.parquet").write_text("placeholder corruption metadata\n")
        manifest = {}
        for modality in ("text", "region"):
            x_path = root / "token_fields" / f"{modality}_{split}.npy"
            pos_path = root / "positions" / f"{modality}_pos_{split}.npy"
            mask_path = root / "masks" / f"{modality}_mask_{split}.npy"
            x_path.write_text("placeholder token field\n")
            pos_path.write_text("placeholder positions\n")
            mask_path.write_text("placeholder mask\n")
            manifest[modality] = {
                "x": str(x_path.relative_to(root)),
                "pos": str(pos_path.relative_to(root)),
                "mask": str(mask_path.relative_to(root)),
            }
        (root / "masks" / f"text_mask_{split}.npy").write_text("placeholder mask\n")
        (root / "token_fields" / f"manifest_{split}.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    checksums = {}
    for path in required_cache_files(layout, splits=("val", "test")):
        if path.exists():
            checksums[str(path.relative_to(root))] = file_sha256(path)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            checksums.setdefault(str(path.relative_to(root)), file_sha256(path))
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True) + "\n")


def _write_valid_cmu_mosei_public_cache(cache_root: Path) -> None:
    from moat_ovha_torch.data.multimodal.cache_schema import (
        MultimodalCacheLayout,
        default_data_card,
        file_sha256,
        required_cache_files,
    )

    layout = MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1")
    root = layout.root
    for folder in ("masks", "positions", "provenance", "supervision", "token_fields"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    data_card = default_data_card(
        "cmu_mosei",
        "v0.1",
        ["text", "audio", "vision"],
        ["sentiment_emotion"],
    )
    data_card["metadata_availability"] = {"speaker_id": False}
    (root / "data_card.json").write_text(json.dumps(data_card, sort_keys=True) + "\n")
    (root / "splits.json").write_text(json.dumps({"val": ["val-utt"], "test": ["test-utt"]}, sort_keys=True) + "\n")
    (root / "samples.parquet").write_text("placeholder sentiment samples\n")
    feature_versions = {
        "text": "frozen-text-v1",
        "audio": "frozen-audio-v1",
        "vision": "frozen-vision-v1",
        "baselines": {
            "cross_attention_transformer": {
                "text": "frozen-text-v1",
                "audio": "frozen-audio-v1",
                "vision": "frozen-vision-v1",
            },
            "ovha_full": {
                "text": "frozen-text-v1",
                "audio": "frozen-audio-v1",
                "vision": "frozen-vision-v1",
            },
        },
    }
    (root / "provenance" / "feature_versions.json").write_text(json.dumps(feature_versions, sort_keys=True) + "\n")
    (root / "provenance" / "pseudo_label_versions.json").write_text(
        json.dumps({"generated_from_splits": ["train"], "version": "test-pseudo-v1"}, sort_keys=True) + "\n"
    )
    for split in ("val", "test"):
        source_id = f"{split}-utt"
        (root / "provenance" / f"source_ids_{split}.txt").write_text(f"{source_id}\n")
        (root / "provenance" / f"sample_records_{split}.jsonl").write_text(
            json.dumps(
                {
                    "source_id": source_id,
                    "split": split,
                    "original_split": split,
                    "raw_ref": f"raw://{source_id}",
                    "license_tag": "test-license",
                    "preprocessing_version": "sentiment-preprocess-v1",
                    "utterance_id": source_id,
                    "dialogue_id": f"dialogue-{split}",
                    "transcript_source": "official_transcript",
                    "missing_modality_mask_ref": f"supervision/missing_modality_mask_{split}.npy",
                    "corruption_metadata_ref": f"supervision/corruption_{split}.parquet",
                },
                sort_keys=True,
            )
            + "\n"
        )
        (root / "provenance" / f"failed_samples_{split}.jsonl").write_text("")
        (root / "supervision" / f"task_labels_{split}.npy").write_text("placeholder sentiment labels\n")
        (root / "supervision" / f"missing_modality_mask_{split}.npy").write_text("placeholder missing mask\n")
        (root / "supervision" / f"corruption_{split}.parquet").write_text("placeholder corruption metadata\n")
        manifest = {}
        for modality in ("text", "audio", "vision"):
            x_path = root / "token_fields" / f"{modality}_{split}.npy"
            pos_path = root / "positions" / f"{modality}_pos_{split}.npy"
            mask_path = root / "masks" / f"{modality}_mask_{split}.npy"
            x_path.write_text("placeholder token field\n")
            pos_path.write_text("placeholder positions\n")
            mask_path.write_text("placeholder mask\n")
            manifest[modality] = {
                "x": str(x_path.relative_to(root)),
                "pos": str(pos_path.relative_to(root)),
                "mask": str(mask_path.relative_to(root)),
            }
        (root / "masks" / f"text_mask_{split}.npy").write_text("placeholder mask\n")
        (root / "token_fields" / f"manifest_{split}.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    checksums = {}
    for path in required_cache_files(layout, splits=("val", "test")):
        if path.exists():
            checksums[str(path.relative_to(root))] = file_sha256(path)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            checksums.setdefault(str(path.relative_to(root)), file_sha256(path))
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True) + "\n")


def _valid_adapter_params() -> dict[str, list[str]]:
    return {
        "TLEO": ["lengthscale", "local_temperature", "scale", "bias"],
        "SPO": ["prototype_temperature", "prototype_logits_shift", "scale", "bias"],
        "LRIO": ["rank_logits", "interaction_temperature", "scale", "bias"],
        "CATO": ["alignment_temperature", "transport_scale", "scale", "bias"],
    }


def _complete_controlled_public_entry_report() -> dict[str, object]:
    required_gates = (
        "Stackability",
        "TLEO collapse",
        "SPO collapse",
        "LRIO collapse",
        "CATO collapse",
        "Router gate",
        "Router decomposition ablations",
        "RCEO gate",
        "Memory gate",
        "Adapter gate",
    )
    gate_table = {name: {"passed": True} for name in required_gates}
    gate_table.update(
        {
            "CATO alignment diagnostics": {"passed": True},
            "no-LRIO ablation": {"passed": True},
            "no-RCEO ablation": {"passed": True},
        }
    )
    return {
        "oracle_matrix_cells": ("learned_learned", "true_learned", "learned_true", "true_true"),
        "families": {
            "tleo_local_evidence": _complete_controlled_family_row(),
            "spo_global_prototype": _complete_controlled_family_row(),
            "lrio_low_rank_interaction": _complete_controlled_family_row(),
            "cato_alignment_transport": _complete_controlled_family_row(),
            "rceo_reliability_corruption": _complete_controlled_family_row(rceo=True),
            "mixed_relation_operator": _complete_controlled_family_row(),
        },
        "gate_table": gate_table,
        "go_no_go": {"controlled_multimodal_passed": True, "enter_public_multimodal": True},
    }


def _complete_controlled_family_row(*, rceo: bool = False) -> dict[str, object]:
    row = {
        "oracle_matrix": {
            "learned_learned": {"loss": 0.1},
            "true_learned": {"loss": 0.1},
            "learned_true": {"loss": 0.1},
            "true_true": {"loss": 0.0},
        },
        "TLEO_oracle_gap": 0.1,
        "SPO_oracle_gap": 0.1,
        "LRIO_oracle_gap": 0.1,
        "CATO_oracle_gap": 0.1,
        "no_evidence_router_delta": 0.1,
        "no_reliability_prior_delta": 0.1,
        "memory_only_router_delta": 0.1,
        "evidence_only_router_delta": 0.1,
    }
    if rceo:
        row["rceo_prior_effect"] = 0.1
    return row


def _passing_region_text_public_gate_report() -> dict[str, object]:
    return {
        "name": "region_text_public",
        "passed": True,
        "checks": {
            "full_beats_same_feature_baseline": {"passed": True},
            "full_beats_required_strong_baselines": {"passed": True},
            "no_cato_drops": {"passed": True},
            "cato_router_load_high": {"passed": True},
            "alignment_entropy_improves": {"passed": True},
            "cato_top_alignment_accuracy_high": {"passed": True},
            "grounding_accuracy_improves_with_entropy": {"passed": True},
            "rceo_visual_stress_router_shift": {"passed": True},
            "step14_public_diagnostics": {"passed": True},
            "robustness_passes": {"passed": True},
            "rceo_reliability_calibrated": {"passed": True},
        },
        "reasons": [],
        "evidence_artifacts": _gate_evidence_artifacts("phrase_region_grounding"),
    }


def _passing_sentiment_public_gate_report() -> dict[str, object]:
    return {
        "name": "sentiment_emotion_public",
        "passed": True,
        "checks": {
            "full_beats_same_feature_baseline": {"passed": True},
            "full_beats_lmf_or_mult_baseline": {"passed": True},
            "no_lrio_drops": {"passed": True},
            "no_spo_drops": {"passed": True},
            "no_rceo_drops": {"passed": True},
            "lrio_router_load_high": {"passed": True},
            "spo_router_load_high": {"passed": True},
            "lrio_rank_entropy_present": {"passed": True},
            "spo_prototype_entropy_present": {"passed": True},
            "spo_top_prototype_differentiates": {"passed": True},
            "step14_public_diagnostics": {"passed": True},
            "robustness_passes": {"passed": True},
            "rceo_reliability_calibrated": {"passed": True},
        },
        "reasons": [],
        "evidence_artifacts": _gate_evidence_artifacts("sentiment_emotion"),
    }


def _gate_evidence_artifacts(task: str) -> dict[str, object]:
    return {
        "task": task,
        "split": "test",
        "generated_by": "scripts/multimodal/evaluate_public_gates.py",
        "statistics_summary": {"path": f"artifacts/{task}_statistics_summary.json", "sha256": "a" * 64},
        "diagnostics": {"path": f"artifacts/{task}_diagnostics.jsonl", "sha256": "b" * 64},
        "robustness_summary": {"path": f"artifacts/{task}_robustness_summary.json", "sha256": "c" * 64},
        "raw_metrics": [{"path": f"artifacts/{task}_raw_metrics_seed1.jsonl", "sha256": "d" * 64}],
    }


def _complete_diagnostic_row() -> dict[str, object]:
    return {
        "router_entropy": 1.0,
        "router_memory_logit_norm": 0.1,
        "router_evidence_logit_norm": 0.2,
        "router_reliability_logit_norm": 0.3,
        "router_load_by_candidate": {"TLEO": 0.25, "SPO": 0.25, "LRIO": 0.25, "CATO": 0.25},
        "router_logit_parts": {"memory": 0.1, "evidence": 0.2, "reliability": 0.3},
        "candidate_loss": {"TLEO": 0.1, "SPO": 0.2, "LRIO": 0.3, "CATO": 0.4},
        "adapter_params": {
            "TLEO_lengthscale": 0.5,
            "SPO_temperature": 1.0,
            "LRIO_rank_entropy": 0.6,
            "CATO_alignment_temperature": 0.7,
        },
        "memory_slot_norm": {"TLEO": 1.0, "SPO": 1.0, "LRIO": 1.0, "CATO": 1.0},
        "stackability_passed": True,
        "candidate_diagnostics": {
            "TLEO": {"lengthscale": 0.5, "local_entropy": 0.2, "local_window_size": 5, "candidate_loss": 0.1},
            "SPO": {
                "prototype_entropy": 0.4,
                "top_prototype": 2,
                "prototype_temperature": 1.0,
                "candidate_loss": 0.2,
            },
            "LRIO": {
                "rank_entropy": 0.6,
                "rank_top_k": [0, 1],
                "pair_interaction_strength": 0.8,
                "candidate_loss": 0.3,
            },
            "CATO": {
                "alignment_entropy": 0.3,
                "top_k_alignment": [0, 2],
                "transport_marginal_error": 0.03,
                "candidate_loss": 0.4,
            },
            "RCEO": {"modality_reliability": 0.9, "reliability_bias_norm": 0.1, "corruption_response": 0.2},
        },
    }


if __name__ == "__main__":
    unittest.main()
