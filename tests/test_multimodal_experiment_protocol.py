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

    def test_region_text_public_alignment_ce_requires_declared_alignment_labels(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        base_public = json.loads((ROOT / "configs" / "multimodal_refcoco_public_smoke.json").read_text())
        missing_alignment_label_contract = {
            **base_public,
            "require_public_alignment_labels": False,
        }

        with self.assertRaisesRegex(
            ValueError,
            "public_alignment_ce requires require_public_alignment_labels=true",
        ):
            MultimodalExperimentConfig.from_mapping(missing_alignment_label_contract)

        contrastive_public = {
            **base_public,
            "name": "refcoco_contrastive_public",
            "require_public_alignment_labels": False,
            "losses_by_stage": {
                "T0": ["cache_validation"],
                "T5": ["task_loss", "public_contrastive_retrieval", "candidate_individual_loss"],
            },
        }
        config = MultimodalExperimentConfig.from_mapping(contrastive_public)

        self.assertIn("public_contrastive_retrieval", config.losses_by_stage["T5"])

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
            "require_public_alignment_labels": True,
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
            "require_public_alignment_labels": True,
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
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
                region_gate_report=_passing_region_text_public_gate_report(tmp_path / "artifacts"),
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
                region_gate_report=_passing_region_text_public_gate_report(tmp_path / "artifacts"),
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
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
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
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
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
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
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            region_report = _passing_region_text_public_gate_report()
            sentiment_report = _passing_sentiment_public_gate_report()
            region_report["checks"].pop("cato_top_alignment_accuracy_high")
            sentiment_report["checks"].pop("no_rceo_drops")

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
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
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            region_report = {**_passing_region_text_public_gate_report(), "reasons": "hidden failure"}
            sentiment_report = _passing_sentiment_public_gate_report()
            sentiment_report["checks"]["robustness_passes"] = {
                "passed": True,
                "reason": "robustness summary failed",
            }

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
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
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
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
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
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

    def test_topconf_main_entry_rejects_public_gate_without_robustness_rows_artifact(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            sentiment_report = _passing_sentiment_public_gate_report(tmp_path / "artifacts")
            sentiment_report["evidence_artifacts"].pop("robustness_rows", None)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
                region_gate_report=_passing_region_text_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "sentiment_emotion_public gate evidence_artifacts missing artifact: robustness_rows",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_rejects_public_gate_artifact_hash_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            region_report = _passing_region_text_public_gate_report()
            region_artifact = tmp_path / "region_statistics_summary.json"
            region_artifact.write_text(json.dumps({"task": "phrase_region_grounding"}) + "\n")
            region_report["evidence_artifacts"]["statistics_summary"] = {
                "path": str(region_artifact),
                "sha256": "0" * 64,
            }
            sentiment_report = _passing_sentiment_public_gate_report()
            missing_raw_metric = tmp_path / "missing_raw_metric.jsonl"
            sentiment_report["evidence_artifacts"]["raw_metrics"] = [
                {"path": str(missing_raw_metric), "sha256": "1" * 64}
            ]

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
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
        self.assertIn(
            "region_text_public gate evidence_artifacts statistics_summary.sha256 does not match file content",
            joined,
        )
        self.assertIn(
            "sentiment_emotion_public gate evidence_artifacts raw_metrics[0].path does not exist",
            joined,
        )

    def test_topconf_main_entry_rejects_public_gate_statistics_content_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            region_report = _passing_region_text_public_gate_report(tmp_path / "artifacts")
            region_statistics = Path(region_report["evidence_artifacts"]["statistics_summary"]["path"])
            region_statistics.write_text(
                json.dumps(
                    {
                        "main_table": {"sentiment_emotion": {"test": {}}},
                        "metadata": {"raw_metric_paths": []},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            region_report["evidence_artifacts"]["statistics_summary"]["sha256"] = file_sha256(region_statistics)

            sentiment_report = _passing_sentiment_public_gate_report(tmp_path / "artifacts")
            sentiment_statistics = Path(sentiment_report["evidence_artifacts"]["statistics_summary"]["path"])
            sentiment_statistics.write_text(
                json.dumps(
                    {
                        "main_table": {"sentiment_emotion": {"test": {"ovha_full": {"mean": 0.7}}}},
                        "metadata": {"raw_metric_paths": []},
                        "per_seed_appendix": [],
                        "reporting_metadata": {"per_seed_table": []},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            sentiment_report["evidence_artifacts"]["statistics_summary"]["sha256"] = file_sha256(sentiment_statistics)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
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
        self.assertIn(
            "region_text_public gate statistics_summary missing main_table entry for task/split: phrase_region_grounding/test",
            joined,
        )
        self.assertIn(
            "sentiment_emotion_public gate statistics_summary does not reference raw_metrics[0]",
            joined,
        )

    def test_topconf_main_entry_rejects_public_gate_statistics_without_per_seed_evidence(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            region_report = _passing_region_text_public_gate_report(tmp_path / "artifacts")
            region_statistics = Path(region_report["evidence_artifacts"]["statistics_summary"]["path"])
            summary = json.loads(region_statistics.read_text())
            summary["per_seed_appendix"] = [
                row
                for row in summary["per_seed_appendix"]
                if not (
                    row["task"] == "phrase_region_grounding"
                    and row["split"] == "test"
                    and row["model"] == "ovha_full"
                    and row["seed"] == 3
                )
            ]
            region_statistics.write_text(json.dumps(summary, sort_keys=True) + "\n")
            region_report["evidence_artifacts"]["statistics_summary"]["sha256"] = file_sha256(region_statistics)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
                region_gate_report=region_report,
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "region_text_public gate statistics_summary validation failed: "
            "phrase_region_grounding/test/ovha_full seed_count disagrees with per_seed_appendix",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_rejects_public_gate_raw_metrics_disagreeing_with_statistics(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            region_report = _passing_region_text_public_gate_report(tmp_path / "artifacts")
            raw_metrics = Path(region_report["evidence_artifacts"]["raw_metrics"][0]["path"])
            raw_metrics.write_text(
                json.dumps(
                    {
                        "task": "phrase_region_grounding",
                        "split": "test",
                        "model": "ovha_full",
                        "seed": 999,
                        "score": 0.0,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            region_report["evidence_artifacts"]["raw_metrics"][0]["sha256"] = file_sha256(raw_metrics)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
                region_gate_report=region_report,
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "region_text_public gate raw_metrics[0] rows must cover statistics_summary per_seed_appendix",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_rejects_public_gate_diagnostics_and_robustness_content_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            region_report = _passing_region_text_public_gate_report(tmp_path / "artifacts")
            region_diagnostics = Path(region_report["evidence_artifacts"]["diagnostics"]["path"])
            region_diagnostics.write_text(json.dumps({"setting": "clean", "public_diagnostics": {}}) + "\n")
            region_report["evidence_artifacts"]["diagnostics"]["sha256"] = file_sha256(region_diagnostics)

            sentiment_report = _passing_sentiment_public_gate_report(tmp_path / "artifacts")
            sentiment_robustness = Path(sentiment_report["evidence_artifacts"]["robustness_summary"]["path"])
            sentiment_robustness.write_text(
                json.dumps(
                    {
                        "full_model": "ovha_full",
                        "baseline_model": "cross_attention_transformer",
                        "full_drop_less_than_baseline": False,
                        "rceo_reliability_monotonic": True,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            sentiment_report["evidence_artifacts"]["robustness_summary"]["sha256"] = file_sha256(sentiment_robustness)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
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
        self.assertIn(
            "region_text_public gate diagnostics missing CATO router load by phrase type",
            joined,
        )
        self.assertIn(
            "sentiment_emotion_public gate robustness_summary does not show lower full-model drop",
            joined,
        )

    def test_topconf_main_entry_rejects_robustness_summary_without_observed_stress_coverage(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            region_report = _passing_region_text_public_gate_report(tmp_path / "artifacts")
            robustness_summary = Path(region_report["evidence_artifacts"]["robustness_summary"]["path"])
            payload = json.loads(robustness_summary.read_text())
            payload["required_stress_coverage"].pop("observed", None)
            robustness_summary.write_text(json.dumps(payload, sort_keys=True) + "\n")
            region_report["evidence_artifacts"]["robustness_summary"]["sha256"] = file_sha256(robustness_summary)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
                region_gate_report=region_report,
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "region_text_public gate robustness_summary required_stress_coverage must list observed stress targets",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_rejects_robustness_summary_with_truncated_required_stress_coverage(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            sentiment_report = _passing_sentiment_public_gate_report(tmp_path / "artifacts")
            robustness_summary = Path(sentiment_report["evidence_artifacts"]["robustness_summary"]["path"])
            payload = json.loads(robustness_summary.read_text())
            payload["required_stress_coverage"]["required"] = ["image_blur"]
            payload["required_stress_coverage"]["observed"] = ["image_blur"]
            robustness_summary.write_text(json.dumps(payload, sort_keys=True) + "\n")
            sentiment_report["evidence_artifacts"]["robustness_summary"]["sha256"] = file_sha256(robustness_summary)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
                region_gate_report=_passing_region_text_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "sentiment_emotion_public gate robustness_summary required_stress_coverage "
            "required missing canonical Step 6 stress targets",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_rejects_robustness_summary_disagreeing_with_rows(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            region_report = _passing_region_text_public_gate_report(tmp_path / "artifacts")
            robustness_summary = Path(region_report["evidence_artifacts"]["robustness_summary"]["path"])
            payload = json.loads(robustness_summary.read_text())
            payload["auc_over_corruption_strength"]["ovha_full"] = 0.99
            robustness_summary.write_text(json.dumps(payload, sort_keys=True) + "\n")
            region_report["evidence_artifacts"]["robustness_summary"]["sha256"] = file_sha256(robustness_summary)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
                region_gate_report=region_report,
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "region_text_public gate robustness_summary.auc_over_corruption_strength.ovha_full "
            "disagrees with robustness_rows recomputation",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_recomputes_public_gate_from_artifacts(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            region_report = _passing_region_text_public_gate_report(tmp_path / "artifacts")
            region_diagnostics = Path(region_report["evidence_artifacts"]["diagnostics"]["path"])
            rows = _gate_diagnostic_rows("phrase_region_grounding")
            for row in rows:
                row["public_diagnostics"]["cato_router_load_by_phrase_type"] = {"object": "not-a-number"}
            region_diagnostics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
            region_report["evidence_artifacts"]["diagnostics"]["sha256"] = file_sha256(region_diagnostics)

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
                region_gate_report=region_report,
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn("region_text_public gate artifact recomputation failed", joined)
        self.assertIn("region-text public diagnostics missing CATO router load by phrase type", joined)

    def test_topconf_main_entry_rejects_public_gate_report_disagreeing_with_artifacts(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            region_report = _passing_region_text_public_gate_report(tmp_path / "artifacts")
            region_report["checks"]["no_cato_drops"]["value"] = 999.0

            report = validate_topconf_main_experiment_entry(
                controlled_report=_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"),
                region_gate_report=region_report,
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "region_text_public gate check no_cato_drops.value disagrees with artifact recomputation",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_recomputes_controlled_report_from_artifacts(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            controlled_report = _complete_controlled_public_entry_report(tmp_path / "controlled_artifacts")
            controlled_rows = Path(controlled_report["evidence_artifacts"]["controlled_rows"]["path"])
            rows = [json.loads(line) for line in controlled_rows.read_text().splitlines() if line.strip()]
            for row in rows:
                if row["family"] == "mixed_relation_operator":
                    row["router_accuracy"] = 0.10
            controlled_rows.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
            controlled_report["evidence_artifacts"]["controlled_rows"]["sha256"] = file_sha256(controlled_rows)

            report = validate_topconf_main_experiment_entry(
                controlled_report=controlled_report,
                region_gate_report=_passing_region_text_public_gate_report(tmp_path / "artifacts"),
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn("controlled report artifact recomputation failed", joined)
        self.assertIn("controlled report required gate did not pass: Router gate", joined)

    def test_topconf_main_entry_rejects_controlled_diagnostics_without_family_coverage(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            controlled_report = _complete_controlled_public_entry_report(tmp_path / "controlled_artifacts")
            diagnostics_report = Path(controlled_report["evidence_artifacts"]["diagnostics_report"]["path"])
            diagnostics_report.write_text(
                json.dumps(
                    {
                        "family": "mixed_relation_operator",
                        "stackability_passed": True,
                        "oracle_matrix": _complete_controlled_family_row()["oracle_matrix"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            controlled_report["evidence_artifacts"]["diagnostics_report"]["sha256"] = file_sha256(diagnostics_report)

            report = validate_topconf_main_experiment_entry(
                controlled_report=controlled_report,
                region_gate_report=_passing_region_text_public_gate_report(tmp_path / "artifacts"),
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn("controlled report diagnostics_report missing controlled family: tleo_local_evidence", joined)
        self.assertIn("controlled report diagnostics_report missing controlled family: cato_alignment_transport", joined)

    def test_topconf_main_entry_rejects_controlled_diagnostics_that_disagree_with_rows(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            controlled_report = _complete_controlled_public_entry_report(tmp_path / "controlled_artifacts")
            diagnostics_report = Path(controlled_report["evidence_artifacts"]["diagnostics_report"]["path"])
            diagnostics_rows = [
                json.loads(line)
                for line in diagnostics_report.read_text().splitlines()
                if line.strip()
            ]
            for row in diagnostics_rows:
                if row["family"] == "cato_alignment_transport":
                    row["oracle_matrix"]["true_learned"]["loss"] = 0.99
            diagnostics_report.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in diagnostics_rows) + "\n"
            )
            controlled_report["evidence_artifacts"]["diagnostics_report"]["sha256"] = file_sha256(diagnostics_report)

            report = validate_topconf_main_experiment_entry(
                controlled_report=controlled_report,
                region_gate_report=_passing_region_text_public_gate_report(tmp_path / "artifacts"),
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "controlled report diagnostics_report cato_alignment_transport "
            "oracle_matrix.true_learned.loss disagrees with controlled_rows",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_rejects_controlled_diagnostics_missing_gate_fields(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            controlled_report = _complete_controlled_public_entry_report(tmp_path / "controlled_artifacts")
            diagnostics_report = Path(controlled_report["evidence_artifacts"]["diagnostics_report"]["path"])
            diagnostics_rows = [
                json.loads(line)
                for line in diagnostics_report.read_text().splitlines()
                if line.strip()
            ]
            for row in diagnostics_rows:
                if row["family"] == "cato_alignment_transport":
                    row.pop("alignment_entropy_delta", None)
            diagnostics_report.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in diagnostics_rows) + "\n"
            )
            controlled_report["evidence_artifacts"]["diagnostics_report"]["sha256"] = file_sha256(diagnostics_report)

            report = validate_topconf_main_experiment_entry(
                controlled_report=controlled_report,
                region_gate_report=_passing_region_text_public_gate_report(tmp_path / "artifacts"),
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "controlled report diagnostics_report cato_alignment_transport missing gate diagnostic: alignment_entropy_delta",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_rejects_controlled_report_disagreeing_with_artifacts(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            controlled_report = _complete_controlled_public_entry_report(tmp_path / "controlled_artifacts")
            controlled_report["families"]["mixed_relation_operator"]["router_accuracy"] = 0.99

            report = validate_topconf_main_experiment_entry(
                controlled_report=controlled_report,
                region_gate_report=_passing_region_text_public_gate_report(tmp_path / "artifacts"),
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "controlled report mixed_relation_operator.router_accuracy disagrees with artifact recomputation",
            "\n".join(report.errors),
        )

    def test_topconf_main_entry_rejects_controlled_report_stackability_disagreeing_with_artifacts(self):
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from moat_ovha_torch.eval.multimodal_main_experiment_entry import (
            CacheValidationTarget,
            validate_topconf_main_experiment_entry,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)

            controlled_report = _complete_controlled_public_entry_report(tmp_path / "controlled_artifacts")
            controlled_report["families"]["tleo_local_evidence"]["stackability_passed"] = False

            report = validate_topconf_main_experiment_entry(
                controlled_report=controlled_report,
                region_gate_report=_passing_region_text_public_gate_report(tmp_path / "artifacts"),
                sentiment_gate_report=_passing_sentiment_public_gate_report(tmp_path / "artifacts"),
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
        self.assertIn(
            "controlled report tleo_local_evidence.stackability_passed disagrees with artifact recomputation",
            "\n".join(report.errors),
        )

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


def _complete_controlled_public_entry_report(artifact_root: Path | None = None) -> dict[str, object]:
    if artifact_root is not None:
        from moat_ovha_torch.data.multimodal.cache_schema import file_sha256
        from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report

        artifact_root.mkdir(parents=True, exist_ok=True)
        controlled_rows = artifact_root / "controlled_multimodal_rows.jsonl"
        diagnostics_report = artifact_root / "controlled_multimodal_diagnostics.jsonl"
        rows = _complete_controlled_rows()
        controlled_rows.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
        diagnostics_report.write_text(
            "\n".join(
                json.dumps(_controlled_diagnostic_row(row), sort_keys=True)
                for row in rows
            )
            + "\n"
        )
        evidence_artifacts = {
            "task": "controlled_multimodal",
            "generated_by": "scripts/multimodal/summarize_controlled_report.py",
            "controlled_rows": {"path": str(controlled_rows), "sha256": file_sha256(controlled_rows)},
            "diagnostics_report": {"path": str(diagnostics_report), "sha256": file_sha256(diagnostics_report)},
        }
        return build_controlled_report(rows, evidence_artifacts=evidence_artifacts)

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
    rows_by_family = {
        row["family"]: {key: value for key, value in row.items() if key != "family"}
        for row in _complete_controlled_rows()
    }
    return {
        "oracle_matrix_cells": ("learned_learned", "true_learned", "learned_true", "true_true"),
        "families": rows_by_family,
        "gate_table": gate_table,
        "go_no_go": {"controlled_multimodal_passed": True, "enter_public_multimodal": True},
        "evidence_artifacts": _controlled_evidence_artifacts(),
    }


def _controlled_evidence_artifacts() -> dict[str, object]:
    return {
        "task": "controlled_multimodal",
        "generated_by": "scripts/multimodal/summarize_controlled_report.py",
        "controlled_rows": {"path": "artifacts/controlled_multimodal_rows.jsonl", "sha256": "e" * 64},
        "diagnostics_report": {"path": "artifacts/controlled_multimodal_diagnostics.jsonl", "sha256": "f" * 64},
    }


def _complete_controlled_rows() -> list[dict[str, object]]:
    rows = [
        {"family": "tleo_local_evidence", **_complete_controlled_family_row()},
        {
            "family": "spo_global_prototype",
            **_complete_controlled_family_row(),
            "prototype_kl_delta": 0.12,
        },
        {
            "family": "lrio_low_rank_interaction",
            **_complete_controlled_family_row(),
            "rank_logits_kl_delta": 0.13,
            "no_lrio_delta": 0.14,
        },
        {
            "family": "cato_alignment_transport",
            **_complete_controlled_family_row(),
            "alignment_entropy_delta": 0.15,
            "alignment_topk_delta": 0.16,
        },
        {
            "family": "rceo_reliability_corruption",
            **_complete_controlled_family_row(rceo=True),
            "rceo_reliability_monotonic": True,
            "rceo_router_load_shift": 0.17,
            "rceo_reliability_curve": [
                {"corruption_strength": 0.0, "mean_reliability": 0.90},
                {"corruption_strength": 0.5, "mean_reliability": 0.65},
            ],
            "no_rceo_delta": 0.18,
        },
        {
            "family": "mixed_relation_operator",
            **_complete_controlled_family_row(),
            "router_accuracy": 0.86,
            "no_lrio_delta": 0.19,
            "no_rceo_delta": 0.20,
        },
    ]
    return rows


def _controlled_diagnostic_row(row: dict[str, object]) -> dict[str, object]:
    keys = (
        "family",
        "stackability_passed",
        "oracle_matrix",
        "TLEO_oracle_gap",
        "SPO_oracle_gap",
        "LRIO_oracle_gap",
        "CATO_oracle_gap",
        "no_evidence_router_delta",
        "no_reliability_prior_delta",
        "memory_only_router_delta",
        "evidence_only_router_delta",
        "no_operator_memory_delta",
        "no_hyper_adapter_delta",
        "prototype_kl_delta",
        "rank_logits_kl_delta",
        "alignment_entropy_delta",
        "alignment_topk_delta",
        "rceo_prior_effect",
        "rceo_reliability_monotonic",
        "rceo_router_load_shift",
        "rceo_reliability_curve",
        "router_accuracy",
        "no_lrio_delta",
        "no_rceo_delta",
    )
    return {key: row[key] for key in keys if key in row}


def _complete_controlled_family_row(*, rceo: bool = False) -> dict[str, object]:
    row = {
        "stackability_passed": True,
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
        "no_operator_memory_delta": 0.1,
        "no_hyper_adapter_delta": 0.1,
    }
    if rceo:
        row["rceo_prior_effect"] = 0.1
    return row


def _passing_region_text_public_gate_report(artifact_root: Path | None = None) -> dict[str, object]:
    evidence_artifacts = _gate_evidence_artifacts("phrase_region_grounding", artifact_root)
    if artifact_root is not None:
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate

        task = "phrase_region_grounding"
        raw_metrics = Path(evidence_artifacts["raw_metrics"][0]["path"])
        report = evaluate_region_text_gate(
            statistics_summary=_gate_statistics_summary(task, raw_metrics),
            diagnostics_rows=_gate_diagnostic_rows(task),
            no_cato_score=_gate_score_for_model(task, "ovha_no_cato", full_score=0.80, baseline_score=0.72),
            robustness_summary=_gate_robustness_summary(task),
            task=task,
            split="test",
        )
    else:
        report = {
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
        }
    report["evidence_artifacts"] = evidence_artifacts
    return report


def _passing_sentiment_public_gate_report(artifact_root: Path | None = None) -> dict[str, object]:
    evidence_artifacts = _gate_evidence_artifacts("sentiment_emotion", artifact_root)
    if artifact_root is not None:
        from moat_ovha_torch.eval.multimodal_public_gates import evaluate_sentiment_gate

        task = "sentiment_emotion"
        raw_metrics = Path(evidence_artifacts["raw_metrics"][0]["path"])
        report = evaluate_sentiment_gate(
            statistics_summary=_gate_statistics_summary(task, raw_metrics),
            diagnostics_rows=_gate_diagnostic_rows(task),
            ablation_scores={
                "ovha_no_lrio": _gate_score_for_model(task, "ovha_no_lrio", full_score=0.76, baseline_score=0.74),
                "ovha_no_spo": _gate_score_for_model(task, "ovha_no_spo", full_score=0.76, baseline_score=0.74),
                "ovha_no_rceo": _gate_score_for_model(task, "ovha_no_rceo", full_score=0.76, baseline_score=0.74),
            },
            robustness_summary=_gate_robustness_summary(task),
            task=task,
            split="test",
        )
    else:
        report = {
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
        }
    report["evidence_artifacts"] = evidence_artifacts
    return report


def _gate_evidence_artifacts(task: str, artifact_root: Path | None = None) -> dict[str, object]:
    if artifact_root is not None:
        from moat_ovha_torch.data.multimodal.cache_schema import file_sha256

        artifact_root.mkdir(parents=True, exist_ok=True)
        statistics = artifact_root / f"{task}_statistics_summary.json"
        diagnostics = artifact_root / f"{task}_diagnostics.jsonl"
        robustness = artifact_root / f"{task}_robustness_summary.json"
        robustness_rows = artifact_root / f"{task}_robustness_rows.jsonl"
        raw_metrics = artifact_root / f"{task}_raw_metrics_seed1.jsonl"
        statistics_summary = _gate_statistics_summary(task, raw_metrics)
        statistics.write_text(json.dumps(statistics_summary, sort_keys=True) + "\n")
        diagnostics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in _gate_diagnostic_rows(task)) + "\n")
        robustness_rows.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in _gate_robustness_rows()) + "\n"
        )
        robustness.write_text(json.dumps(_gate_robustness_summary(task), sort_keys=True) + "\n")
        raw_metrics.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in statistics_summary["per_seed_appendix"])
            + "\n"
        )
        return {
            "task": task,
            "split": "test",
            "generated_by": "scripts/multimodal/evaluate_public_gates.py",
            "statistics_summary": {"path": str(statistics), "sha256": file_sha256(statistics)},
            "diagnostics": {"path": str(diagnostics), "sha256": file_sha256(diagnostics)},
            "robustness_summary": {"path": str(robustness), "sha256": file_sha256(robustness)},
            "robustness_rows": {"path": str(robustness_rows), "sha256": file_sha256(robustness_rows)},
            "raw_metrics": [{"path": str(raw_metrics), "sha256": file_sha256(raw_metrics)}],
        }
    return {
        "task": task,
        "split": "test",
        "generated_by": "scripts/multimodal/evaluate_public_gates.py",
        "statistics_summary": {"path": f"artifacts/{task}_statistics_summary.json", "sha256": "a" * 64},
        "diagnostics": {"path": f"artifacts/{task}_diagnostics.jsonl", "sha256": "b" * 64},
        "robustness_summary": {"path": f"artifacts/{task}_robustness_summary.json", "sha256": "c" * 64},
        "robustness_rows": {"path": f"artifacts/{task}_robustness_rows.jsonl", "sha256": "e" * 64},
        "raw_metrics": [{"path": f"artifacts/{task}_raw_metrics_seed1.jsonl", "sha256": "d" * 64}],
    }


def _gate_statistics_summary(task: str, raw_metrics: Path) -> dict[str, object]:
    models = _gate_models_for_task(task)
    full_score = 0.80 if task == "phrase_region_grounding" else 0.76
    baseline_score = 0.72 if task == "phrase_region_grounding" else 0.74
    model_scores = {
        model: _gate_score_for_model(task, model, full_score=full_score, baseline_score=baseline_score)
        for model in models
    }
    model_rows = {
        model: {
            "mean": score,
            "std": 0.01,
            "ci95": [score - (1.96 * 0.01 / (3 ** 0.5)), score + (1.96 * 0.01 / (3 ** 0.5))],
            "seed_count": 3,
            "per_seed_scores": [score - 0.01, score, score + 0.01],
            "higher_is_better": True,
        }
        for model, score in model_scores.items()
    }
    parameter_count = {model: 120000 + index for index, model in enumerate(models)}
    feature_versions = (
        {"text": "frozen-text-v1", "audio": "frozen-audio-v1", "vision": "frozen-vision-v1"}
        if task == "sentiment_emotion"
        else {"text": "frozen-text-v1", "region": "frozen-region-v1"}
    )
    hardware = {"accelerator": "unit-test-cpu", "wall_clock_hours": 0.17}
    full_delta = full_score - baseline_score
    paired = {
        "common_seed_count": 3,
        "model_delta": "ovha_full_minus_cross_attention_transformer",
        "mean_delta": full_delta,
        "metric_direction": "higher_is_better",
        "paired_permutation_p": 0.25,
        "paired_bootstrap_ci95": [full_delta, full_delta],
        "baseline_comparisons": {
            model: {
                "common_seed_count": 3,
                "mean_delta": full_score - score,
                "metric_direction": "higher_is_better",
                "paired_permutation_p": 0.25,
                "paired_bootstrap_ci95": [full_score - score, full_score - score],
            }
            for model, score in model_scores.items()
            if model not in {"ovha_full", "cross_attention_transformer"}
        },
    }
    per_seed_table = [
        {
            "task": task,
            "split": "test",
            "model": model,
            "seed": seed,
            "score": score + (seed - 2) * 0.01,
            "raw_metric_path": str(raw_metrics),
            "parameter_count": parameter_count[model],
            "training_steps": 1000,
            "frozen_feature_extractor_version": feature_versions,
            "hardware": hardware,
            "seed_count_rationale": "unit-test fixture uses the plan minimum of 3 seeds; production main tables should use 5 seeds",
            "public_metrics": _public_metrics_for_task(task),
        }
        for model, score in model_scores.items()
        for seed in (1, 2, 3)
    ]
    return {
        "main_table": {task: {"test": model_rows}},
        "paired_tests": {task: {"test": paired}},
        "metadata": {
            "parameter_count": parameter_count,
            "baseline_strength": {},
            "training_steps": 1000,
            "frozen_feature_extractor_version": feature_versions,
            "hardware": hardware,
            "label_provenance": {},
            "raw_metric_paths": [str(raw_metrics)],
        },
        "per_seed_appendix": per_seed_table,
        "reporting_metadata": {
            "parameter_count": parameter_count,
            "training_steps": {model: 1000 for model in models},
            "frozen_feature_versions": feature_versions,
            "hardware": hardware,
            "wall_clock_summary": {"wall_clock_hours": 0.17},
            "seed_count_rationale": "unit-test fixture uses the plan minimum of 3 seeds; production main tables should use 5 seeds",
            "per_seed_table": per_seed_table,
        },
    }


def _public_metrics_for_task(task: str) -> dict[str, object]:
    if task == "sentiment_emotion":
        return {
            "mae": 0.3,
            "pearson_correlation": 0.7,
            "accuracy": 0.74,
            "f1": 0.73,
            "missing_modality_performance_drop": 0.08,
            "corruption_robustness_auc": 0.82,
            "router_load_by_corruption_type": {"missing_audio": 0.45, "audio_noise": 0.42},
            "lrio_rank_entropy": 0.6,
            "spo_prototype_entropy": 0.5,
            "rceo_reliability_calibration": {
                "ece": 0.08,
                "bin_count": 5,
                "calibration_curve": [{"confidence": 0.8, "accuracy": 0.76}],
            },
        }
    return {
        "acc_at_0_5": 0.80,
        "recall_at_1": 0.78,
        "recall_at_5": 0.90,
        "mean_iou": 0.62,
        "phrase_region_topk_accuracy": 0.85,
        "alignment_entropy": 0.4,
        "cato_router_load": 0.55,
        "cato_candidate_loss": 0.2,
        "cato_top_alignment_accuracy": 0.7,
        "null_unmatched_rate": 0.05,
    }


def _gate_models_for_task(task: str) -> tuple[str, ...]:
    if task == "sentiment_emotion":
        return (
            "ovha_full",
            "cross_attention_transformer",
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
    return (
        "ovha_full",
        "cross_attention_transformer",
        "text_only",
        "region_only",
        "concat_fusion",
        "modality_expert_moe",
        "clip_style_region_text_retrieval",
        "cato_only",
        "ovha_no_cato",
        "ovha_no_rceo",
        "ovha_no_evidence_router",
    )


def _gate_score_for_model(task: str, model: str, *, full_score: float, baseline_score: float) -> float:
    if model == "ovha_full":
        return full_score
    if model == "cross_attention_transformer":
        return baseline_score
    if task == "sentiment_emotion":
        return {
            "tfn_lmf": 0.73,
            "mult_style_crossmodal_transformer": 0.72,
            "ovha_no_lrio": 0.70,
            "ovha_no_spo": 0.71,
            "ovha_no_rceo": 0.68,
            "ovha_no_evidence_router": 0.69,
        }.get(model, 0.71)
    return {
        "modality_expert_moe": 0.71,
        "ovha_no_cato": 0.70,
        "ovha_no_rceo": 0.69,
        "ovha_no_evidence_router": 0.68,
    }.get(model, 0.70)


def _gate_diagnostic_rows(task: str) -> list[dict[str, object]]:
    if task == "sentiment_emotion":
        return [
            {
                "setting": "clean",
                "router_load_by_candidate": {"TLEO": 0.10, "SPO": 0.35, "LRIO": 0.40, "CATO": 0.15},
                "candidate_diagnostics": {
                    "LRIO": {"rank_entropy": 0.6},
                    "SPO": {
                        "prototype_entropy": 0.5,
                        "top_prototype_by_class": {"negative": 1, "positive": 4},
                    },
                },
                "public_diagnostics": {
                    "lrio_rank_entropy_by_modality_pair": {"text_audio": 0.42, "text_vision": 0.37},
                    "spo_prototype_load_by_emotion_class": {
                        "negative": {"p0": 0.70, "p1": 0.20, "p2": 0.10},
                        "positive": {"p0": 0.15, "p1": 0.75, "p2": 0.10},
                    },
                    "rceo_reliability_shift_under_missing_noisy_modality": {"missing_audio": -0.18},
                    "router_load_by_condition": {
                        "clean": {"TLEO": 0.10, "SPO": 0.35, "LRIO": 0.40, "CATO": 0.15},
                        "corrupted": {"TLEO": 0.15, "SPO": 0.45, "LRIO": 0.25, "CATO": 0.15},
                        "missing": {"TLEO": 0.20, "SPO": 0.45, "LRIO": 0.20, "CATO": 0.15},
                    },
                },
            },
        ]
    return [
        _gate_region_diagnostic(
            "clean",
            {"CATO": 0.55, "TLEO": 0.20, "SPO": 0.15, "LRIO": 0.10},
            cato_entropy=0.30,
            grounding_accuracy=0.74,
            top_alignment_accuracy=0.72,
            rceo_reliability=0.90,
        ),
        _gate_region_diagnostic(
            "no_cato",
            {"CATO": 0.0, "TLEO": 0.40, "SPO": 0.40, "LRIO": 0.20},
            cato_entropy=0.90,
            grounding_accuracy=0.52,
            top_alignment_accuracy=0.20,
        ),
        _gate_region_diagnostic(
            "corrupted_visual",
            {"CATO": 0.28, "TLEO": 0.30, "SPO": 0.32, "LRIO": 0.10},
            cato_entropy=0.58,
            grounding_accuracy=0.61,
            top_alignment_accuracy=0.56,
            rceo_reliability=0.55,
            rceo_corruption_response=0.35,
        ),
    ]


def _gate_region_diagnostic(
    setting: str,
    loads: dict[str, float],
    *,
    cato_entropy: float,
    grounding_accuracy: float,
    top_alignment_accuracy: float,
    rceo_reliability: float | None = None,
    rceo_corruption_response: float | None = None,
) -> dict[str, object]:
    candidate_diagnostics: dict[str, object] = {
        "CATO": {
            "alignment_entropy": cato_entropy,
            "grounding_accuracy": grounding_accuracy,
            "top_alignment_accuracy": top_alignment_accuracy,
        }
    }
    if rceo_corruption_response is not None:
        candidate_diagnostics["RCEO"] = {"corruption_response": rceo_corruption_response}
    row: dict[str, object] = {
        "setting": setting,
        "router_load_by_candidate": loads,
        "candidate_diagnostics": candidate_diagnostics,
        "public_diagnostics": {
            "cato_router_load_by_phrase_type": {"object_noun_phrase": 0.52, "attribute_phrase": 0.41},
            "no_cato_delta_by_object_size": {"small": 0.08, "medium": 0.06, "large": 0.04},
            "no_cato_delta_by_phrase_length": {"short": 0.05, "long": 0.07},
            "rceo_reliability_shift_under_blurred_regions": -0.22,
        },
    }
    if rceo_reliability is not None:
        row["rceo_reliability"] = rceo_reliability
    return row


def _gate_robustness_summary(task: str) -> dict[str, object]:
    from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows

    summary = summarize_robustness_rows(
        _gate_robustness_rows(),
        full_model="ovha_full",
        baseline_model="cross_attention_transformer",
    )
    summary["task"] = task
    return summary


def _gate_robustness_rows() -> list[dict[str, object]]:
    rows = [
        {
            "model": "ovha_full",
            "corruption_type": "image_blur",
            "corruption_strength": 0.0,
            "score": 0.75,
            "rceo_reliability": 0.90,
            "rceo_observed_reliability": 0.88,
            "router_load_by_candidate": {"TLEO": 0.20, "SPO": 0.20, "LRIO": 0.40, "CATO": 0.20},
            "candidate_loss": {"TLEO": 0.12, "SPO": 0.20, "LRIO": 0.26, "CATO": 0.18},
        },
        {
            "model": "ovha_full",
            "corruption_type": "image_blur",
            "corruption_strength": 0.25,
            "score": 0.72,
            "rceo_reliability": 0.65,
            "rceo_observed_reliability": 0.63,
            "router_load_by_candidate": {"TLEO": 0.18, "SPO": 0.28, "LRIO": 0.32, "CATO": 0.22},
            "candidate_loss": {"TLEO": 0.13, "SPO": 0.17, "LRIO": 0.28, "CATO": 0.19},
        },
    ]
    for stress in _canonical_stress_rows():
        rows.append(
            {
                **stress,
                "model": "ovha_full",
                "corruption_strength": 0.5,
                "score": 0.69,
                "rceo_reliability": 0.45,
                "rceo_observed_reliability": 0.43,
                "router_load_by_candidate": {"TLEO": 0.18, "SPO": 0.40, "LRIO": 0.20, "CATO": 0.22},
                "candidate_loss": {"TLEO": 0.14, "SPO": 0.14, "LRIO": 0.31, "CATO": 0.19},
            }
        )
    rows.extend(
        [
            {
                "model": "cross_attention_transformer",
                "corruption_type": "image_blur",
                "corruption_strength": 0.0,
                "score": 0.75,
            },
            {
                "model": "cross_attention_transformer",
                "corruption_type": "image_blur",
                "corruption_strength": 0.5,
                "score": 0.63,
            },
            {
                "model": "ovha_no_rceo",
                "corruption_type": "image_blur",
                "corruption_strength": 0.0,
                "score": 0.74,
            },
            {
                "model": "ovha_no_rceo",
                "corruption_type": "image_blur",
                "corruption_strength": 0.5,
                "score": 0.58,
            },
            {
                "model": "ovha_no_evidence_router",
                "corruption_type": "image_blur",
                "corruption_strength": 0.0,
                "score": 0.74,
            },
            {
                "model": "ovha_no_evidence_router",
                "corruption_type": "image_blur",
                "corruption_strength": 0.5,
                "score": 0.57,
            },
        ]
    )
    return rows


def _canonical_stress_rows() -> list[dict[str, object]]:
    return [
        {"corruption_type": "missing_modality", "missing_modalities": ["text"]},
        {"corruption_type": "missing_modality", "missing_modalities": ["vision"]},
        {"corruption_type": "missing_modality", "missing_modalities": ["audio"]},
        {"corruption_type": "image_blur"},
        {"corruption_type": "image_crop"},
        {"corruption_type": "image_occlusion"},
        {"corruption_type": "audio_noise"},
        {"corruption_type": "audio_masking"},
        {"corruption_type": "text_token_mask"},
        {"corruption_type": "text_paraphrase"},
        {
            "corruption_type": "hard_negative_caption_mismatch",
            "mismatch_source_id": "other-caption",
        },
        {
            "corruption_type": "hard_negative_region_mismatch",
            "mismatch_source_id": "other-region",
        },
        {
            "corruption_type": "hard_negative_audio_mismatch",
            "mismatch_source_id": "other-audio",
        },
    ]


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
