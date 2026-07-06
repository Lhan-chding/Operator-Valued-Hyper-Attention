from __future__ import annotations

import importlib
import importlib.util
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
            ROOT / "configs" / "multimodal_refcoco_public_main.json",
            ROOT / "configs" / "multimodal_cmu_mosei_public_main.json",
            ROOT / "configs" / "multimodal_external_sota_references.json",
            ROOT / "configs" / "multimodal_robustness_smoke.json",
            ROOT / "moat_ovha_torch" / "config_multimodal.py",
            ROOT / "moat_ovha_torch" / "models" / "multimodal" / "baselines.py",
            ROOT / "moat_ovha_torch" / "eval" / "multimodal_diagnostics.py",
            ROOT / "scripts" / "multimodal" / "run_public_smoke.py",
            ROOT / "scripts" / "multimodal" / "bootstrap_public_downloads.py",
            ROOT / "scripts" / "multimodal" / "build_public_main_runbook.py",
            ROOT / "scripts" / "multimodal" / "build_external_sota_runbook.py",
            ROOT / "scripts" / "multimodal" / "validate_public_main_artifacts.py",
            ROOT / "scripts" / "multimodal" / "run_robustness_stress_smoke.py",
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
        self.assertIn("task_loss", config.losses_by_stage["T5"])
        self.assertNotIn("public_alignment_ce", config.losses_by_stage["T5"])
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

    def test_public_main_configs_are_non_smoke_five_seed_t5_plans(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task

        expectations = [
            (
                ROOT / "configs" / "multimodal_refcoco_public_main.json",
                "refcoco",
                "phrase_region_grounding",
                None,
                set(baseline_names_for_task("phrase_region_grounding")),
            ),
            (
                ROOT / "configs" / "multimodal_cmu_mosei_public_main.json",
                "cmu_mosei",
                "sentiment_emotion",
                None,
                set(baseline_names_for_task("sentiment_emotion")),
            ),
            (
                ROOT / "configs" / "multimodal_meld_public_main.json",
                "meld",
                "sentiment_emotion",
                None,
                set(baseline_names_for_task("sentiment_emotion")),
            ),
        ]
        for path, dataset, task_type, alignment_loss, required_baselines in expectations:
            with self.subTest(path=path):
                config = MultimodalExperimentConfig.from_file(path)
                payload = json.loads(path.read_text())

                self.assertEqual(config.dataset_name, dataset)
                self.assertEqual(config.task_type, task_type)
                self.assertEqual(config.training_stages, ("T0", "T5"))
                if dataset == "cmu_mosei" and path.name == "multimodal_cmu_mosei_public_main.json":
                    self.assertEqual(config.candidate_names, ("SPO", "TANSO"))
                    self.assertEqual(config.candidate_pool_names, ("SPO", "LRIO", "TANSO"))
                elif dataset in {"cmu_mosei", "meld"}:
                    self.assertEqual(config.candidate_names, ("SPO", "LRIO", "TANSO"))
                    expected_lrio_pairs = (
                        (("text", "audio"), ("text", "vision"))
                        if dataset == "cmu_mosei"
                        else (("text", "audio"), ("text", "vision"), ("audio", "vision"))
                    )
                    self.assertEqual(config.lrio_pairs, expected_lrio_pairs)
                else:
                    self.assertEqual(config.candidate_names, ("PRSO", "SRO", "TLEO", "CATO"))
                self.assertGreaterEqual(len(config.seeds), 5)
                self.assertEqual(len(set(config.seeds)), len(config.seeds))
                if dataset == "refcoco":
                    self.assertEqual(set(config.eval_splits), {"val", "testA", "testB"})
                    self.assertEqual(config.checkpoint_selection_metric, "acc_at_0_5")
                else:
                    self.assertEqual(set(config.eval_splits), {"val", "test"})
                self.assertTrue(config.enforce_same_features_for_baselines)
                self.assertTrue(config.fail_on_missing_cache_artifact)
                self.assertFalse(config.allow_hidden_losses)
                self.assertTrue(required_baselines.issubset(set(config.baseline_names)))
                self.assertNotIn("smoke", config.name)
                self.assertNotIn("smoke", str(config.output_dir))
                self.assertNotIn("smoke", path.name)
                self.assertEqual(payload["main_table_seed_policy"], "five_seed_default")
                self.assertIn("T0", config.losses_by_stage)
                self.assertIn("T5", config.losses_by_stage)
                self.assertIn("task_loss", config.losses_by_stage["T5"])
                self.assertIn("candidate_individual_loss", config.losses_by_stage["T5"])
                if alignment_loss is not None:
                    self.assertIn(alignment_loss, config.losses_by_stage["T5"])
                    self.assertTrue(config.require_public_alignment_labels)

    def test_region_checkpoint_selection_accepts_grounding_metrics(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig, selection_score_from_public_metrics

        base_public = json.loads((ROOT / "configs" / "multimodal_refcoco_public_smoke.json").read_text())
        for metric_name in ("acc_at_0_5", "recall_at_1", "mean_iou", "mrr"):
            with self.subTest(metric_name=metric_name):
                config = MultimodalExperimentConfig.from_mapping(
                    {
                        **base_public,
                        "checkpoint_selection_metric": metric_name,
                    }
                )
                self.assertEqual(config.checkpoint_selection_metric, metric_name)
                score = selection_score_from_public_metrics(
                    config,
                    {"acc_at_0_5": 0.75, "recall_at_1": 0.70, "mean_iou": 0.65, "mrr": 0.80},
                )
                self.assertAlmostEqual(score, 1.0 - {"acc_at_0_5": 0.75, "recall_at_1": 0.70, "mean_iou": 0.65, "mrr": 0.80}[metric_name])

        cmu_public = json.loads((ROOT / "configs" / "multimodal_cmu_mosei_public_main.json").read_text())
        with self.assertRaisesRegex(ValueError, "grounding checkpoint selection metric"):
            MultimodalExperimentConfig.from_mapping({**cmu_public, "checkpoint_selection_metric": "acc_at_0_5"})

    def test_cmu_tanso_public_main_config_is_admission_run_not_original_table_overwrite(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        path = ROOT / "configs" / "multimodal_cmu_mosei_tanso_public_main.json"
        config = MultimodalExperimentConfig.from_file(path)

        self.assertEqual(config.candidate_names, ("SPO", "LRIO", "TANSO"))
        self.assertEqual(config.base_candidate, "SPO")
        self.assertEqual(config.residual_candidates, ("LRIO", "TANSO"))
        self.assertIn("cmu_mosei_tanso_main", str(config.output_dir))
        self.assertIn("TANSO", config.adapter_params_by_candidate)
        self.assertNotEqual(
            config.output_dir,
            MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_main.json").output_dir,
        )

    def test_region_text_public_alignment_ce_requires_declared_alignment_labels(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        base_public = json.loads((ROOT / "configs" / "multimodal_refcoco_public_smoke.json").read_text())
        missing_alignment_label_contract = {
            **base_public,
            "require_public_alignment_labels": False,
        }
        config = MultimodalExperimentConfig.from_mapping(missing_alignment_label_contract)
        self.assertNotIn("public_alignment_ce", config.losses_by_stage["T5"])

        duplicate_alignment_contract = {
            **base_public,
            "require_public_alignment_labels": False,
            "losses_by_stage": {
                "T0": ["cache_validation"],
                "T5": ["task_loss", "public_alignment_ce", "candidate_individual_loss"],
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            "public_alignment_ce requires require_public_alignment_labels=true",
        ):
            MultimodalExperimentConfig.from_mapping(duplicate_alignment_contract)

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
        from moat_ovha_torch.models.multimodal.baselines import (
            baseline_names_for_task,
            baseline_protocol_for_name,
            external_reference_names_for_task,
            missing_required_baselines,
            ovha_ablation_names_for_task,
            same_feature_probe_names_for_task,
        )

        region = set(baseline_names_for_task("phrase_region_grounding"))
        self.assertEqual(
            region,
            {
                "random_valid",
                "train_slot_prior",
                "box_prior",
                "prso_clip_similarity",
                "candidate_mlp_reranker",
                "cross_attention_reranker",
                "index_prior_only",
                "text_only",
                "region_only",
                "concat_fusion",
                "cato_only",
                "ovha_no_cato",
                "ovha_no_rceo",
                "ovha_no_evidence_router",
            },
        )

        sentiment = set(baseline_names_for_task("sentiment_emotion"))
        self.assertEqual(
            sentiment,
            {
                "text_only",
                "audio_only",
                "vision_only",
                "concat_fusion",
                "spo_only",
                "lrio_only",
                "ovha_tanso_only",
                "ovha_spo_lrio",
                "ovha_lrio_tanso",
                "ovha_all_candidates_exploratory",
                "ovha_no_rceo",
                "ovha_with_evidence_router",
            },
        )
        self.assertNotIn("GroundingDINO", region)
        self.assertNotIn("MISA", sentiment)
        self.assertEqual(baseline_protocol_for_name("phrase_region_grounding", "concat_fusion"), "same_feature_sanity_probe")
        self.assertEqual(baseline_protocol_for_name("phrase_region_grounding", "prso_clip_similarity"), "same_candidate_strong_reranker")
        self.assertEqual(baseline_protocol_for_name("phrase_region_grounding", "cross_attention_reranker"), "same_candidate_strong_reranker")
        self.assertEqual(baseline_protocol_for_name("phrase_region_grounding", "clip_geometry_mlp"), "same_candidate_strong_reranker")
        self.assertEqual(
            baseline_protocol_for_name("phrase_region_grounding", "box_aware_cross_attention_reranker"),
            "same_candidate_strong_reranker",
        )
        self.assertEqual(
            baseline_protocol_for_name("phrase_region_grounding", "lightweight_transvg_style_reranker"),
            "same_candidate_strong_reranker",
        )
        self.assertEqual(
            baseline_protocol_for_name("phrase_region_grounding", "gdino_score_clip_geometry_mlp"),
            "same_candidate_strong_reranker",
        )
        self.assertEqual(
            baseline_protocol_for_name("phrase_region_grounding", "gdino_score_box_aware_cross_attention_reranker"),
            "same_candidate_strong_reranker",
        )
        self.assertEqual(baseline_protocol_for_name("phrase_region_grounding", "ovha_no_cato"), "internal_ovha_ablation")
        self.assertEqual(baseline_protocol_for_name("sentiment_emotion", "Self-MM"), "external_sota_reference_or_reproduction")
        self.assertEqual(
            set(same_feature_probe_names_for_task("sentiment_emotion")),
            {"text_only", "audio_only", "vision_only", "concat_fusion"},
        )
        self.assertTrue(
            {
                "random_valid",
                "train_slot_prior",
                "index_prior_only",
                "text_only",
                "region_only",
                "concat_fusion",
            }.issubset(set(same_feature_probe_names_for_task("phrase_region_grounding")))
        )
        self.assertEqual(
            set(ovha_ablation_names_for_task("sentiment_emotion")),
            {
                "spo_only",
                "lrio_only",
                "ovha_tanso_only",
                "ovha_spo_lrio",
                "ovha_lrio_tanso",
                "ovha_all_candidates_exploratory",
                "ovha_no_rceo",
                "ovha_with_evidence_router",
            },
        )
        self.assertTrue({"MDETR", "GLIP", "GroundingDINO", "GroundingDINO-1.5"}.issubset(set(external_reference_names_for_task("phrase_region_grounding"))))
        self.assertTrue({"TFN", "LMF", "MulT", "MISA", "MAG-BERT", "Self-MM"}.issubset(set(external_reference_names_for_task("sentiment_emotion"))))

    def test_sentiment_ovha_ablation_registry_uses_canonical_non_duplicate_models(self):
        from moat_ovha_torch.models.multimodal.baselines import (
            missing_required_baselines,
            ovha_ablation_names_for_task,
        )

        canonical = {
            "spo_only",
            "lrio_only",
            "ovha_tanso_only",
            "ovha_spo_lrio",
            "ovha_lrio_tanso",
            "ovha_all_candidates_exploratory",
            "ovha_no_rceo",
            "ovha_with_evidence_router",
        }

        ablations = set(ovha_ablation_names_for_task("sentiment_emotion"))

        self.assertEqual(ablations, canonical)
        canonical_baselines = (
            "text_only",
            "audio_only",
            "vision_only",
            "concat_fusion",
            *tuple(canonical),
        )

        self.assertEqual(missing_required_baselines("sentiment_emotion", canonical_baselines), ())

    def test_visual_genome_uses_region_text_public_protocol_contracts(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements
        from moat_ovha_torch.eval.multimodal_statistics import REQUIRED_PUBLIC_METRICS_BY_TASK
        from moat_ovha_torch.models.multimodal.baselines import (
            baseline_names_for_task,
            external_reference_names_for_task,
        )

        region_baselines = baseline_names_for_task("phrase_region_grounding")
        self.assertEqual(baseline_names_for_task("visual_genome"), region_baselines)
        self.assertEqual(
            external_reference_names_for_task("visual_genome"),
            external_reference_names_for_task("phrase_region_grounding"),
        )
        self.assertEqual(
            REQUIRED_PUBLIC_METRICS_BY_TASK["visual_genome"],
            REQUIRED_PUBLIC_METRICS_BY_TASK["phrase_region_grounding"],
        )

        public_entry = validate_public_entry_requirements(
            "visual_genome",
            _complete_controlled_public_entry_report(),
        )
        self.assertTrue(public_entry.ok, public_entry.errors)

        config = MultimodalExperimentConfig.from_mapping(
            {
                "name": "visual_genome_public_smoke",
                "dataset_name": "visual_genome",
                "task_type": "visual_genome",
                "seeds": [1, 2, 3],
                "training_stages": ["T0", "T5"],
                "candidate_names": ["TLEO", "SPO", "LRIO", "CATO"],
                "baseline_names": list(region_baselines),
                "eval_episode_count": 16,
                "require_public_alignment_labels": True,
                "losses_by_stage": {
                    "T0": ["cache_validation"],
                    "T5": ["task_loss", "public_alignment_ce", "candidate_individual_loss"],
                },
                "adapter_params_by_candidate": _valid_adapter_params(),
            }
        )

        self.assertEqual(config.training_stages, ("T0", "T5"))

    def test_external_sota_runbook_keeps_references_separate_from_same_feature_baselines(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "build_external_sota_runbook.py"),
                "--references",
                str(ROOT / "configs" / "multimodal_external_sota_references.json"),
                "--output-dir",
                str(Path(tempfile.mkdtemp()) / "external_sota"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "external_sota_runbook")
        self.assertIn("refcoco", payload["references_by_dataset"])
        self.assertIn("cmu_mosei", payload["references_by_dataset"])
        evidence_types = {
            row["evidence_type"]
            for rows in payload["references_by_dataset"].values()
            for row in rows
        }
        self.assertTrue({"external_reference", "external_reproduction"}.issubset(evidence_types))
        self.assertIn("same-feature", payload["policy"])

    def test_mosei_standard_metric_cli_recomputes_external_reproduction_arrays(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("numpy") is None:
            self.skipTest("torch and numpy are required for MOSEI metric CLI")
        import numpy as np

        tmp_path = Path(tempfile.mkdtemp())
        pred_path = tmp_path / "mult_predictions.npy"
        truth_path = tmp_path / "mult_truths.npy"
        output_path = tmp_path / "mult_standard_metrics.jsonl"
        np.save(pred_path, np.array([[-1.2], [0.2], [1.1], [2.6], [0.0]], dtype=np.float32))
        np.save(truth_path, np.array([[-1.0], [0.0], [1.0], [3.0], [-0.2]], dtype=np.float32))

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "recompute_mosei_standard_metrics.py"),
                "--predictions",
                str(pred_path),
                "--truths",
                str(truth_path),
                "--model",
                "MulT",
                "--seed",
                "1",
                "--split",
                "test",
                "--output-jsonl",
                str(output_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["model"], "MulT")
        self.assertEqual(payload["seed"], 1)
        self.assertEqual(payload["metrics"]["acc2_excl0"], 1.0)
        self.assertAlmostEqual(payload["metrics"]["mae"], 0.22, places=6)
        rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
        self.assertEqual(rows[0]["artifact_type"], "external_mosei_standard_metric")
        self.assertEqual(rows[0]["metrics"]["f1_excl0"], 1.0)

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
                "text_only",
                "region_only",
                "concat_fusion",
                "cato_only",
                "ovha_no_cato",
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
        self.assertIn("concat_fusion", valid.baseline_names)
        self.assertIn("ovha_no_cato", valid.baseline_names)

        invalid = {
            "name": "bad_refcoco",
            "dataset_name": "refcoco",
            "task_type": "phrase_region_grounding",
            "seeds": [1, 2, 3],
            "training_stages": ["T0", "T5"],
            "candidate_names": ["TLEO", "SPO", "LRIO", "CATO"],
            "baseline_names": ["text_only", "concat_fusion"],
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

    def test_public_data_acceptance_builds_cache_and_runs_public_smoke_from_raw_manifest(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for public data acceptance smoke")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw_refcoco"
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "acceptance_artifacts"
            _write_valid_refcoco_raw_manifest(raw_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"), sort_keys=True) + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "accept_public_data.py"),
                    str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                    "--raw-root",
                    str(raw_root),
                    "--cache-root",
                    str(cache_root),
                    "--controlled-report",
                    str(controlled_report_path),
                    "--train-smoke-steps",
                    "1",
                    "--train-split",
                    "train",
                    "--eval-smoke-split",
                    "val",
                    "--artifact-root",
                    str(artifact_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            smoke_payload_path = artifact_root / "public_acceptance_smoke_payload.json"
            smoke_payload_exists = smoke_payload_path.exists()

        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "public_data_acceptance")
        self.assertEqual(payload["dataset_name"], "refcoco")
        self.assertEqual(payload["phases"]["raw_manifest"]["ok"], True)
        self.assertEqual(payload["phases"]["cache"]["ok"], True)
        self.assertEqual(payload["phases"]["public_entry"]["ok"], True)
        self.assertEqual(payload["phases"]["public_smoke"]["ok"], True)
        self.assertEqual(payload["phases"]["public_smoke"]["optimizer_steps"], 1)
        self.assertEqual(payload["phases"]["public_smoke"]["eval_smoke_rows"], 1)
        self.assertTrue(smoke_payload_exists)

    def test_public_data_acceptance_summary_exposes_multiseed_baseline_artifact_bundle(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for public data acceptance smoke")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw_refcoco"
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "acceptance_artifacts"
            _write_valid_refcoco_raw_manifest(raw_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"), sort_keys=True) + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "accept_public_data.py"),
                    str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                    "--raw-root",
                    str(raw_root),
                    "--cache-root",
                    str(cache_root),
                    "--controlled-report",
                    str(controlled_report_path),
                    "--train-smoke-steps",
                    "1",
                    "--train-baseline-smoke-steps",
                    "1",
                    "--train-all-config-seeds",
                    "--train-split",
                    "train",
                    "--eval-smoke-split",
                    "val",
                    "--artifact-root",
                    str(artifact_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            public_smoke = payload["phases"]["public_smoke"]

        self.assertTrue(payload["ok"], payload)
        self.assertEqual(public_smoke["seed_count"], 3)
        self.assertEqual(public_smoke["seeds"], [201, 202, 203])
        self.assertEqual(public_smoke["optimizer_steps"], 3)
        self.assertEqual(public_smoke["eval_smoke_rows"], 3)
        smoke_config = json.loads((ROOT / "configs" / "multimodal_refcoco_public_smoke.json").read_text())
        expected_baseline_rows = public_smoke["seed_count"] * len(smoke_config["baseline_names"])
        self.assertEqual(public_smoke["eval_smoke_baseline_rows"], expected_baseline_rows)
        self.assertEqual(public_smoke["baseline_training_status"], "trained_smoke")
        self.assertEqual(public_smoke["baseline_smoke_training_steps"], 1)
        self.assertEqual(public_smoke["baseline_optimizer_steps"], expected_baseline_rows)
        self.assertIn("smoke_raw_metrics", public_smoke["artifacts"])
        self.assertIn("smoke_baseline_raw_metrics", public_smoke["artifacts"])
        self.assertIn("smoke_statistics_preview", public_smoke["artifacts"])
        self.assertIn("smoke_robustness_summary", public_smoke["artifacts"])
        self.assertIn("smoke_robustness_rows", public_smoke["artifacts"])

    def test_ubuntu_public_acceptance_commands_request_multiseed_and_baseline_smoke(self):
        guide = (ROOT / "docs" / "ubuntu_multimodal_dataset_download.md").read_text()

        for marker in (
            "RefCOCO public acceptance",
            "CMU-MOSEI public acceptance",
        ):
            start = guide.index(marker)
            end = guide.index("```", guide.index("```bash", start) + len("```bash"))
            block = guide[start:end]
            with self.subTest(marker=marker):
                self.assertIn("--train-all-config-seeds", block)
                self.assertIn("--train-baseline-smoke-steps", block)
                self.assertIn("1", block)

    def test_ubuntu_public_dataset_bootstrap_writes_fast_download_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "ovha_public_downloads.sh"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "bootstrap_public_downloads.py"),
                    "--datasets",
                    "refcoco",
                    "cmu_mosei",
                    "--repo-root",
                    "/srv/Operator-Valued-Hyper-Attention",
                    "--download-root",
                    "/data/ovha_datasets/raw_multimodal/_downloads",
                    "--use-hf-mirror",
                    "--include-system-packages",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            payload = json.loads(result.stdout) if result.stdout.strip() else {}
            script = output.read_text() if output.exists() else ""
            executable = bool(output.stat().st_mode & 0o111) if output.exists() else False

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["datasets"], ["refcoco", "cmu_mosei"])
        self.assertEqual(payload["output"], str(output))
        self.assertTrue(executable)
        self.assertIn("NEEDRESTART_MODE=l", script)
        self.assertIn("DEBIAN_FRONTEND=noninteractive", script)
        self.assertIn("HF_XET_HIGH_PERFORMANCE=1", script)
        self.assertIn("HF_ENDPOINT=https://hf-mirror.com", script)
        self.assertIn("aria2c -c -x16 -s16 -k1M", script)
        self.assertIn("http://images.cocodataset.org/zips/train2014.zip", script)
        self.assertIn("https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip", script)
        self.assertIn("mmdatasdk.cmu_mosei", script)
        self.assertIn("CMU-MultimodalSDK", script)

    def test_public_main_runbook_cli_wires_gate_bundles_and_topconf_manifest_without_smoke_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "runbook"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_public_main_runbook.py"),
                    "--output-dir",
                    str(output_dir),
                    "--cache-root",
                    str(tmp_path / "cache"),
                    "--controlled-report",
                    str(tmp_path / "controlled_report.json"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
            runbook_path = output_dir / "public_main_runbook.json"
            commands_path = output_dir / "public_main_commands.sh"
            runbook = json.loads(runbook_path.read_text()) if runbook_path.exists() else {}
            commands = commands_path.read_text() if commands_path.exists() else ""

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "public_main_runbook")
        self.assertTrue(runbook["requires_real_main_metrics"])
        self.assertIn("smoke artifacts must not be used as top-conference main-table inputs", runbook["policy"])
        self.assertEqual(runbook["datasets"], ["refcoco", "cmu_mosei"])
        self.assertEqual(
            runbook["training_configs"]["region_text"]["config"],
            "configs/multimodal_refcoco_public_main.json",
        )
        self.assertEqual(
            runbook["training_configs"]["sentiment"]["config"],
            "configs/multimodal_cmu_mosei_public_main.json",
        )
        self.assertEqual(runbook["training_configs"]["region_text"]["seed_count"], 5)
        self.assertEqual(runbook["training_configs"]["sentiment"]["seed_count"], 5)
        self.assertIn(
            "scripts/multimodal/validate_training_plan.py configs/multimodal_refcoco_public_main.json",
            commands,
        )
        self.assertIn(
            "scripts/multimodal/validate_training_plan.py configs/multimodal_cmu_mosei_public_main.json",
            commands,
        )
        self.assertIn("scripts/multimodal/run_public_main.py configs/multimodal_refcoco_public_main.json", commands)
        self.assertIn("scripts/multimodal/run_public_main.py configs/multimodal_cmu_mosei_public_main.json", commands)
        self.assertIn("${OVHA_PUBLIC_MAIN_TRAIN_STEPS:?", commands)
        self.assertIn("${OVHA_PUBLIC_MAIN_BASELINE_TRAIN_STEPS:?", commands)
        self.assertLess(
            commands.index("scripts/multimodal/run_public_main.py configs/multimodal_refcoco_public_main.json"),
            commands.index("scripts/multimodal/validate_public_main_artifacts.py --config configs/multimodal_refcoco_public_main.json"),
        )
        self.assertIn("scripts/multimodal/validate_public_main_artifacts.py --config configs/multimodal_refcoco_public_main.json", commands)
        self.assertIn("scripts/multimodal/validate_public_main_artifacts.py --config configs/multimodal_cmu_mosei_public_main.json", commands)
        self.assertIn("scripts/multimodal/validate_cache.py", commands)
        self.assertIn("scripts/multimodal/build_public_gate_report.py region_text", commands)
        self.assertIn("scripts/multimodal/build_public_gate_report.py sentiment", commands)
        self.assertIn("scripts/multimodal/build_topconf_entry_manifest.py", commands)
        self.assertIn("--validate", commands)
        self.assertIn("--cache-target refcoco", commands)
        self.assertIn("--cache-target cmu_mosei", commands)
        self.assertIn("outputs/multimodal/refcoco_main/raw_metrics.jsonl", commands)
        self.assertIn("outputs/multimodal/cmu_mosei_main/raw_metrics.jsonl", commands)
        self.assertNotIn("public_smoke_raw_metrics", commands)
        self.assertNotIn("public_smoke_statistics_preview", commands)
        self.assertEqual(runbook["required_real_inputs"]["region_text"]["split"], "testA")
        self.assertEqual(runbook["required_real_inputs"]["sentiment"]["split"], "test")

    def test_public_main_artifact_validator_requires_config_seed_model_coverage_and_rejects_smoke(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = ROOT / "configs" / "multimodal_refcoco_public_main.json"
            config = MultimodalExperimentConfig.from_file(config_path)
            raw_metrics = tmp_path / "raw_metrics.jsonl"
            diagnostics = tmp_path / "diagnostics.jsonl"
            robustness_rows = tmp_path / "robustness_rows.jsonl"
            rows = _public_main_metric_rows(config_path, raw_metrics)
            diagnostics_rows = [
                {
                    "artifact_type": "public_main_diagnostics",
                    "dataset": "refcoco",
                    "task": "phrase_region_grounding",
                    "split": "testA",
                    "seed": seed,
                    "stage": "T5_eval",
                    "router_load_by_candidate": {"TLEO": 0.2, "SPO": 0.1, "LRIO": 0.2, "CATO": 0.5},
                    "stackability_passed": True,
                }
                for seed in [201, 202, 203, 204, 205]
            ]
            robustness = [
                {
                    "artifact_type": "public_main_robustness_row",
                    "dataset": "refcoco",
                    "task": "phrase_region_grounding",
                    "split": "testA",
                    "seed": 201,
                    "model": config.main_model_name,
                    "corruption_type": "image_blur",
                    "corruption_strength": 0.4,
                    "score": 0.74,
                },
                {
                    "artifact_type": "public_main_robustness_row",
                    "dataset": "refcoco",
                    "task": "phrase_region_grounding",
                    "split": "testA",
                    "seed": 201,
                    "model": "concat_fusion",
                    "corruption_type": "image_blur",
                    "corruption_strength": 0.4,
                    "score": 0.70,
                },
            ]
            raw_metrics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
            diagnostics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in diagnostics_rows) + "\n")
            robustness_rows.write_text("\n".join(json.dumps(row, sort_keys=True) for row in robustness) + "\n")

            ok_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "validate_public_main_artifacts.py"),
                    "--config",
                    str(config_path),
                    "--raw-metrics",
                    str(raw_metrics),
                    "--diagnostics",
                    str(diagnostics),
                    "--robustness-rows",
                    str(robustness_rows),
                    "--external-sota-references",
                    str(ROOT / "configs" / "multimodal_external_sota_references.json"),
                    "--split",
                    "testA",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            rows[0] = {
                **rows[0],
                "artifact_type": "public_smoke_raw_metric",
                "evidence_scope": "public_smoke_only_not_topconf_main_table",
                "not_topconf_main_table": True,
            }
            rows = [row for row in rows if not (row["model"] == config.main_model_name and row["seed"] == 205)]
            raw_metrics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
            bad_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "validate_public_main_artifacts.py"),
                    "--config",
                    str(config_path),
                    "--raw-metrics",
                    str(raw_metrics),
                    "--diagnostics",
                    str(diagnostics),
                    "--robustness-rows",
                    str(robustness_rows),
                    "--split",
                    "testA",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        ok_payload = json.loads(ok_result.stdout)
        self.assertEqual(ok_result.returncode, 0, ok_result.stdout + ok_result.stderr)
        self.assertTrue(ok_payload["ok"], ok_payload)
        self.assertEqual(ok_payload["coverage"]["seed_count"], 5)
        self.assertIn("concat_fusion", ok_payload["coverage"]["models"])
        self.assertNotIn("GroundingDINO", ok_payload["coverage"]["models"])
        self.assertTrue(ok_payload["external_sota_references"]["provided"])
        self.assertGreaterEqual(ok_payload["external_sota_references"]["reference_count"], 10)
        self.assertEqual(ok_payload["mode"], "public_main_artifact_validation")

        bad_payload = json.loads(bad_result.stdout)
        self.assertEqual(bad_result.returncode, 2)
        self.assertFalse(bad_payload["ok"])
        joined = "\n".join(bad_payload["errors"])
        self.assertIn("not_topconf_main_table rows cannot enter public main artifacts", joined)
        self.assertIn(f"missing configured seed coverage for model {config.main_model_name}: 205", joined)

    def test_public_main_runner_writes_non_smoke_main_artifacts_and_passes_preflight(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for public main runner smoke")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "refcoco_main"
            controlled_report_path = tmp_path / "controlled_report.json"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"), sort_keys=True) + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "run_public_main.py"),
                    str(ROOT / "configs" / "multimodal_refcoco_public_main.json"),
                    "--cache-root",
                    str(cache_root),
                    "--controlled-report",
                    str(controlled_report_path),
                    "--artifact-root",
                    str(artifact_root),
                    "--train-steps",
                    "1",
                    "--baseline-train-steps",
                    "1",
                    "--train-split",
                    "train",
                    "--eval-split",
                    "testA",
                    "--d-model",
                    "8",
                    "--memory-tokens",
                    "1",
                    "--device",
                    "cpu",
                    "--progress-interval",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
            raw_metrics = artifact_root / "raw_metrics.jsonl"
            diagnostics = artifact_root / "diagnostics.jsonl"
            robustness_rows = artifact_root / "robustness_rows.jsonl"
            raw_rows = [
                json.loads(line)
                for line in raw_metrics.read_text().splitlines()
                if line.strip()
            ] if raw_metrics.exists() else []
            preflight = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "validate_public_main_artifacts.py"),
                    "--config",
                    str(ROOT / "configs" / "multimodal_refcoco_public_main.json"),
                    "--raw-metrics",
                    str(raw_metrics),
                    "--diagnostics",
                    str(diagnostics),
                    "--robustness-rows",
                    str(robustness_rows),
                    "--split",
                    "testA",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "public_main_training")
        self.assertIn("[public-main:start]", result.stderr)
        self.assertIn(
            "[public-main:train] seed=201 model=ovha_refcoco_prso_sro_tleo_cato_primary step=1/1",
            result.stderr,
        )
        self.assertIn(
            "[public-main:model:done] seed=201 model=ovha_refcoco_prso_sro_tleo_cato_primary",
            result.stderr,
        )
        self.assertIn("[public-main:model:start] seed=201 model=text_only", result.stderr)
        self.assertEqual(payload["seed_count"], 5)
        self.assertEqual(payload["artifacts"]["raw_metrics"]["path"], str(raw_metrics))
        config_payload = json.loads((ROOT / "configs" / "multimodal_refcoco_public_main.json").read_text())
        self.assertEqual(len(raw_rows), payload["seed_count"] * (1 + len(config_payload["baseline_names"])))
        self.assertEqual({row["artifact_type"] for row in raw_rows}, {"public_main_raw_metric"})
        self.assertEqual({row["evidence_scope"] for row in raw_rows}, {"public_main_table"})
        self.assertEqual({row["public_metrics_scope"] for row in raw_rows}, {"public_main_metrics"})
        rows_by_model = {row["model"]: row for row in raw_rows if row["seed"] == 201}
        self.assertEqual(
            rows_by_model["text_only"]["model_protocol"],
            "same_feature_sanity_probe_public_main_v1",
        )
        self.assertEqual(
            rows_by_model["concat_fusion"]["model_protocol"],
            "same_feature_sanity_probe_public_main_v1",
        )
        self.assertEqual(
            rows_by_model["cato_only"]["model_protocol"],
            "internal_ovha_ablation_public_main_v1",
        )
        self.assertEqual(
            rows_by_model["ovha_no_cato"]["model_protocol"],
            "internal_ovha_ablation_public_main_v1",
        )
        self.assertEqual(
            rows_by_model["ovha_no_rceo"]["model_protocol"],
            "internal_ovha_ablation_public_main_v1",
        )
        self.assertEqual(
            rows_by_model["ovha_no_evidence_router"]["model_protocol"],
            "internal_ovha_ablation_public_main_v1",
        )
        sanity_parameter_count = rows_by_model["text_only"]["parameter_count"]
        self.assertNotEqual(rows_by_model["cato_only"]["parameter_count"], sanity_parameter_count)
        self.assertNotEqual(rows_by_model["ovha_no_cato"]["parameter_count"], sanity_parameter_count)
        self.assertNotEqual(rows_by_model["ovha_no_rceo"]["parameter_count"], sanity_parameter_count)
        self.assertNotEqual(rows_by_model["ovha_no_evidence_router"]["parameter_count"], sanity_parameter_count)
        self.assertNotIn("smoke", json.dumps(raw_rows, sort_keys=True).lower())
        self.assertNotIn("not_topconf", json.dumps(raw_rows, sort_keys=True).lower())
        self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)

    def test_public_main_runner_can_run_single_pilot_seed_without_downgrading_main_config(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for public main runner smoke")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "refcoco_pilot"
            controlled_report_path = tmp_path / "controlled_report.json"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"), sort_keys=True) + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "run_public_main.py"),
                    str(ROOT / "configs" / "multimodal_refcoco_public_main.json"),
                    "--cache-root",
                    str(cache_root),
                    "--controlled-report",
                    str(controlled_report_path),
                    "--artifact-root",
                    str(artifact_root),
                    "--train-steps",
                    "1",
                    "--baseline-train-steps",
                    "1",
                    "--train-split",
                    "train",
                    "--eval-split",
                    "testA",
                    "--d-model",
                    "8",
                    "--memory-tokens",
                    "1",
                    "--device",
                    "cpu",
                    "--progress-interval",
                    "1",
                    "--pilot-seed",
                    "203",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
            raw_metrics = artifact_root / "raw_metrics.jsonl"
            raw_rows = [
                json.loads(line)
                for line in raw_metrics.read_text().splitlines()
                if line.strip()
            ] if raw_metrics.exists() else []

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["ok"], payload)
        self.assertTrue(payload["pilot_seed_subset"])
        self.assertEqual(payload["configured_seed_count"], 5)
        self.assertEqual(payload["seed_count"], 1)
        self.assertEqual(payload["seeds"], [203])
        self.assertIn(
            "[public-main:train] seed=203 model=ovha_refcoco_prso_sro_tleo_cato_primary step=1/1",
            result.stderr,
        )
        self.assertNotIn(
            "[public-main:train] seed=201 model=ovha_refcoco_prso_sro_tleo_cato_primary step=1/1",
            result.stderr,
        )
        config_payload = json.loads((ROOT / "configs" / "multimodal_refcoco_public_main.json").read_text())
        self.assertEqual(len(raw_rows), 1 + len(config_payload["baseline_names"]))
        self.assertEqual({row["seed"] for row in raw_rows}, {203})

    def test_public_main_internal_ovha_ablation_variant_mapping_is_structural(self):
        module = importlib.import_module("scripts.multimodal.run_public_main")

        self.assertEqual(module._ovha_variant_kwargs("cato_only"), {"candidate_names": ("CATO",)})
        self.assertEqual(module._ovha_variant_kwargs("ovha_no_cato"), {"candidate_names": ("TLEO", "SPO", "LRIO")})
        self.assertEqual(module._ovha_variant_kwargs("ovha_no_lrio"), {"candidate_names": ("TLEO", "SPO", "CATO")})
        self.assertEqual(module._ovha_variant_kwargs("ovha_no_spo"), {"candidate_names": ("TLEO", "LRIO", "CATO")})
        self.assertEqual(module._ovha_variant_kwargs("ovha_no_lrio", ("SPO", "LRIO")), {"candidate_names": ("SPO",)})
        self.assertEqual(module._ovha_variant_kwargs("ovha_no_spo", ("SPO", "LRIO")), {"candidate_names": ("LRIO",)})
        self.assertEqual(module._ovha_variant_kwargs("spo_only", ("SPO", "LRIO", "TANSO")), {"candidate_names": ("SPO",)})
        self.assertEqual(module._ovha_variant_kwargs("lrio_only", ("SPO", "LRIO", "TANSO")), {"candidate_names": ("LRIO",)})
        self.assertEqual(module._ovha_variant_kwargs("ovha_tanso_only", ("SPO", "LRIO", "TANSO")), {"candidate_names": ("TANSO",)})
        self.assertEqual(module._ovha_variant_kwargs("ovha_no_tanso", ("SPO", "LRIO", "TANSO")), {"candidate_names": ("SPO",)})
        self.assertEqual(
            module._ovha_variant_kwargs("ovha_spo_lrio", ("SPO", "LRIO", "TANSO")),
            {
                "candidate_names": ("SPO", "LRIO"),
                "composition_mode": "base_plus_residual",
                "base_candidate": "SPO",
                "residual_candidates": ("LRIO",),
            },
        )
        self.assertEqual(
            module._ovha_variant_kwargs("ovha_spo_tanso", ("SPO", "LRIO", "TANSO")),
            {
                "candidate_names": ("SPO", "TANSO"),
                "composition_mode": "base_plus_residual",
                "base_candidate": "SPO",
                "residual_candidates": ("TANSO",),
            },
        )
        self.assertEqual(
            module._ovha_variant_kwargs("ovha_lrio_tanso", ("SPO", "LRIO", "TANSO")),
            {
                "candidate_names": ("LRIO", "TANSO"),
                "composition_mode": "base_plus_residual",
                "base_candidate": "TANSO",
                "residual_candidates": ("LRIO",),
            },
        )
        self.assertEqual(
            module._ovha_variant_kwargs("ovha_all_candidates_exploratory", ("SPO", "LRIO", "TANSO")),
            {
                "candidate_names": ("SPO", "LRIO", "TANSO"),
                "composition_mode": "base_plus_residual",
                "base_candidate": "SPO",
                "residual_candidates": ("LRIO", "TANSO"),
            },
        )
        self.assertEqual(module._ovha_variant_kwargs("ovha_no_rceo"), {"use_reliability_prior": False})
        self.assertEqual(module._ovha_variant_kwargs("ovha_no_evidence_router"), {"use_evidence_router": False})
        self.assertEqual(module._ovha_variant_kwargs("ovha_with_evidence_router"), {"use_evidence_router": True})
        with self.assertRaisesRegex(ValueError, "unknown OVHA ablation baseline"):
            module._ovha_variant_kwargs("concat_fusion")

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

    def test_public_entry_accepts_trained_controlled_diagnostics_artifact_type(self):
        from moat_ovha_torch.data.multimodal.cache_schema import file_sha256
        from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements

        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp) / "controlled_artifacts"
            report_payload = _complete_controlled_public_entry_report(artifact_root)
            diagnostics_path = Path(report_payload["evidence_artifacts"]["diagnostics_report"]["path"])
            diagnostics_rows = [
                {**json.loads(line), "artifact_type": "controlled_training_diagnostics"}
                for line in diagnostics_path.read_text().splitlines()
                if line.strip()
            ]
            diagnostics_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in diagnostics_rows) + "\n"
            )
            report_payload["evidence_artifacts"]["diagnostics_report"]["sha256"] = file_sha256(diagnostics_path)

            report = validate_public_entry_requirements(
                "phrase_region_grounding",
                report_payload,
                require_artifact_files=True,
            )

        self.assertTrue(report.ok, report.errors)

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

    def test_public_smoke_runner_runs_real_train_smoke_after_entry_validation(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for public training smoke")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "public_training_artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"), sort_keys=True) + "\n"
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                "--cache-root",
                str(cache_root),
                "--controlled-report",
                str(controlled_report_path),
                "--train-smoke-steps",
                "1",
                "--train-split",
                "train",
                "--artifact-root",
                str(artifact_root),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            metrics_path = Path(payload["training"]["artifacts"]["metrics"]["path"])
            diagnostics_path = Path(payload["training"]["artifacts"]["diagnostics"]["path"])
            diagnostics_summary_path = Path(payload["training"]["artifacts"]["diagnostics_summary"]["path"])
            metrics_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            diagnostics_rows = [json.loads(line) for line in diagnostics_path.read_text().splitlines() if line.strip()]
            diagnostics_summary = json.loads(diagnostics_summary_path.read_text())

        training = payload["training"]
        stages = {stage["stage"]: stage for stage in training["stage_history"]}
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "public_trained_smoke")
        self.assertEqual(training["config_name"], "multimodal_refcoco_public_smoke")
        self.assertEqual(training["train_split"], "train")
        self.assertEqual(training["validated_training_stages"], ["T0", "T5"])
        self.assertEqual(training["optimizer_steps"], 1)
        self.assertGreater(training["parameter_l2_delta"], 0.0)
        self.assertGreater(training["max_grad_norm"], 0.0)
        self.assertEqual(stages["T0"]["loss_names_observed"], ["cache_validation"])
        self.assertEqual(stages["T0"]["optimizer_steps"], 0)
        self.assertEqual(
            stages["T5"]["loss_names_observed"],
            [
                "residual_gate_utility_loss",
                "router_marginal_utility",
                "task_loss",
            ],
        )
        self.assertEqual(len(metrics_rows), 1)
        self.assertEqual(metrics_rows[0]["stage"], "T5")
        self.assertEqual(metrics_rows[0]["split"], "train")
        self.assertNotIn("public_alignment_ce", metrics_rows[0])
        self.assertGreater(metrics_rows[0]["task_loss"], 0.0)
        self.assertEqual(len(diagnostics_rows), 1)
        self.assertEqual(diagnostics_summary["artifact_type"], "public_smoke_diagnostics_summary")
        self.assertEqual(diagnostics_summary["source_rows_path"], str(diagnostics_path))
        self.assertEqual(diagnostics_summary["row_count"], 1)
        self.assertEqual(diagnostics_summary["valid_row_count"], 1)
        self.assertEqual(diagnostics_summary["errors"], [])
        diagnostic = diagnostics_rows[0]
        self.assertEqual(diagnostic["artifact_type"], "public_training_diagnostics")
        self.assertEqual(diagnostic["stage"], "T5")
        self.assertEqual(diagnostic["split"], "train")
        self.assertEqual(diagnostic["task"], "phrase_region_grounding")
        self.assertTrue(diagnostic["stackability_passed"])
        self.assertEqual(set(diagnostic["router_logit_parts"]), {"memory", "evidence", "reliability"})
        self.assertEqual(set(diagnostic["router_load_by_candidate"]), {"PRSO", "SRO", "TLEO", "CATO"})
        self.assertEqual(set(diagnostic["candidate_loss"]), {"PRSO", "SRO", "TLEO", "CATO"})
        self.assertEqual(set(diagnostic["memory_slot_norm"]), {"PRSO", "SRO", "TLEO", "CATO"})
        self.assertIn("PRSO_alignment_temperature", diagnostic["adapter_params"])
        self.assertIn("SRO_scale", diagnostic["adapter_params"])
        self.assertIn("CATO_alignment_temperature", diagnostic["adapter_params"])
        self.assertIn("candidate_loss", diagnostic["candidate_diagnostics"]["CATO"])
        self.assertIn("corruption_response", diagnostic["candidate_diagnostics"]["RCEO"])

    def test_sentiment_public_train_smoke_maps_missing_modality_to_rceo_reliability(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for public training smoke")

        import numpy as np

        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256
        from moat_ovha_torch.eval.multimodal_statistics import SENTIMENT_REQUIRED_PUBLIC_METRICS

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "sentiment_training_artifacts"
            _write_valid_cmu_mosei_public_cache(cache_root)
            layout = MultimodalCacheLayout(cache_root, "cmu_mosei", "v0.1")
            missing_mask_path = layout.root / "supervision" / "missing_modality_mask_train.npy"
            np.save(missing_mask_path, np.array([[False, True, False]], dtype=bool))
            val_missing_mask_path = layout.root / "supervision" / "missing_modality_mask_val.npy"
            np.save(val_missing_mask_path, np.array([[False, True, False]], dtype=bool))
            checksums_path = layout.root / "checksums.json"
            checksums = json.loads(checksums_path.read_text())
            checksums[str(missing_mask_path.relative_to(layout.root))] = file_sha256(missing_mask_path)
            checksums[str(val_missing_mask_path.relative_to(layout.root))] = file_sha256(val_missing_mask_path)
            checksums_path.write_text(json.dumps(checksums, sort_keys=True) + "\n")
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"), sort_keys=True) + "\n"
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_cmu_mosei_public_smoke.json"),
                "--cache-root",
                str(cache_root),
                "--controlled-report",
                str(controlled_report_path),
                "--train-smoke-steps",
                "1",
                "--train-split",
                "train",
                "--eval-smoke-split",
                "val",
                "--artifact-root",
                str(artifact_root),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            metrics_path = Path(payload["training"]["artifacts"]["metrics"]["path"])
            smoke_raw_metrics_path = Path(payload["training"]["artifacts"]["smoke_raw_metrics"]["path"])
            smoke_baseline_metrics_path = Path(
                payload["training"]["artifacts"]["smoke_baseline_raw_metrics"]["path"]
            )
            smoke_robustness_rows_path = Path(payload["training"]["artifacts"]["smoke_robustness_rows"]["path"])
            smoke_robustness_summary_path = Path(
                payload["training"]["artifacts"]["smoke_robustness_summary"]["path"]
            )
            eval_diagnostics_path = Path(payload["training"]["artifacts"]["eval_diagnostics"]["path"])
            metrics_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            eval_diagnostics = [json.loads(line) for line in eval_diagnostics_path.read_text().splitlines() if line.strip()]
            smoke_raw_rows = [
                json.loads(line) for line in smoke_raw_metrics_path.read_text().splitlines() if line.strip()
            ]
            smoke_baseline_rows = [
                json.loads(line) for line in smoke_baseline_metrics_path.read_text().splitlines() if line.strip()
            ]
            smoke_robustness_rows = [
                json.loads(line) for line in smoke_robustness_rows_path.read_text().splitlines() if line.strip()
            ]
            smoke_robustness_summary = json.loads(smoke_robustness_summary_path.read_text())

        self.assertEqual(len(metrics_rows), 1)
        row = metrics_rows[0]
        self.assertEqual(row["stage"], "T5")
        self.assertEqual(row["split"], "train")
        self.assertEqual(row["rceo_modality_order"], ["text", "audio", "vision"])
        self.assertEqual(row["rceo_modality_reliability"], {"text": 1.0, "audio": 0.0, "vision": 1.0})
        self.assertAlmostEqual(row["rceo_reliability_mean"], 2.0 / 3.0)
        self.assertAlmostEqual(row["rceo_corruption_response"], 1.0 / 3.0)
        self.assertEqual(len(eval_diagnostics), 1)
        public_diagnostics = eval_diagnostics[0]["public_diagnostics"]
        self.assertIn("lrio_rank_entropy_by_modality_pair", public_diagnostics)
        self.assertIn("spo_prototype_usage", public_diagnostics)
        self.assertIn("heuristic_debug", public_diagnostics)
        self.assertIn("omitted_spo_prototype_load_by_emotion_class", public_diagnostics["heuristic_debug"])
        self.assertIn("rceo_reliability_shift_under_missing_noisy_modality", public_diagnostics)
        self.assertIn("router_load_by_condition", public_diagnostics)
        self.assertEqual(len(smoke_raw_rows), 1)
        smoke_raw = smoke_raw_rows[0]
        self.assertEqual(smoke_raw["task"], "sentiment_emotion")
        self.assertEqual(set(smoke_raw["public_metrics"]), set(SENTIMENT_REQUIRED_PUBLIC_METRICS))
        self.assertEqual(
            smoke_raw["public_metrics_scope"],
            "sentiment_emotion_smoke_real_metrics_not_topconf_main_table",
        )
        self.assertIn("clean", smoke_raw["public_metrics"]["router_load_by_corruption_type"])
        self.assertEqual(
            set(smoke_raw["public_metrics"]["router_load_by_corruption_type"]["clean"]),
            set(json.loads((ROOT / "configs" / "multimodal_cmu_mosei_public_smoke.json").read_text())["candidate_names"]),
        )
        self.assertGreaterEqual(smoke_raw["public_metrics"]["rceo_reliability_calibration"]["bin_count"], 1)
        self.assertEqual(len(smoke_baseline_rows), len(payload["baselines"]))
        for baseline_row in smoke_baseline_rows:
            self.assertEqual(baseline_row["task"], "sentiment_emotion")
            self.assertEqual(set(baseline_row["public_metrics"]), set(SENTIMENT_REQUIRED_PUBLIC_METRICS))
            self.assertEqual(
                baseline_row["public_metrics_scope"],
                "sentiment_emotion_smoke_real_metrics_not_topconf_main_table",
            )
            self.assertIn("clean", baseline_row["public_metrics"]["router_load_by_corruption_type"])
        self.assertEqual(len(smoke_robustness_rows), 2 * (1 + len(payload["baselines"])))
        ovha_robustness = [row for row in smoke_robustness_rows if row["model"] == "ovha_full"]
        self.assertEqual({row["corruption_type"] for row in ovha_robustness}, {"clean_smoke", "missing_modality_smoke"})
        corrupted = next(row for row in ovha_robustness if row["corruption_type"] == "missing_modality_smoke")
        self.assertEqual(corrupted["artifact_type"], "public_smoke_robustness_row")
        self.assertEqual(corrupted["evidence_scope"], "smoke_robustness_preview_only_not_topconf_gate")
        self.assertTrue(corrupted["not_topconf_main_table"])
        self.assertEqual(corrupted["missing_modalities"], ["modality"])
        self.assertGreater(corrupted["corruption_strength"], 0.0)
        self.assertEqual(set(corrupted["router_load_by_candidate"]), {"TLEO", "SPO", "LRIO", "CATO"})
        self.assertEqual(set(corrupted["candidate_loss"]), {"TLEO", "SPO", "LRIO", "CATO"})
        self.assertGreaterEqual(corrupted["rceo_reliability"], 0.0)
        self.assertLessEqual(corrupted["rceo_reliability"], 1.0)
        self.assertEqual(smoke_robustness_summary["artifact_type"], "public_smoke_robustness_summary")
        self.assertEqual(
            smoke_robustness_summary["evidence_scope"],
            "smoke_robustness_preview_only_not_topconf_gate",
        )
        self.assertTrue(smoke_robustness_summary["not_topconf_main_table"])
        self.assertEqual(smoke_robustness_summary["source_rows_path"], str(smoke_robustness_rows_path))
        self.assertEqual(smoke_robustness_summary["full_model"], "ovha_full")
        self.assertEqual(smoke_robustness_summary["baseline_model"], "concat_fusion")
        self.assertFalse(smoke_robustness_summary["required_stress_coverage"]["passed"])
        self.assertNotIn("missing_audio", smoke_robustness_summary["required_stress_coverage"]["observed"])
        self.assertGreaterEqual(smoke_robustness_summary["rceo_reliability_calibration"]["bin_count"], 1)
        self.assertIn(
            "not valid top-conference robustness evidence",
            smoke_robustness_summary["evidence_limitations"],
        )

    def test_robustness_stress_smoke_cli_generates_step6_rows_from_public_raw_metrics(self):
        from moat_ovha_torch.eval.multimodal_robustness import DEFAULT_REQUIRED_STRESS_TARGETS

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_metrics = tmp_path / "raw_metrics.jsonl"
            output_rows = tmp_path / "robustness_rows.jsonl"
            output_summary = tmp_path / "robustness_summary.json"
            rows = _robustness_stress_raw_metric_rows()
            raw_metrics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "run_robustness_stress_smoke.py"),
                    str(ROOT / "configs" / "multimodal_robustness_smoke.json"),
                    "--raw-metrics",
                    str(raw_metrics),
                    "--output-rows",
                    str(output_rows),
                    "--output-summary",
                    str(output_summary),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            stress_rows = [json.loads(line) for line in output_rows.read_text().splitlines() if line.strip()]
            summary = json.loads(output_summary.read_text())

        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "public_robustness_stress_smoke")
        self.assertEqual(payload["artifacts"]["robustness_rows"]["path"], str(output_rows))
        self.assertEqual(payload["artifacts"]["robustness_summary"]["path"], str(output_summary))
        self.assertEqual(payload["stress_family_count"], len(DEFAULT_REQUIRED_STRESS_TARGETS))
        self.assertEqual(
            len(stress_rows),
            len(_robustness_stress_raw_metric_rows()) * (1 + len(DEFAULT_REQUIRED_STRESS_TARGETS)),
        )
        self.assertTrue(all(row["artifact_type"] == "public_robustness_stress_smoke_row" for row in stress_rows))
        self.assertTrue(all(row["not_topconf_main_table"] for row in stress_rows))
        self.assertTrue(
            all(row["evidence_scope"] == "robustness_stress_smoke_protocol_only_not_topconf_gate" for row in stress_rows)
        )
        observed = set(summary["required_stress_coverage"]["observed"])
        self.assertTrue(set(DEFAULT_REQUIRED_STRESS_TARGETS).issubset(observed))
        self.assertTrue(summary["required_stress_coverage"]["passed"], summary["required_stress_coverage"])
        self.assertEqual(summary["robustness_significance"]["common_seed_count"], 3)
        self.assertGreater(summary["robustness_significance"]["drop_delta"], 0.0)
        self.assertIn("hard_negative_caption_mismatch", {row["corruption_type"] for row in stress_rows})
        mismatch_row = next(row for row in stress_rows if row["corruption_type"] == "hard_negative_caption_mismatch")
        self.assertIn("mismatch_source_id", mismatch_row)
        missing_audio = next(row for row in stress_rows if row["corruption_type"] == "missing_audio")
        self.assertEqual(missing_audio["missing_modalities"], ["audio"])
        self.assertEqual(set(missing_audio["router_load_by_candidate"]), {"TLEO", "SPO", "LRIO", "CATO"})
        self.assertEqual(set(missing_audio["candidate_loss"]), {"TLEO", "SPO", "LRIO", "CATO"})
        self.assertEqual(summary["artifact_type"], "public_robustness_stress_smoke_summary")
        self.assertEqual(summary["source_rows_path"], str(output_rows))
        self.assertIn(
            "not valid top-conference robustness evidence",
            summary["evidence_limitations"],
        )

    def test_public_smoke_runner_can_execute_all_config_seeds_for_dev_multiseed_evidence(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for public training smoke")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "public_training_artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"), sort_keys=True) + "\n"
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                "--cache-root",
                str(cache_root),
                "--controlled-report",
                str(controlled_report_path),
                "--train-smoke-steps",
                "1",
                "--train-split",
                "train",
                "--artifact-root",
                str(artifact_root),
                "--train-all-config-seeds",
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            metrics_path = Path(payload["training"]["artifacts"]["metrics"]["path"])
            diagnostics_path = Path(payload["training"]["artifacts"]["diagnostics"]["path"])
            metrics_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            diagnostics_rows = [json.loads(line) for line in diagnostics_path.read_text().splitlines() if line.strip()]

        training = payload["training"]
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(training["seeds"], [201, 202, 203])
        self.assertEqual(training["optimizer_steps"], 3)
        self.assertEqual(training["optimizer_steps_per_seed"], 1)
        self.assertGreater(training["parameter_l2_delta_min"], 0.0)
        self.assertEqual([row["seed"] for row in metrics_rows], [201, 202, 203])
        self.assertEqual([row["seed"] for row in diagnostics_rows], [201, 202, 203])

    def test_public_smoke_runner_evaluates_heldout_split_after_training(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for public training smoke")
        from moat_ovha_torch.eval.multimodal_statistics import REGION_TEXT_REQUIRED_PUBLIC_METRICS

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "public_training_artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"), sort_keys=True) + "\n"
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                "--cache-root",
                str(cache_root),
                "--controlled-report",
                str(controlled_report_path),
                "--train-smoke-steps",
                "1",
                "--train-split",
                "train",
                "--eval-smoke-split",
                "val",
                "--artifact-root",
                str(artifact_root),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            eval_metrics_path = Path(payload["training"]["artifacts"]["eval_metrics"]["path"])
            eval_diagnostics_path = Path(payload["training"]["artifacts"]["eval_diagnostics"]["path"])
            smoke_raw_metrics_path = Path(payload["training"]["artifacts"]["smoke_raw_metrics"]["path"])
            smoke_baseline_metrics_path = Path(
                payload["training"]["artifacts"]["smoke_baseline_raw_metrics"]["path"]
            )
            eval_diagnostics_summary_path = Path(
                payload["training"]["artifacts"]["eval_diagnostics_summary"]["path"]
            )
            smoke_statistics_preview_path = Path(
                payload["training"]["artifacts"]["smoke_statistics_preview"]["path"]
            )
            eval_rows = [json.loads(line) for line in eval_metrics_path.read_text().splitlines() if line.strip()]
            eval_diagnostics = [json.loads(line) for line in eval_diagnostics_path.read_text().splitlines() if line.strip()]
            smoke_raw_rows = [
                json.loads(line) for line in smoke_raw_metrics_path.read_text().splitlines() if line.strip()
            ]
            smoke_baseline_rows = [
                json.loads(line) for line in smoke_baseline_metrics_path.read_text().splitlines() if line.strip()
            ]
            eval_diagnostics_summary = json.loads(eval_diagnostics_summary_path.read_text())
            smoke_statistics_preview = json.loads(smoke_statistics_preview_path.read_text())

        training = payload["training"]
        self.assertEqual(training["eval_smoke_split"], "val")
        self.assertEqual(training["eval_smoke_rows"], 1)
        self.assertEqual(training["eval_smoke_baseline_rows"], len(payload["baselines"]))
        self.assertEqual(len(eval_rows), 1)
        self.assertEqual(eval_rows[0]["stage"], "T5_eval")
        self.assertEqual(eval_rows[0]["split"], "val")
        self.assertEqual(eval_rows[0]["seed"], 201)
        self.assertGreaterEqual(eval_rows[0]["total_loss"], 0.0)
        self.assertEqual(len(eval_diagnostics), 1)
        self.assertEqual(eval_diagnostics[0]["artifact_type"], "public_eval_diagnostics")
        self.assertEqual(eval_diagnostics[0]["stage"], "T5_eval")
        self.assertEqual(eval_diagnostics[0]["split"], "val")
        self.assertTrue(eval_diagnostics[0]["stackability_passed"])
        public_diagnostics = eval_diagnostics[0]["public_diagnostics"]
        self.assertIn("cato_router_load_by_phrase_type", public_diagnostics)
        self.assertIn("no_cato_delta_by_object_size", public_diagnostics)
        self.assertIn("no_cato_delta_by_phrase_length", public_diagnostics)
        self.assertIn("rceo_reliability_shift_under_blurred_regions", public_diagnostics)
        self.assertEqual(eval_diagnostics_summary["artifact_type"], "public_smoke_diagnostics_summary")
        self.assertEqual(eval_diagnostics_summary["source_rows_path"], str(eval_diagnostics_path))
        self.assertEqual(eval_diagnostics_summary["row_count"], 1)
        self.assertEqual(eval_diagnostics_summary["valid_row_count"], 1)
        self.assertEqual(eval_diagnostics_summary["errors"], [])
        self.assertEqual(len(smoke_raw_rows), 1)
        smoke_raw = smoke_raw_rows[0]
        self.assertEqual(smoke_raw["artifact_type"], "public_smoke_raw_metric")
        self.assertEqual(smoke_raw["evidence_scope"], "public_smoke_only_not_topconf_main_table")
        self.assertTrue(smoke_raw["not_topconf_main_table"])
        self.assertEqual(smoke_raw["task"], "phrase_region_grounding")
        self.assertEqual(smoke_raw["dataset"], "refcoco")
        self.assertEqual(smoke_raw["model"], "ovha_full")
        self.assertEqual(smoke_raw["metric_name"], "heldout_task_loss_smoke")
        self.assertEqual(smoke_raw["split"], "val")
        self.assertEqual(smoke_raw["seed"], 201)
        self.assertFalse(smoke_raw["higher_is_better"])
        self.assertAlmostEqual(smoke_raw["score"], eval_rows[0]["task_loss"], places=7)
        self.assertEqual(smoke_raw["raw_metric_path"], str(smoke_raw_metrics_path))
        self.assertIn("not a same-feature baseline comparison", smoke_raw["evidence_limitations"])
        self.assertGreater(smoke_raw["parameter_count"], 0)
        self.assertEqual(smoke_raw["training_steps"], 1)
        self.assertEqual(
            smoke_raw["frozen_feature_extractor_version"],
            {"region": "clip-region-test", "text": "clip-text-test"},
        )
        self.assertEqual(
            smoke_raw["label_provenance"],
            {
                "must_report_as": "ground_truth",
                "source": "refcoco",
                "supervision_type": "ground_truth",
            },
        )
        self.assertEqual(smoke_raw["hardware"]["accelerator"], "cpu")
        self.assertEqual(smoke_raw["hardware"]["device"], "cpu")
        self.assertGreaterEqual(smoke_raw["hardware"]["wall_clock_hours"], 0.0)
        self.assertIn("production main tables should use 5 seeds", smoke_raw["seed_count_rationale"])
        self.assertTrue(set(REGION_TEXT_REQUIRED_PUBLIC_METRICS).issubset(set(smoke_raw["public_metrics"])))
        self.assertIn("candidate_iou_at_0_5", smoke_raw["public_metrics"])
        self.assertEqual(
            smoke_raw["public_metrics_scope"],
            "region_text_smoke_real_metrics_not_topconf_main_table",
        )
        self.assertIn("public metric inventory uses smoke-scale real metrics", smoke_raw["evidence_limitations"])
        self.assertEqual(len(smoke_baseline_rows), len(payload["baselines"]))
        baseline_models = {row["model"] for row in smoke_baseline_rows}
        self.assertEqual(baseline_models, set(payload["baselines"]))
        self.assertIn("concat_fusion", baseline_models)
        self.assertIn("ovha_no_cato", baseline_models)
        for row in smoke_baseline_rows:
            self.assertEqual(row["artifact_type"], "public_smoke_baseline_raw_metric")
            self.assertEqual(row["evidence_scope"], "same_feature_baseline_smoke_only_not_topconf_main_table")
            self.assertTrue(row["not_topconf_main_table"])
            self.assertTrue(row["same_feature_source"])
            self.assertEqual(row["baseline_protocol"], "deterministic_same_feature_probe_smoke")
            self.assertEqual(row["training_status"], "not_trained")
            self.assertEqual(row["task"], "phrase_region_grounding")
            self.assertEqual(row["dataset"], "refcoco")
            self.assertEqual(row["split"], "val")
            self.assertEqual(row["seed"], 201)
            self.assertEqual(row["metric_name"], "heldout_task_loss_smoke")
            self.assertFalse(row["higher_is_better"])
            self.assertGreaterEqual(row["score"], 0.0)
            self.assertEqual(row["raw_metric_path"], str(smoke_baseline_metrics_path))
            self.assertEqual(
                row["frozen_feature_extractor_version"],
                {"region": "clip-region-test", "text": "clip-text-test"},
            )
            self.assertTrue(set(REGION_TEXT_REQUIRED_PUBLIC_METRICS).issubset(set(row["public_metrics"])))
            self.assertIn("candidate_iou_at_0_5", row["public_metrics"])
            self.assertEqual(
                row["public_metrics_scope"],
                "region_text_smoke_real_metrics_not_topconf_main_table",
            )
            self.assertIn("not a trained strong baseline", row["evidence_limitations"])
            self.assertIn("public metric inventory uses smoke-scale real metrics", row["evidence_limitations"])
        self.assertEqual(smoke_statistics_preview["artifact_type"], "public_smoke_statistics_preview")
        self.assertEqual(
            smoke_statistics_preview["evidence_scope"],
            "smoke_statistics_preview_only_not_topconf_main_table",
        )
        self.assertTrue(smoke_statistics_preview["not_topconf_main_table"])
        self.assertEqual(
            smoke_statistics_preview["source_raw_metric_paths"],
            [str(smoke_raw_metrics_path), str(smoke_baseline_metrics_path)],
        )
        self.assertEqual(smoke_statistics_preview["full_model"], "ovha_full")
        self.assertEqual(smoke_statistics_preview["baseline_model"], "concat_fusion")
        self.assertFalse(smoke_statistics_preview["validation"]["ok"])
        self.assertIn("at least 3 seeds", "\n".join(smoke_statistics_preview["validation"]["errors"]))
        self.assertIn(
            "ovha_full",
            smoke_statistics_preview["summary"]["main_table"]["phrase_region_grounding"]["val"],
        )
        self.assertIn(
            "concat_fusion",
            smoke_statistics_preview["summary"]["main_table"]["phrase_region_grounding"]["val"],
        )
        self.assertIn("not valid top-conference main-table evidence", smoke_statistics_preview["evidence_limitations"])

    def test_public_smoke_runner_can_train_same_feature_baseline_smoke_models(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required for public training smoke")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "public_training_artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(tmp_path / "controlled_artifacts"), sort_keys=True) + "\n"
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                "--cache-root",
                str(cache_root),
                "--controlled-report",
                str(controlled_report_path),
                "--train-smoke-steps",
                "1",
                "--train-baseline-smoke-steps",
                "1",
                "--train-split",
                "train",
                "--eval-smoke-split",
                "val",
                "--artifact-root",
                str(artifact_root),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            smoke_baseline_metrics_path = Path(
                payload["training"]["artifacts"]["smoke_baseline_raw_metrics"]["path"]
            )
            smoke_baseline_rows = [
                json.loads(line) for line in smoke_baseline_metrics_path.read_text().splitlines() if line.strip()
            ]

        training = payload["training"]
        self.assertEqual(training["baseline_smoke_training_steps"], 1)
        self.assertEqual(training["baseline_optimizer_steps"], len(payload["baselines"]))
        self.assertEqual(training["baseline_training_status"], "trained_smoke")
        self.assertEqual(len(smoke_baseline_rows), len(payload["baselines"]))
        for row in smoke_baseline_rows:
            self.assertEqual(row["baseline_protocol"], "trainable_same_feature_linear_probe_smoke")
            self.assertEqual(row["training_status"], "trained_smoke")
            self.assertTrue(row["same_feature_source"])
            self.assertGreater(row["parameter_count"], 0)
            self.assertEqual(row["training_steps"], 1)
            self.assertEqual(row["baseline_optimizer_steps"], 1)
            self.assertGreater(row["baseline_parameter_l2_delta"], 0.0)
            self.assertGreater(row["baseline_grad_l2_norm"], 0.0)
            self.assertIn("not a trained strong baseline", row["evidence_limitations"])
            self.assertIn("trainable same-feature linear probe smoke", row["evidence_limitations"])
            self.assertNotIn(
                "deterministic probe only verifies baseline artifact plumbing and same-feature provenance",
                row["evidence_limitations"],
            )

    def test_public_smoke_runner_rejects_unverified_controlled_artifact_descriptors(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(json.dumps(_complete_controlled_public_entry_report(), sort_keys=True) + "\n")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                "--cache-root",
                str(cache_root),
                "--controlled-report",
                str(controlled_report_path),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        joined = "\n".join(payload["errors"])
        self.assertIn("controlled report evidence_artifacts controlled_rows.path does not exist", joined)
        self.assertIn("controlled report evidence_artifacts diagnostics_report.path does not exist", joined)

    def test_public_smoke_runner_rejects_controlled_report_artifact_content_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import file_sha256

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "controlled_artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report = _complete_controlled_public_entry_report(artifact_root)
            controlled_rows = Path(controlled_report["evidence_artifacts"]["controlled_rows"]["path"])
            rows = [json.loads(line) for line in controlled_rows.read_text().splitlines() if line.strip()]
            rows[0]["oracle_matrix"]["true_learned"]["loss"] += 1.0
            controlled_rows.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
            controlled_report["evidence_artifacts"]["controlled_rows"]["sha256"] = file_sha256(controlled_rows)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(json.dumps(controlled_report, sort_keys=True) + "\n")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                "--cache-root",
                str(cache_root),
                "--controlled-report",
                str(controlled_report_path),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "controlled report tleo_local_evidence.oracle_matrix.true_learned.loss disagrees with artifact recomputation",
            "\n".join(payload["errors"]),
        )

    def test_public_smoke_runner_rejects_controlled_diagnostics_artifact_content_mismatch(self):
        from moat_ovha_torch.data.multimodal.cache_schema import file_sha256

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "controlled_artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report = _complete_controlled_public_entry_report(artifact_root)
            diagnostics_report = Path(controlled_report["evidence_artifacts"]["diagnostics_report"]["path"])
            rows = [json.loads(line) for line in diagnostics_report.read_text().splitlines() if line.strip()]
            rows[0]["oracle_matrix"]["true_learned"]["loss"] += 1.0
            diagnostics_report.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
            controlled_report["evidence_artifacts"]["diagnostics_report"]["sha256"] = file_sha256(diagnostics_report)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(json.dumps(controlled_report, sort_keys=True) + "\n")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                "--cache-root",
                str(cache_root),
                "--controlled-report",
                str(controlled_report_path),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "controlled report diagnostics_report tleo_local_evidence oracle_matrix.true_learned.loss disagrees with controlled_rows",
            "\n".join(payload["errors"]),
        )

    def test_public_smoke_runner_rejects_diagnostics_artifact_that_duplicates_controlled_rows(self):
        from moat_ovha_torch.data.multimodal.cache_schema import file_sha256

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "controlled_artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report = _complete_controlled_public_entry_report(artifact_root)
            controlled_rows = Path(controlled_report["evidence_artifacts"]["controlled_rows"]["path"])
            diagnostics_report = Path(controlled_report["evidence_artifacts"]["diagnostics_report"]["path"])
            diagnostics_report.write_text(controlled_rows.read_text())
            controlled_report["evidence_artifacts"]["diagnostics_report"]["sha256"] = file_sha256(diagnostics_report)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(json.dumps(controlled_report, sort_keys=True) + "\n")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "multimodal" / "run_public_smoke.py"),
                str(ROOT / "configs" / "multimodal_refcoco_public_smoke.json"),
                "--cache-root",
                str(cache_root),
                "--controlled-report",
                str(controlled_report_path),
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        joined = "\n".join(payload["errors"])
        self.assertIn(
            "controlled report evidence_artifacts diagnostics_report.sha256 must differ from controlled_rows.sha256",
            joined,
        )
        self.assertIn(
            "controlled report diagnostics_report tleo_local_evidence artifact_type must be controlled_diagnostics",
            joined,
        )

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

    def test_topconf_main_entry_cli_accepts_artifact_reports_and_cache_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            region_gate_path = tmp_path / "region_gate_report.json"
            sentiment_gate_path = tmp_path / "sentiment_gate_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(artifact_root / "controlled"), sort_keys=True) + "\n"
            )
            region_gate_path.write_text(
                json.dumps(_passing_region_text_public_gate_report(artifact_root / "region"), sort_keys=True) + "\n"
            )
            sentiment_gate_path.write_text(
                json.dumps(_passing_sentiment_public_gate_report(artifact_root / "sentiment"), sort_keys=True) + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "validate_topconf_entry.py"),
                    "--controlled-report",
                    str(controlled_report_path),
                    "--region-gate-report",
                    str(region_gate_path),
                    "--sentiment-gate-report",
                    str(sentiment_gate_path),
                    "--cache-target",
                    "refcoco",
                    str(cache_root),
                    "refcoco",
                    "v0.1",
                    "val,test",
                    "--cache-target",
                    "cmu_mosei",
                    str(cache_root),
                    "cmu_mosei",
                    "v0.1",
                    "val,test",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "topconf_main_entry_validation")
        self.assertEqual(payload["cache_targets"], ["cmu_mosei", "refcoco"])

    def test_topconf_main_entry_cli_accepts_public_gate_bundle_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(artifact_root / "controlled"), sort_keys=True) + "\n"
            )
            region_bundle = _build_public_gate_bundle_fixture(
                tmp_path,
                gate="region_text",
                task="phrase_region_grounding",
            )
            sentiment_bundle = _build_public_gate_bundle_fixture(
                tmp_path,
                gate="sentiment",
                task="sentiment_emotion",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "validate_topconf_entry.py"),
                    "--controlled-report",
                    str(controlled_report_path),
                    "--region-gate-bundle",
                    str(region_bundle),
                    "--sentiment-gate-bundle",
                    str(sentiment_bundle),
                    "--cache-target",
                    "refcoco",
                    str(cache_root),
                    "refcoco",
                    "v0.1",
                    "val,test",
                    "--cache-target",
                    "cmu_mosei",
                    str(cache_root),
                    "cmu_mosei",
                    "v0.1",
                    "val,test",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "topconf_main_entry_validation")
        self.assertEqual(payload["gate_sources"]["region_text_public"], str(region_bundle / "region_text_gate_report.json"))
        self.assertEqual(
            payload["gate_sources"]["sentiment_emotion_public"],
            str(sentiment_bundle / "sentiment_gate_report.json"),
        )

    def test_topconf_main_entry_cli_accepts_reproducible_manifest_with_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(artifact_root / "controlled"), sort_keys=True) + "\n"
            )
            region_bundle = _build_public_gate_bundle_fixture(
                tmp_path,
                gate="region_text",
                task="phrase_region_grounding",
            )
            sentiment_bundle = _build_public_gate_bundle_fixture(
                tmp_path,
                gate="sentiment",
                task="sentiment_emotion",
            )
            manifest_path = tmp_path / "topconf_entry_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "controlled_report": "controlled_report.json",
                        "region_gate_bundle": "region_text_bundle",
                        "sentiment_gate_bundle": "sentiment_bundle",
                        "cache_targets": [
                            {
                                "name": "refcoco",
                                "cache_root": "cache",
                                "dataset": "refcoco",
                                "version": "v0.1",
                                "splits": ["val", "test"],
                            },
                            {
                                "name": "cmu_mosei",
                                "cache_root": "cache",
                                "dataset": "cmu_mosei",
                                "version": "v0.1",
                                "splits": ["val", "test"],
                            },
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "validate_topconf_entry.py"),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["manifest"], str(manifest_path))
        self.assertEqual(payload["cache_targets"], ["cmu_mosei", "refcoco"])
        self.assertEqual(payload["gate_sources"]["region_text_public"], str(region_bundle / "region_text_gate_report.json"))
        self.assertEqual(
            payload["gate_sources"]["sentiment_emotion_public"],
            str(sentiment_bundle / "sentiment_gate_report.json"),
        )

    def test_topconf_entry_manifest_builder_writes_relative_manifest_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "artifacts"
            manifest_path = tmp_path / "topconf_entry_manifest.json"
            _write_valid_refcoco_public_cache(cache_root)
            _write_valid_cmu_mosei_public_cache(cache_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(artifact_root / "controlled"), sort_keys=True) + "\n"
            )
            region_bundle = _build_public_gate_bundle_fixture(
                tmp_path,
                gate="region_text",
                task="phrase_region_grounding",
            )
            sentiment_bundle = _build_public_gate_bundle_fixture(
                tmp_path,
                gate="sentiment",
                task="sentiment_emotion",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_topconf_entry_manifest.py"),
                    "--output",
                    str(manifest_path),
                    "--controlled-report",
                    str(controlled_report_path),
                    "--region-gate-bundle",
                    str(region_bundle),
                    "--sentiment-gate-bundle",
                    str(sentiment_bundle),
                    "--cache-target",
                    "refcoco",
                    str(cache_root),
                    "refcoco",
                    "v0.1",
                    "val,test",
                    "--cache-target",
                    "cmu_mosei",
                    str(cache_root),
                    "cmu_mosei",
                    "v0.1",
                    "val,test",
                    "--validate",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            manifest = json.loads(manifest_path.read_text())

        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "topconf_entry_manifest_build")
        self.assertEqual(payload["manifest"], str(manifest_path))
        self.assertTrue(payload["validation"]["ok"], payload["validation"])
        self.assertEqual(manifest["controlled_report"], "controlled_report.json")
        self.assertEqual(manifest["region_gate_bundle"], "region_text_bundle")
        self.assertEqual(manifest["sentiment_gate_bundle"], "sentiment_bundle")
        self.assertEqual(manifest["cache_targets"][0]["cache_root"], "cache")
        self.assertEqual(manifest["cache_targets"][0]["splits"], ["val", "test"])

    def test_topconf_main_entry_cli_rejects_missing_required_cache_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "cache"
            artifact_root = tmp_path / "artifacts"
            _write_valid_refcoco_public_cache(cache_root)
            controlled_report_path = tmp_path / "controlled_report.json"
            region_gate_path = tmp_path / "region_gate_report.json"
            sentiment_gate_path = tmp_path / "sentiment_gate_report.json"
            controlled_report_path.write_text(
                json.dumps(_complete_controlled_public_entry_report(artifact_root / "controlled"), sort_keys=True) + "\n"
            )
            region_gate_path.write_text(
                json.dumps(_passing_region_text_public_gate_report(artifact_root / "region"), sort_keys=True) + "\n"
            )
            sentiment_gate_path.write_text(
                json.dumps(_passing_sentiment_public_gate_report(artifact_root / "sentiment"), sort_keys=True) + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "validate_topconf_entry.py"),
                    "--controlled-report",
                    str(controlled_report_path),
                    "--region-gate-report",
                    str(region_gate_path),
                    "--sentiment-gate-report",
                    str(sentiment_gate_path),
                    "--cache-target",
                    "refcoco",
                    str(cache_root),
                    "refcoco",
                    "v0.1",
                    "val,test",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "top-conference main experiments require at least one sentiment/emotion data cache",
            "\n".join(payload["errors"]),
        )

    def test_public_gate_bundle_cli_generates_topconf_evidence_artifacts_from_raw_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_metrics_path = tmp_path / "region_raw_metrics.jsonl"
            diagnostics_path = tmp_path / "region_diagnostics.jsonl"
            robustness_rows_path = tmp_path / "region_robustness_rows.jsonl"
            output_dir = tmp_path / "gate_bundle"
            raw_rows = [
                {**row, "higher_is_better": True}
                for row in _gate_statistics_summary("phrase_region_grounding", raw_metrics_path)["per_seed_appendix"]
            ]
            raw_metrics_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in raw_rows) + "\n")
            diagnostics_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in _gate_diagnostic_rows("phrase_region_grounding")) + "\n"
            )
            robustness_rows_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in _gate_robustness_rows()) + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_public_gate_report.py"),
                    "region_text",
                    "--raw-metrics",
                    str(raw_metrics_path),
                    "--diagnostics",
                    str(diagnostics_path),
                    "--robustness-rows",
                    str(robustness_rows_path),
                    "--task",
                    "phrase_region_grounding",
                    "--split",
                    "test",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            gate_report_path = Path(payload["artifacts"]["gate_report"]["path"])
            statistics_path = Path(payload["artifacts"]["statistics_summary"]["path"])
            robustness_summary_path = Path(payload["artifacts"]["robustness_summary"]["path"])
            gate_report = json.loads(gate_report_path.read_text())
            statistics = json.loads(statistics_path.read_text())
            robustness_summary = json.loads(robustness_summary_path.read_text())

        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["mode"], "public_gate_evidence_bundle")
        self.assertEqual(payload["gate"], "region_text")
        self.assertEqual(gate_report["name"], "region_text_public")
        self.assertTrue(gate_report["passed"], gate_report["reasons"])
        evidence = gate_report["evidence_artifacts"]
        self.assertEqual(evidence["task"], "phrase_region_grounding")
        self.assertEqual(evidence["split"], "test")
        self.assertIn("statistics_summary", evidence)
        self.assertIn("diagnostics", evidence)
        self.assertIn("robustness_summary", evidence)
        self.assertIn("robustness_rows", evidence)
        self.assertEqual(len(evidence["raw_metrics"]), 1)
        self.assertEqual(statistics["metadata"]["raw_metric_paths"], [str(raw_metrics_path.resolve())])
        self.assertEqual(robustness_summary["task"], "phrase_region_grounding")
        self.assertEqual(payload["validation"]["statistics"]["ok"], True)
        self.assertEqual(payload["validation"]["evidence_errors"], [])

    def test_public_gate_bundle_cli_rejects_smoke_raw_metrics_as_topconf_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_metrics_path = tmp_path / "region_public_smoke_raw_metrics.jsonl"
            diagnostics_path = tmp_path / "region_diagnostics.jsonl"
            robustness_rows_path = tmp_path / "region_robustness_rows.jsonl"
            output_dir = tmp_path / "gate_bundle"
            raw_rows = [
                {
                    **row,
                    "higher_is_better": True,
                    "artifact_type": "public_smoke_raw_metric",
                    "evidence_scope": "public_smoke_only_not_topconf_main_table",
                    "not_topconf_main_table": True,
                    "public_metrics_scope": "region_text_smoke_proxy_not_topconf_main_table",
                }
                for row in _gate_statistics_summary("phrase_region_grounding", raw_metrics_path)["per_seed_appendix"]
            ]
            raw_metrics_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in raw_rows) + "\n")
            diagnostics_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in _gate_diagnostic_rows("phrase_region_grounding")) + "\n"
            )
            robustness_rows_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in _gate_robustness_rows()) + "\n"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "build_public_gate_report.py"),
                    "region_text",
                    "--raw-metrics",
                    str(raw_metrics_path),
                    "--diagnostics",
                    str(diagnostics_path),
                    "--robustness-rows",
                    str(robustness_rows_path),
                    "--task",
                    "phrase_region_grounding",
                    "--split",
                    "test",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["validation"]["statistics"]["ok"])
        joined = "\n".join(payload["validation"]["statistics"]["errors"])
        self.assertIn("not_topconf_main_table rows cannot enter top-conference main-table statistics", joined)
        self.assertIn("evidence_scope is not top-conference main-table evidence", joined)

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
                        "baseline_model": "concat_fusion",
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

    def test_topconf_main_entry_rejects_controlled_diagnostics_without_semantic_marker(self):
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
                if row["family"] == "tleo_local_evidence":
                    row.pop("artifact_type", None)
                if row["family"] == "cato_alignment_transport":
                    row["active_operator"] = "CATO"
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
        joined = "\n".join(report.errors)
        self.assertIn(
            "controlled report diagnostics_report tleo_local_evidence artifact_type must be controlled_diagnostics",
            joined,
        )
        self.assertIn(
            "controlled report diagnostics_report cato_alignment_transport must not duplicate controlled_rows active_operator",
            joined,
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
    import numpy as np

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
    (root / "splits.json").write_text(
        json.dumps(
            {
                "train": ["train-source"],
                "val": ["val-source"],
                "test": ["test-source"],
                "testA": ["testA-source"],
                "testB": ["testB-source"],
            },
            sort_keys=True,
        )
        + "\n"
    )
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
    for split in ("train", "val", "test", "testA", "testB"):
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
                    "candidate_region_boxes": [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.9, 0.9]],
                    "candidate_region_annotation_ids": [100, 200],
                    "target_region_index": 1,
                    "candidate_permutation_seed": 1000,
                    "candidate_region_source": "annotated_boxes",
                    "box_coordinate_convention": "xyxy_normalized",
                },
                sort_keys=True,
            )
            + "\n"
        )
        (root / "provenance" / f"failed_samples_{split}.jsonl").write_text("")
        np.save(root / "supervision" / f"task_labels_{split}.npy", np.asarray([[0.0, 1.0]], dtype=np.float32))
        (root / "supervision" / f"alignment_pairs_{split}.parquet").write_text("placeholder alignment pairs\n")
        np.save(root / "supervision" / f"bbox_targets_{split}.npy", np.zeros((1, 4), dtype=np.float32))
        np.save(
            root / "supervision" / f"candidate_region_boxes_{split}.npy",
            np.asarray([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.9, 0.9]]], dtype=np.float32),
        )
        np.save(root / "supervision" / f"region_targets_{split}.npy", np.ones((1,), dtype=np.int64))
        (root / "supervision" / f"target_slot_histogram_by_valid_count_{split}.json").write_text(
            json.dumps(
                {
                    "audit_name": "target_slot_histogram_by_valid_count",
                    "sample_count": 1,
                    "by_valid_count": {
                        "2": {
                            "sample_count": 1,
                            "target_slot_counts": [0, 1],
                            "expected_per_slot": 0.5,
                            "max_deviation": 0.5,
                            "max_fraction": 1.0,
                        }
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        (root / "supervision" / f"corruption_{split}.parquet").write_text("placeholder corruption metadata\n")
        manifest = {}
        for modality in ("text", "region"):
            x_path = root / "token_fields" / f"{modality}_{split}.npy"
            pos_path = root / "positions" / f"{modality}_pos_{split}.npy"
            mask_path = root / "masks" / f"{modality}_mask_{split}.npy"
            np.save(x_path, np.zeros((1, 2, 3), dtype=np.float32))
            np.save(pos_path, np.zeros((1, 2, 1), dtype=np.float32))
            np.save(mask_path, np.ones((1, 2), dtype=bool))
            manifest[modality] = {
                "x": str(x_path.relative_to(root)),
                "pos": str(pos_path.relative_to(root)),
                "mask": str(mask_path.relative_to(root)),
            }
        np.save(root / "masks" / f"text_mask_{split}.npy", np.ones((1, 2), dtype=bool))
        (root / "token_fields" / f"manifest_{split}.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    checksums = {}
    for path in required_cache_files(layout, splits=("val", "test", "testA", "testB")):
        if path.exists():
            checksums[str(path.relative_to(root))] = file_sha256(path)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            checksums.setdefault(str(path.relative_to(root)), file_sha256(path))
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True) + "\n")


def _write_valid_refcoco_raw_manifest(raw_root: Path) -> None:
    import numpy as np

    for folder in ("annotations", "features", "provenance"):
        (raw_root / folder).mkdir(parents=True, exist_ok=True)
    split_ids = {
        "train": ["train-source"],
        "val": ["val-source"],
        "test": ["test-source"],
        "testA": ["testA-source"],
        "testB": ["testB-source"],
    }
    records = []
    for row_index, (split, source_ids) in enumerate(split_ids.items()):
        source_id = source_ids[0]
        records.append(
            {
                "source_id": source_id,
                "split": split,
                "original_split": split,
                "raw_ref": f"raw://{source_id}",
                "license_tag": "test-license",
                "preprocessing_version": "preprocess-v1",
                "image_id": f"image-{source_id}",
                "caption_id": f"caption-{source_id}",
                "phrase_span": {"start": 0, "end": 2},
                "region_box": [0.0, 0.0, 1.0, 1.0],
                "target_region_index": row_index % 2,
                "candidate_region_boxes": [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.9, 0.9]],
                "candidate_region_annotation_ids": [100 + row_index * 2, 101 + row_index * 2],
                "candidate_permutation_seed": 1000 + row_index,
                "candidate_region_source": "annotated_boxes",
                "box_coordinate_convention": "xyxy_normalized",
            }
        )
    (raw_root / "splits.json").write_text(json.dumps(split_ids, sort_keys=True) + "\n")
    (raw_root / "annotations" / "refs.json").write_text(json.dumps({"records": records}, sort_keys=True) + "\n")
    (raw_root / "annotations" / "instances.json").write_text(
        json.dumps({"records": [{"source_id": record["source_id"], "image_id": record["image_id"]} for record in records]}, sort_keys=True)
        + "\n"
    )
    (raw_root / "provenance" / "failed_samples.jsonl").write_text("")
    np.save(raw_root / "features" / "text_features.npy", np.zeros((5, 2, 3), dtype=np.float32))
    np.save(raw_root / "features" / "region_features.npy", np.zeros((5, 2, 3), dtype=np.float32))


def _write_valid_cmu_mosei_public_cache(cache_root: Path) -> None:
    import numpy as np

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
    (root / "splits.json").write_text(
        json.dumps({"train": ["train-utt"], "val": ["val-utt"], "test": ["test-utt"]}, sort_keys=True) + "\n"
    )
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
    for split in ("train", "val", "test"):
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
        np.save(root / "supervision" / f"task_labels_{split}.npy", np.zeros((1,), dtype=np.int64))
        np.save(root / "supervision" / f"missing_modality_mask_{split}.npy", np.zeros((1, 3), dtype=bool))
        (root / "supervision" / f"corruption_{split}.parquet").write_text("placeholder corruption metadata\n")
        manifest = {}
        for modality in ("text", "audio", "vision"):
            x_path = root / "token_fields" / f"{modality}_{split}.npy"
            pos_path = root / "positions" / f"{modality}_pos_{split}.npy"
            mask_path = root / "masks" / f"{modality}_mask_{split}.npy"
            np.save(x_path, np.zeros((1, 2, 3), dtype=np.float32))
            np.save(pos_path, np.zeros((1, 2, 1), dtype=np.float32))
            np.save(mask_path, np.ones((1, 2), dtype=bool))
            manifest[modality] = {
                "x": str(x_path.relative_to(root)),
                "pos": str(pos_path.relative_to(root)),
                "mask": str(mask_path.relative_to(root)),
            }
        np.save(root / "masks" / f"text_mask_{split}.npy", np.ones((1, 2), dtype=bool))
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
        "LRIO": ["rank_logits", "rank_logits_by_pair", "interaction_temperature", "interaction_temperature_by_pair", "scale", "bias"],
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
    diagnostics = {key: row[key] for key in keys if key in row}
    diagnostics["artifact_type"] = "controlled_diagnostics"
    return diagnostics


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
                "full_beats_sanity_probe_or_robustness_advantage": {"passed": True},
                "lrio_admission_rows_present": {"passed": True},
                "no_rceo_drops": {"passed": True},
                "spo_router_load_high": {"passed": True},
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
        "model_delta": "ovha_full_minus_concat_fusion",
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
            if model not in {"ovha_full", "concat_fusion"}
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


def _public_main_metric_rows(config_path: Path, raw_metrics: Path) -> list[dict[str, object]]:
    from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

    config = MultimodalExperimentConfig.from_file(config_path)
    models = (config.main_model_name, *config.baseline_names)
    rows = []
    for model_index, model in enumerate(models):
        for seed_index, seed in enumerate(config.seeds):
            score = 0.80 - 0.01 * model_index + 0.001 * seed_index
            rows.append(
                {
                    "artifact_type": "public_main_raw_metric",
                    "evidence_scope": "public_main_table",
                    "dataset": config.dataset_name,
                    "task": config.task_type,
                    "model": model,
                    "stage": "T5_eval",
                    "split": "testA",
                    "seed": seed,
                    "metric_name": "main_task_score",
                    "score": score,
                    "higher_is_better": True,
                    "parameter_count": 123456 + model_index,
                    "training_steps": 1000,
                    "frozen_feature_extractor_version": {"text": "frozen-text-v1", "region": "frozen-region-v1"},
                    "hardware": {"accelerator": "A800", "wall_clock_hours": 1.5},
                    "label_provenance": {"supervision_type": "ground_truth", "source": config.dataset_name},
                    "public_metrics": _public_metrics_for_task(config.task_type),
                    "public_metrics_scope": "public_main_metrics",
                    "raw_metric_path": str(raw_metrics),
                }
            )
    return rows


def _public_metrics_for_task(task: str) -> dict[str, object]:
    if task == "sentiment_emotion":
        return {
            "mae": 0.3,
            "mse_loss": 0.12,
            "l1_loss": 0.3,
            "pearson_correlation": 0.7,
            "acc7": 0.42,
            "acc5": 0.48,
            "acc2_excl0": 0.74,
            "f1_excl0": 0.73,
            "acc2_nonneg": 0.75,
            "f1_nonneg": 0.74,
            "accuracy": 0.74,
            "f1": 0.73,
            "missing_modality_performance_drop": 0.08,
            "corruption_robustness_auc": 0.82,
            "router_load_by_corruption_type": {
                "missing_audio": {"TLEO": 0.20, "SPO": 0.45, "LRIO": 0.15, "CATO": 0.20},
                "audio_noise": {"TLEO": 0.10, "SPO": 0.25, "LRIO": 0.50, "CATO": 0.15},
            },
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
        from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task

        return ("ovha_full", *baseline_names_for_task(task))
    return (
        "ovha_full",
        "text_only",
        "region_only",
        "concat_fusion",
        "cato_only",
        "ovha_no_cato",
        "ovha_no_rceo",
        "ovha_no_evidence_router",
    )


def _gate_score_for_model(task: str, model: str, *, full_score: float, baseline_score: float) -> float:
    if model == "ovha_full":
        return full_score
    if model == "concat_fusion":
        return baseline_score
    if task == "sentiment_emotion":
        return {
            "text_only": 0.73,
            "audio_only": 0.72,
            "vision_only": 0.71,
            "spo_only": 0.705,
            "lrio_only": 0.706,
            "ovha_tanso_only": 0.704,
            "ovha_no_tanso": 0.707,
            "ovha_spo_lrio": 0.707,
            "ovha_spo_tanso": 0.708,
            "ovha_lrio_tanso": 0.709,
            "ovha_no_lrio": 0.70,
            "ovha_no_spo": 0.71,
            "ovha_no_rceo": 0.68,
            "ovha_with_evidence_router": 0.69,
        }.get(model, 0.71)
    return {
        "cato_only": 0.71,
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
        baseline_model="concat_fusion",
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
                "model": "concat_fusion",
                "corruption_type": "image_blur",
                "corruption_strength": 0.0,
                "score": 0.75,
            },
            {
                "model": "concat_fusion",
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


def _robustness_stress_raw_metric_rows() -> list[dict[str, object]]:
    corruptions = [
        "missing_text",
        "missing_vision",
        "missing_audio",
        "image_blur",
        "image_crop",
        "image_occlusion",
        "audio_noise",
        "audio_masking",
        "text_token_mask",
        "text_paraphrase",
        "hard_negative_caption_mismatch",
        "hard_negative_region_mismatch",
        "hard_negative_audio_mismatch",
    ]
    model_drops = {
        "ovha_full": 0.06,
        "concat_fusion": 0.18,
        "ovha_no_rceo": 0.20,
        "ovha_no_evidence_router": 0.22,
    }
    rows = []
    for seed, clean_score in ((401, 0.80), (402, 0.78), (403, 0.76)):
        for model, drop in model_drops.items():
            stress_scores = {corruption: clean_score * (1.0 - drop) for corruption in corruptions}
            reliability = {
                corruption: 0.90 if corruption.startswith("missing_") and model != "ovha_full" else 0.55
                for corruption in corruptions
            }
            reliability.update({corruption: 0.62 for corruption in corruptions if model == "ovha_full"})
            rows.append(
                {
                    "artifact_type": "public_raw_metric_fixture",
                    "dataset": "refcoco",
                    "task": "phrase_region_grounding",
                    "model": model,
                    "split": "test",
                    "seed": seed,
                    "source_id": f"refcoco-{seed}-{model}",
                    "metric_name": "acc_at_0_5",
                    "score": clean_score,
                    "higher_is_better": True,
                    "public_metrics": {
                        "clean_robustness_score": clean_score,
                        "robustness_score_by_corruption_type": stress_scores,
                        "corruption_strength_by_type": {corruption: 0.5 for corruption in corruptions},
                        "rceo_reliability_by_corruption_type": reliability,
                        "router_load_by_corruption_type": {
                            corruption: {"TLEO": 0.20, "SPO": 0.35, "LRIO": 0.25, "CATO": 0.20}
                            for corruption in corruptions
                        },
                    },
                    "raw_metric_path": "raw_metrics.jsonl",
                }
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


def _build_public_gate_bundle_fixture(tmp_path: Path, *, gate: str, task: str) -> Path:
    bundle_root = tmp_path / f"{gate}_bundle"
    raw_metrics_path = tmp_path / f"{gate}_raw_metrics.jsonl"
    diagnostics_path = tmp_path / f"{gate}_diagnostics.jsonl"
    robustness_rows_path = tmp_path / f"{gate}_robustness_rows.jsonl"
    raw_rows = [
        {**row, "higher_is_better": True}
        for row in _gate_statistics_summary(task, raw_metrics_path)["per_seed_appendix"]
    ]
    raw_metrics_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in raw_rows) + "\n")
    diagnostics_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in _gate_diagnostic_rows(task)) + "\n"
    )
    robustness_rows_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in _gate_robustness_rows()) + "\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "multimodal" / "build_public_gate_report.py"),
            gate,
            "--raw-metrics",
            str(raw_metrics_path),
            "--diagnostics",
            str(diagnostics_path),
            "--robustness-rows",
            str(robustness_rows_path),
            "--task",
            task,
            "--split",
            "test",
            "--output-dir",
            str(bundle_root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    return bundle_root


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
