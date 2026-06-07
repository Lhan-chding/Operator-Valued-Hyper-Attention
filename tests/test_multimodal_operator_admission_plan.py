import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class MultimodalOperatorAdmissionPlanTests(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_tanso_base_and_shift_are_distinct_candidates_with_null_source_diagnostics(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        batch = _sentiment_batch(torch)
        model = MultimodalOVHA(
            field_dims={name: int(field.x.shape[-1]) for name, field in batch.fields.items()},
            query_dim=int(batch.query.x.shape[-1]),
            output_dim=int(batch.target_y.shape[-1]),
            d_model=8,
            memory_tokens=1,
            candidate_names=("TANSOBase", "TANSOShift"),
            use_evidence_router=False,
            composition_mode="base_plus_residual",
            base_candidate="TANSOBase",
            residual_candidates=("TANSOShift",),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )

        output = model(batch)
        base_diag = output.diagnostics["candidate_diagnostics"]["TANSOBase"]
        shift_diag = output.diagnostics["candidate_diagnostics"]["TANSOShift"]

        self.assertEqual(base_diag["semantic_role"], "full_predictor")
        self.assertEqual(shift_diag["semantic_role"], "residual_delta")
        for diagnostics in (base_diag, shift_diag):
            self.assertIn("null_source_load", diagnostics)
            total_load = (
                diagnostics["audio_source_load"]
                + diagnostics["vision_source_load"]
                + diagnostics["null_source_load"]
            )
            self.assertTrue(torch.allclose(total_load, torch.ones_like(total_load), atol=1e-5))
            self.assertIn("null", diagnostics["source_gate_tensor"])
        self.assertEqual(output.diagnostics["composition"]["base_candidate"], "TANSOBase")
        self.assertEqual(output.diagnostics["composition"]["residual_candidates"], ("TANSOShift",))
        self.assertEqual(tuple(output.y_hat.shape), tuple(batch.target_y.shape))

    def test_cmu_tanso_primary_config_encodes_five_seed_official_protocol(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_tanso_primary_main.json")

        self.assertEqual(config.main_model_name, "ovha_tanso_primary")
        self.assertEqual(config.candidate_names, ("TANSOBase",))
        self.assertEqual(config.composition_mode, "tanso_base")
        self.assertEqual(config.seeds, (301, 302, 303, 304, 305))
        self.assertFalse(config.allow_hidden_losses)
        self.assertEqual(config.loss_metadata["candidate_individual_loss"]["weight"], 0.0)
        self.assertEqual(config.loss_metadata["val_affine_calibration"]["stage"], "validation_postfit")

    def test_cmu_tanso_no_rceo_selfmm_config_promotes_clean_primary_protocol(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        path = ROOT / "configs" / "multimodal_cmu_mosei_tanso_no_rceo_selfmm_official.json"
        config = MultimodalExperimentConfig.from_file(path)

        self.assertEqual(config.main_model_name, "ovha_tanso_no_rceo_primary")
        self.assertEqual(config.candidate_names, ("TANSOBase",))
        self.assertEqual(config.composition_mode, "tanso_base")
        self.assertFalse(config.use_evidence_router)
        self.assertFalse(config.use_reliability_prior)
        self.assertEqual(config.feature_source, "selfmm_official_unaligned_50")
        self.assertEqual(config.checkpoint_selection_metric, "mosei_composite")
        self.assertEqual(
            config.checkpoint_selection_weights,
            {
                "mae": 0.35,
                "pearson_correlation": 0.25,
                "acc7": 0.15,
                "acc5": 0.15,
                "acc2_excl0": 0.05,
                "acc2_nonneg": 0.05,
            },
        )
        self.assertEqual(config.lr_schedule, "warmup_cosine")
        self.assertEqual(config.warmup_steps, 500)
        self.assertEqual(config.min_lr_ratio, 0.05)
        self.assertIn("binary_margin_auxiliary", config.losses_by_stage["T5"])
        self.assertTrue(config.loss_metadata["ordinal_acc5_acc7_auxiliary"]["class_balanced"])
        self.assertEqual(config.loss_metadata["binary_margin_auxiliary"]["margin"], 0.15)

    def test_cmu_tanso_mechanism_config_covers_proof_plan_without_smoke_scope(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        path = ROOT / "configs" / "multimodal_cmu_mosei_tanso_mechanism_selfmm_official.json"
        config = MultimodalExperimentConfig.from_file(path)

        self.assertEqual(config.main_model_name, "ovha_tanso_full")
        self.assertEqual(config.feature_source, "selfmm_official_unaligned_50")
        self.assertEqual(config.cache_version, "v0.1_selfmm_official")
        self.assertEqual(config.seeds, (301, 302, 303, 304, 305))
        self.assertEqual(config.candidate_names, ("TANSOBase",))
        self.assertEqual(config.composition_mode, "tanso_base")
        self.assertNotIn("smoke", str(config.output_dir).lower())
        proof_plan_models = {
            "raw_tanso_mlp",
            "ovha_tanso_no_source_gate",
            "ovha_tanso_no_hyper_adapter",
            "ovha_tanso_no_operator_memory",
            "ovha_tanso_no_gate_aux",
            "spo_only",
            "lrio_only",
            "ovha_lrio_tanso",
            "ovha_all_candidates_exploratory",
        }
        self.assertTrue(proof_plan_models.issubset(set(config.baseline_names)))
        self.assertEqual(config.loss_metadata["tanso_source_oracle_gate_loss"]["weight"], 0.01)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_tanso_mechanism_candidate_options_disable_components_with_diagnostics(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        batch = _sentiment_batch(torch)
        model = MultimodalOVHA(
            field_dims={name: int(field.x.shape[-1]) for name, field in batch.fields.items()},
            query_dim=int(batch.query.x.shape[-1]),
            output_dim=int(batch.target_y.shape[-1]),
            d_model=8,
            memory_tokens=2,
            candidate_names=("TANSOBase",),
            use_evidence_router=False,
            composition_mode="tanso_base",
            lrio_pairs=(("text", "audio"), ("text", "vision")),
            candidate_options={
                "TANSOBase": {
                    "source_gate_mode": "uniform_nonverbal",
                    "use_operator_memory": False,
                    "use_hyper_adapter": False,
                }
            },
        )

        output = model(batch)
        diagnostics = output.diagnostics["candidate_diagnostics"]["TANSOBase"]

        self.assertEqual(diagnostics["source_gate_mode"], "uniform_nonverbal")
        self.assertFalse(diagnostics["operator_memory_enabled"])
        self.assertFalse(diagnostics["hyper_adapter_enabled"])
        self.assertGreater(float(diagnostics["fixed_source_gate_applied"].detach().cpu().item()), 0.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_cmu_tanso_primary_baselines_do_not_inherit_tanso_base_composition(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_smoke import _ovha_composition_kwargs

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_tanso_primary_main.json")

        self.assertEqual(
            _ovha_composition_kwargs(config, ("TANSOBase",)),
            {"composition_mode": "tanso_base"},
        )
        self.assertEqual(
            _ovha_composition_kwargs(config, ("SPO",)),
            {"composition_mode": "convex_mixture"},
        )
        self.assertEqual(
            _ovha_composition_kwargs(config, ("LRIO",)),
            {"composition_mode": "convex_mixture"},
        )

    def test_cmu_tanso_mechanism_variants_are_registered_for_public_main_runner(self):
        from moat_ovha_torch.models.multimodal.baselines import baseline_protocol_for_name
        from scripts.multimodal.run_public_main import _config_for_ovha_ablation, _ovha_variant_kwargs
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig

        active = ("SPO", "LRIO", "TANSO", "TANSOBase", "TANSOShift")

        self.assertEqual(baseline_protocol_for_name("sentiment_emotion", "raw_tanso_mlp"), "same_feature_mechanism_baseline")
        self.assertEqual(
            _ovha_variant_kwargs("ovha_tanso_no_source_gate", active)["candidate_options"]["TANSOBase"]["source_gate_mode"],
            "uniform_nonverbal",
        )
        self.assertFalse(
            _ovha_variant_kwargs("ovha_tanso_no_hyper_adapter", active)["candidate_options"]["TANSOBase"]["use_hyper_adapter"]
        )
        self.assertFalse(
            _ovha_variant_kwargs("ovha_tanso_no_operator_memory", active)["candidate_options"]["TANSOBase"]["use_operator_memory"]
        )

        config = MultimodalExperimentConfig.from_file(
            ROOT / "configs" / "multimodal_cmu_mosei_tanso_mechanism_selfmm_official.json"
        )
        no_aux = _config_for_ovha_ablation(config, "ovha_tanso_no_gate_aux")
        self.assertEqual(no_aux.loss_metadata["tanso_source_oracle_gate_loss"]["weight"], 0.0)

    def test_public_main_runner_selects_only_requested_configured_baselines(self):
        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_main import _selected_baselines

        config = MultimodalExperimentConfig.from_file(
            ROOT / "configs" / "multimodal_cmu_mosei_tanso_mechanism_selfmm_official.json"
        )

        selected = _selected_baselines(
            config,
            ("raw_tanso_mlp", "ovha_tanso_no_source_gate", "raw_tanso_mlp"),
        )

        self.assertEqual(selected, ("raw_tanso_mlp", "ovha_tanso_no_source_gate"))
        with self.assertRaises(ValueError):
            _selected_baselines(config, ("not_a_configured_baseline",))

    def test_ubuntu_cmu_mechanism_runner_defaults_to_missing_only_continuation(self):
        path = ROOT / "scripts" / "multimodal" / "run_cmu_mosei_tanso_mechanism_selfmm_official.sh"
        source = path.read_text()

        self.assertIn("multimodal_cmu_mosei_tanso_mechanism_selfmm_official.json", source)
        self.assertIn("scripts/multimodal/run_public_main.py", source)
        self.assertIn("CACHE_ROOT", source)
        self.assertIn("--cache-root", source)
        self.assertIn("--skip-main-model", source)
        self.assertIn("--only-baseline", source)
        self.assertIn("RUN_SCOPE=\"${RUN_SCOPE:-missing_only}\"", source)
        self.assertIn("REFERENCE_RAW_METRICS", source)
        self.assertIn("scripts/multimodal/validate_public_main_artifacts.py", source)
        self.assertIn("scripts/multimodal/summarize_public_results.py", source)
        self.assertIn("raw_tanso_mlp", source)
        self.assertIn("ovha_tanso_no_source_gate", source)
        self.assertIn("ovha_tanso_no_hyper_adapter", source)
        self.assertIn("ovha_tanso_no_operator_memory", source)
        self.assertIn("ovha_tanso_no_gate_aux", source)

    def test_operator_admission_eval_reports_residual_utility_and_rejects_interference(self):
        rows = [
            {"sample_id": "a", "split": "val", "model": "base", "truth": 1.0, "prediction": 0.0},
            {"sample_id": "b", "split": "val", "model": "base", "truth": 0.0, "prediction": 1.0},
            {"sample_id": "a", "split": "val", "model": "candidate", "prediction": 4.0},
            {"sample_id": "b", "split": "val", "model": "candidate", "prediction": -0.1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            predictions = tmp_path / "predictions.jsonl"
            output = tmp_path / "admission.json"
            predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "multimodal" / "operator_admission_eval.py"),
                    "--predictions-jsonl",
                    str(predictions),
                    "--base-model",
                    "base",
                    "--candidate-model",
                    "candidate",
                    "--output",
                    str(output),
                    "--bootstrap-samples",
                    "32",
                    "--seed",
                    "7",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(output.read_text())

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(payload["admission_decision"], "diagnostic_only")
        self.assertGreater(payload["residual_alignment"], 0.0)
        self.assertGreater(payload["non_interference_delta"], 0.0)
        self.assertIn("paired_bootstrap_p", payload)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_mosei_composite_selection_and_lr_schedule_helpers_follow_pro_plan(self):
        import torch

        from scripts.multimodal.run_public_main import _lr_for_step, _mosei_composite_score

        score = _mosei_composite_score(
            {
                "mae": 0.56,
                "pearson_correlation": 0.74,
                "acc7": 0.52,
                "acc5": 0.54,
                "acc2_excl0": 0.84,
                "acc2_nonneg": 0.82,
            },
            {
                "mae": 0.35,
                "pearson_correlation": 0.25,
                "acc7": 0.15,
                "acc5": 0.15,
                "acc2_excl0": 0.05,
                "acc2_nonneg": 0.05,
            },
        )

        self.assertAlmostEqual(score, 0.419, places=6)
        self.assertAlmostEqual(_lr_for_step(250, 0.0003, 500, 12000, 0.05), 0.00015)
        self.assertAlmostEqual(_lr_for_step(500, 0.0003, 500, 12000, 0.05), 0.0003)
        self.assertLess(_lr_for_step(12000, 0.0003, 500, 12000, 0.05), 0.00002)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_metric_aware_calibration_returns_all_required_affine_variants(self):
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_main import _apply_task_calibration, _fit_task_calibrator

        config = MultimodalExperimentConfig.from_file(
            ROOT / "configs" / "multimodal_cmu_mosei_tanso_no_rceo_selfmm_official.json"
        )
        batch = replace(
            _sentiment_batch(torch),
            target_y=torch.tensor([[[1.0], [-1.0], [2.0]], [[0.0], [1.0], [-2.0]]]),
        )
        prediction = torch.tensor([[[0.5], [-0.4], [1.2]], [[0.1], [0.4], [-1.0]]])

        calibrator = _fit_task_calibrator(config, prediction, batch)
        self.assertEqual(calibrator["selected"], "composite_affine_calibration")
        self.assertEqual(
            set(calibrator["calibrators"]),
            {"mse_affine_calibration", "huber_affine_calibration", "composite_affine_calibration"},
        )
        self.assertIn("validation_metrics", calibrator)
        calibrated = _apply_task_calibration(prediction, calibrator)
        self.assertEqual(tuple(calibrated.shape), tuple(prediction.shape))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_metric_aware_calibration_can_fit_inside_no_grad_evaluation_block(self):
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_main import _fit_task_calibrator

        config = MultimodalExperimentConfig.from_file(
            ROOT / "configs" / "multimodal_cmu_mosei_tanso_no_rceo_selfmm_official.json"
        )
        batch = replace(
            _sentiment_batch(torch),
            target_y=torch.tensor([[[1.0], [-1.0], [2.0]], [[0.0], [1.0], [-2.0]]]),
        )
        prediction = torch.tensor([[[0.5], [-0.4], [1.2]], [[0.1], [0.4], [-1.0]]])

        with torch.no_grad():
            calibrator = _fit_task_calibrator(config, prediction, batch)

        self.assertEqual(calibrator["selected"], "composite_affine_calibration")
        self.assertEqual(calibrator["calibrators"]["huber_affine_calibration"]["objective"], "validation_huber_affine")

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_class_balanced_ordinal_and_binary_margin_losses_are_available(self):
        import torch

        from scripts.multimodal.run_public_smoke import (
            _binary_margin_auxiliary_loss,
            _class_balanced_ordinal_ce_from_scalar,
        )

        batch = replace(
            _sentiment_batch(torch),
            target_y=torch.tensor([[[1.0], [-1.0], [0.0]], [[2.0], [-2.0], [1.0]]]),
        )
        prediction = torch.tensor([[[0.05], [-0.05], [-0.02]], [[0.2], [-0.2], [0.01]]])
        binary_loss = _binary_margin_auxiliary_loss(
            prediction,
            batch,
            margin=0.15,
            targets=("excl0", "nonneg"),
        )
        ordinal_loss = _class_balanced_ordinal_ce_from_scalar(
            prediction[..., 0].reshape(-1),
            batch.target_y[..., 0].reshape(-1),
            bins=torch.arange(-3, 4, dtype=torch.float32),
        )

        self.assertGreater(float(binary_loss.item()), 0.0)
        self.assertGreater(float(ordinal_loss.item()), 0.0)

    def test_mosei_operator_admission_runner_writes_admitted_bank_and_val_table(self):
        runner = ROOT / "scripts" / "multimodal" / "run_mosei_operator_admission.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "admission"
            result = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--base-config",
                    str(ROOT / "configs" / "multimodal_cmu_mosei_tanso_no_rceo_selfmm_official.json"),
                    "--candidate-residual",
                    "LRIO",
                    "--base-composite",
                    "0.407",
                    "--candidate-composite",
                    "0.405",
                    "--candidate-mae",
                    "0.560",
                    "--base-mae",
                    "0.559",
                    "--candidate-acc7",
                    "0.522",
                    "--base-acc7",
                    "0.523",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((output_dir / "admitted_bank.json").exists())
            self.assertTrue((output_dir / "base_vs_candidate_val_table.csv").exists())
            payload = json.loads((output_dir / "admitted_bank.json").read_text())

        self.assertEqual(payload["base_model"], "TANSOBase-noRCEO")
        self.assertEqual(payload["candidate_residuals"], ["LRIO"])
        self.assertEqual(payload["admitted_residuals"], ["LRIO"])


def _sentiment_batch(torch):
    from moat_ovha_torch.data.multimodal.typed_batch import (
        MultimodalEpisodeBatch,
        ProvenanceBank,
        QueryField,
        SupervisionBank,
        TokenField,
    )

    batch_size = 2
    query_count = 3
    token_count = 4
    dim = 5
    fields = {}
    for offset, name in enumerate(("text", "audio", "vision")):
        values = torch.arange(batch_size * token_count * dim, dtype=torch.float32).view(batch_size, token_count, dim)
        fields[name] = TokenField(
            modality=name,
            x=(values / 50.0) + float(offset),
            pos=torch.linspace(0.0, 1.0, token_count).view(1, token_count, 1).repeat(batch_size, 1, 2),
            mask=torch.ones(batch_size, token_count, dtype=torch.bool),
            quality=torch.ones(batch_size, token_count, 1),
            attrs=None,
        )
    query_x = torch.ones(batch_size, query_count, dim) * 0.25
    return MultimodalEpisodeBatch(
        fields=fields,
        query=QueryField(
            x=query_x,
            pos=torch.linspace(0.0, 1.0, query_count).view(1, query_count, 1).repeat(batch_size, 1, 2),
            query_type=torch.zeros(batch_size, query_count, dtype=torch.long),
            mask=torch.ones(batch_size, query_count, dtype=torch.bool),
        ),
        target_y=torch.zeros(batch_size, query_count, 1),
        target_mask=torch.ones(batch_size, query_count, dtype=torch.bool),
        task_type="sentiment_emotion",
        split="train",
        source_dataset="cmu_mosei",
        supervision=SupervisionBank(
            task_label=None,
            alignment_pairs=None,
            alignment_weights=None,
            bbox_targets=None,
            region_targets=None,
            timestamp_targets=None,
            modality_missing_mask=None,
            corruption_metadata=None,
            weak_labels=None,
            weak_label_confidence=None,
            pseudo_label_source=None,
        ),
        provenance=ProvenanceBank(
            source_id=["sample-0", "sample-1"],
            original_split=["train", "train"],
            raw_ref=["synthetic", "synthetic"],
            license_tag=["synthetic", "synthetic"],
            preprocessing_version="test",
            feature_extractor_version={"text": "test", "audio": "test", "vision": "test"},
            pseudo_label_version={},
        ),
        hidden=None,
    )
