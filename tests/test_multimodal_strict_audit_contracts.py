import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


class MultimodalStrictAuditStaticContracts(unittest.TestCase):
    def test_cmu_public_configs_use_task_aware_spo_lrio_tanso_bank(self):
        for name in ("multimodal_cmu_mosei_public_main.json", "multimodal_cmu_mosei_public_smoke.json"):
            with self.subTest(config=name):
                payload = json.loads((ROOT / "configs" / name).read_text())

                self.assertEqual(payload["candidate_names"], ["SPO", "LRIO", "TANSO"])
                self.assertEqual(payload["residual_candidates"], ["LRIO", "TANSO"])
                self.assertEqual(
                    payload["lrio_pairs"],
                    [["text", "audio"], ["text", "vision"]],
                )

    def test_tanso_candidate_enters_cmu_text_anchored_shift_stack(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for TANSO tensor checks")
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(31)
        batch = _batch(torch)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=16,
            memory_tokens=2,
            candidate_names=("SPO", "LRIO", "TANSO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
            composition_mode="base_plus_residual",
            base_candidate="SPO",
            residual_candidates=("LRIO", "TANSO"),
        )

        output = model(batch)
        tanso = output.candidate_outputs["TANSO"]

        self.assertEqual(tuple(output.y_hat.shape), (3, 2, 1))
        self.assertEqual(tuple(output.candidate_values.shape), (3, 2, 3, 1))
        self.assertNotIn("candidate_loss", output.diagnostics)
        self.assertIn("TANSO", output.diagnostics["composition"]["residual_gate_by_candidate"])
        self.assertEqual(tuple(tanso.value.shape), (3, 2, 1))
        self.assertGreaterEqual(float(tanso.diagnostics["shift_magnitude"].detach().cpu()), 0.0)
        self.assertIn("audio_shift_load", tanso.diagnostics)
        self.assertIn("vision_shift_load", tanso.diagnostics)
        self.assertNotIn("shift_direction_alignment", tanso.diagnostics)

    def test_tanso_v2_exposes_text_token_temporal_lag_and_source_oracle_diagnostics(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for TANSO tensor checks")
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(41)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=16,
            memory_tokens=2,
            candidate_names=("SPO", "LRIO", "TANSO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
            composition_mode="base_plus_residual",
            base_candidate="SPO",
            residual_candidates=("LRIO", "TANSO"),
        )

        output = model(_batch(torch))
        tanso_diag = output.diagnostics["candidate_diagnostics"]["TANSO"]

        self.assertEqual(tanso_diag["tanso_version"], "v2_temporal_lag_residual")
        self.assertIn("text_token_anchor_count", tanso_diag)
        self.assertIn("temporal_lag_hypotheses", tanso_diag)
        self.assertIn("temporal_lag_load", tanso_diag)
        self.assertIn("source_gate_tensor", tanso_diag)
        self.assertIn("source_specific_raw_delta", tanso_diag)
        self.assertIn("source_specific_temporal_entropy", tanso_diag)
        self.assertIn("source_oracle_alpha", output.diagnostics["public_residual_oracles"])

    def test_tanso_text_pair_features_are_field_order_invariant(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for TANSO tensor checks")
        import torch

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceBank
        from moat_ovha_torch.models.multimodal.hyper_adapter import _candidate_query_feature
        from moat_ovha_torch.models.multimodal.memory import _candidate_episode_feature

        query_features = torch.zeros(2, 3, 4)
        text_audio = torch.full((2, 3, 4), 2.0)
        text_vision = torch.full((2, 3, 4), 6.0)
        audio_vision = torch.full((2, 3, 4), 99.0)
        evidence = MultimodalEvidenceBank(
            query_features=query_features,
            global_features=torch.zeros(2, 4),
            local_features=query_features,
            prototype_features=query_features,
            low_rank_features=torch.full((2, 3, 4), -5.0),
            alignment_features=query_features,
            candidate_evidence_logits=torch.zeros(2, 3, 5),
            local_entropy=torch.zeros(()),
            alignment_entropy=torch.zeros(()),
            field_features={},
            diagnostics={},
            all_pair_features={
                "audio__text": text_audio,
                "audio__vision": audio_vision,
                "text__vision": text_vision,
            },
        )

        expected = torch.full((2, 3, 4), 4.0)
        self.assertTrue(torch.allclose(_candidate_query_feature("TANSO", evidence), expected))
        self.assertTrue(torch.allclose(_candidate_episode_feature("TANSO", evidence), expected.mean(dim=1)))

    def test_tanso_admission_gate_closes_without_nonverbal_sources(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for TANSO tensor checks")
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(37)
        base = _batch(torch)
        batch = replace(
            base,
            fields={"text": base.fields["text"]},
            provenance=replace(base.provenance, feature_extractor_version={"text": "unit"}),
        )
        model = MultimodalOVHA(
            field_dims={"text": 5},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=("SPO", "TANSO"),
            composition_mode="base_plus_residual",
            base_candidate="SPO",
            residual_candidates=("TANSO",),
        )

        output = model(batch)
        spo = output.candidate_outputs["SPO"].value

        self.assertEqual(float(output.diagnostics["candidate_diagnostics"]["TANSO"]["nonverbal_source_count"]), 0.0)
        self.assertEqual(float(output.diagnostics["operator_admission_gate"]["TANSO"]), 0.0)
        self.assertTrue(torch.allclose(output.router_weights[..., 1], torch.zeros_like(output.router_weights[..., 1])))
        self.assertTrue(torch.allclose(output.y_hat, spo, atol=1e-7))
        self.assertTrue(torch.allclose(output.diagnostics["composition"]["residual_gate_tensor_by_candidate"]["TANSO"], torch.zeros_like(spo), atol=1e-7))

    def test_model_forward_outputs_and_diagnostics_do_not_depend_on_target_y(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for target isolation checks")
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(38)
        batch = _batch(torch)
        altered = replace(batch, target_y=-batch.target_y)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=("SPO", "LRIO", "TANSO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
            composition_mode="base_plus_residual",
            base_candidate="SPO",
            residual_candidates=("LRIO", "TANSO"),
        )
        model.eval()

        with torch.no_grad():
            original = model(batch)
            changed = model(altered)

        self.assertTrue(torch.allclose(original.y_hat, changed.y_hat, atol=1e-7))
        self.assertNotIn("shift_direction_alignment", original.candidate_outputs["TANSO"].diagnostics)
        self.assertNotIn("candidate_loss", original.diagnostics)
        self.assertNotIn("candidate_loss_by_sample", original.diagnostics)
        self.assertNotIn("candidate_loss", original.diagnostics["candidate_diagnostics"]["TANSO"])
        self.assertEqual(original.diagnostics["composition"]["mode"], changed.diagnostics["composition"]["mode"])

    def test_cmu_public_main_uses_reverse_evidence_router_ablation_not_duplicate(self):
        payload = json.loads((ROOT / "configs" / "multimodal_cmu_mosei_public_main.json").read_text())

        self.assertFalse(payload["use_evidence_router"])
        self.assertNotIn("ovha_no_evidence_router", payload["baseline_names"])
        self.assertIn("ovha_with_evidence_router", payload["baseline_names"])

    def test_cmu_public_main_declares_metric_aligned_pro_losses_and_calibration(self):
        payload = json.loads((ROOT / "configs" / "multimodal_cmu_mosei_public_main.json").read_text())
        t5_losses = payload["losses_by_stage"]["T5"]

        self.assertIn("huber_l1_task_loss", t5_losses)
        self.assertIn("ordinal_acc5_acc7_auxiliary", t5_losses)
        self.assertIn("residual_oracle_gate_loss", t5_losses)
        self.assertIn("tanso_source_oracle_gate_loss", t5_losses)
        self.assertIn("val_affine_calibration", t5_losses)
        self.assertGreater(payload["loss_metadata"]["huber_l1_task_loss"]["weight"], 0.0)
        self.assertGreater(payload["loss_metadata"]["ordinal_acc5_acc7_auxiliary"]["weight"], 0.0)
        self.assertEqual(payload["loss_metadata"]["val_affine_calibration"]["stage"], "validation_postfit")

    def test_public_runner_implements_metric_aligned_losses_and_val_calibration(self):
        runner = (ROOT / "scripts" / "multimodal" / "run_public_main.py").read_text()
        smoke = (ROOT / "scripts" / "multimodal" / "run_public_smoke.py").read_text()

        for token in (
            "_huber_l1_task_loss",
            "_ordinal_acc5_acc7_auxiliary_loss",
            "_residual_oracle_gate_loss",
            "_tanso_source_oracle_gate_loss",
        ):
            self.assertIn(token, smoke)
        self.assertIn("_fit_affine_calibrator", runner)
        self.assertIn("_apply_affine_calibration", runner)
        self.assertIn("post_calibration_public_metrics", runner)

    def test_mosei_standard_metrics_match_mult_exclude_zero_binary_protocol(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for MOSEI metric tensor checks")
        import torch

        from moat_ovha_torch.eval.mosei_standard_metrics import mosei_standard_metrics

        prediction = torch.tensor([[[-0.2]], [[0.7]], [[0.3]], [[-0.1]]])
        target = torch.tensor([[[-1.0]], [[0.0]], [[1.0]], [[0.0]]])
        mask = torch.ones(4, 1, dtype=torch.bool)

        metrics = mosei_standard_metrics(prediction, target, mask)

        self.assertEqual(metrics["acc2_excl0"], 1.0)
        self.assertEqual(metrics["f1_excl0"], 1.0)
        self.assertEqual(metrics["acc2_nonneg"], 0.75)
        self.assertLess(metrics["f1_nonneg"], 1.0)

    def test_mosei_standard_metrics_accept_flat_single_target_tensors(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for MOSEI metric tensor checks")
        import torch

        from moat_ovha_torch.eval.mosei_standard_metrics import mosei_standard_metrics

        prediction = torch.tensor([[-1.2], [0.2], [1.1], [2.6], [0.0]])
        target = torch.tensor([[-1.0], [0.0], [1.0], [3.0], [-0.2]])
        mask = torch.ones_like(target, dtype=torch.bool)

        metrics = mosei_standard_metrics(prediction, target, mask)

        self.assertEqual(metrics["acc2_excl0"], 1.0)
        self.assertEqual(metrics["acc5"], 1.0)
        self.assertAlmostEqual(metrics["mae"], 0.22, places=6)

    def test_public_main_runner_exposes_official_selection_split_contract(self):
        source = (ROOT / "scripts" / "multimodal" / "run_public_main.py").read_text()

        self.assertIn("--selection-split", source)
        self.assertIn('"official_val_selection_best_checkpoint"', source)
        self.assertNotIn("_split_train_val_batch(\n        train_batch,", source)

    def test_multimodal_primitives_are_not_shared_linear_head_stubs(self):
        tleo = (ROOT / "moat_ovha_torch" / "models" / "multimodal" / "primitives" / "typed_local_evidence.py").read_text()
        lrio = (ROOT / "moat_ovha_torch" / "models" / "multimodal" / "primitives" / "low_rank_interaction.py").read_text()
        cato = (ROOT / "moat_ovha_torch" / "models" / "multimodal" / "primitives" / "alignment_transport.py").read_text()

        self.assertIn("self.query_proj = nn.ModuleDict", tleo)
        self.assertIn("self.key_proj = nn.ModuleDict", tleo)
        self.assertIn("self.value_proj = nn.ModuleDict", tleo)
        self.assertIn("self.source_gate = nn.ModuleDict", tleo)

        self.assertIn("self.branch = nn.ModuleDict", lrio)
        self.assertIn("self.trunk = nn.ModuleDict", lrio)
        self.assertIn("self.pair_gate = nn.ModuleDict", lrio)
        self.assertIn("active_pair_names", lrio)

        self.assertIn("self.query_proj = nn.ModuleDict", cato)
        self.assertIn("self.key_proj = nn.ModuleDict", cato)
        self.assertIn("self.value_proj = nn.ModuleDict", cato)
        self.assertIn("_source_transport_marginal_error", cato)


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed; strict multimodal audit tests skipped.")
class MultimodalStrictAuditContracts(unittest.TestCase):
    def test_untrained_controlled_router_does_not_use_task_type_hidden_truth(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(7)
        batch = _batch(torch, task_type="cato_alignment_transport")
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
        )

        with torch.no_grad():
            output = model(batch)

        self.assertLess(
            float(output.router_weights.max().item()),
            0.70,
            "untrained controlled router must not become one-hot from task_type or hidden active-operator leakage",
        )
        self.assertEqual(
            float(output.evidence.diagnostics["controlled_family_relation_prior_rate"].item()),
            0.0,
        )

    def test_adapter_special_params_receive_task_loss_gradient(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(11)
        batch = _batch(torch)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=16,
            memory_tokens=2,
        )

        output = model(batch)
        loss = (output.y_hat - batch.target_y).square().mean()
        loss.backward()

        special_rows = {
            "TLEO": slice(2, 4),
            "SPO": slice(2, 7),
            "LRIO": slice(2, 7),
            "CATO": slice(2, 4),
        }
        for candidate, row_slice in special_rows.items():
            with self.subTest(candidate=candidate):
                grad = model.joint_router_adapter.hyper_adapter.heads[candidate].weight.grad
                self.assertIsNotNone(grad)
                self.assertGreater(
                    float(grad[row_slice].norm().item()),
                    1e-8,
                    f"{candidate} adapter special parameters must affect candidate output, not just diagnostics",
                )

    def test_rceo_is_pure_reliability_prior_not_evidence_logit_duplicate(self):
        import torch

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder
        from moat_ovha_torch.models.multimodal.reliability_prior import RCEOReliabilityPrior

        torch.manual_seed(13)
        batch = _batch(torch)
        encoder = MultimodalEvidenceEncoder(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            d_model=10,
        )
        evidence = encoder(batch)
        perturbed = evidence.__class__(
            query_features=evidence.query_features,
            global_features=evidence.global_features,
            local_features=evidence.local_features,
            prototype_features=evidence.prototype_features,
            low_rank_features=evidence.low_rank_features,
            alignment_features=evidence.alignment_features,
            candidate_evidence_logits=evidence.candidate_evidence_logits + 100.0,
            local_entropy=evidence.local_entropy,
            alignment_entropy=evidence.alignment_entropy,
            field_features=evidence.field_features,
            diagnostics=evidence.diagnostics,
        )
        rceo = RCEOReliabilityPrior(d_model=10)

        bias = rceo(batch, evidence).operator_logit_bias
        perturbed_bias = rceo(batch, perturbed).operator_logit_bias

        self.assertTrue(
            torch.allclose(bias, perturbed_bias, atol=1e-6),
            "RCEO must not reintroduce candidate evidence logits when router evidence is disabled",
        )

    def test_lrio_preserves_ordered_pair_direction(self):
        from moat_ovha_torch.models.multimodal.primitives.low_rank_interaction import LRIOPrimitive

        primitive = LRIOPrimitive(d_model=8, output_dim=1, pairs=(("text", "audio"), ("audio", "text")))

        self.assertEqual(primitive.pairs, (("text", "audio"), ("audio", "text")))
        self.assertIn("text__audio", primitive.branch)
        self.assertIn("audio__text", primitive.branch)

    def test_rceo_exposes_pair_reliability_for_lrio_pairs(self):
        import torch

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder
        from moat_ovha_torch.models.multimodal.reliability_prior import RCEOReliabilityPrior

        batch = _batch(torch)
        encoder = MultimodalEvidenceEncoder(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            d_model=10,
        )
        reliability = RCEOReliabilityPrior(
            d_model=10,
            candidate_names=("SPO", "LRIO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )(batch, encoder(batch))

        self.assertEqual(reliability.pair_names, ("text__audio", "text__vision"))
        self.assertEqual(tuple(reliability.pair_reliability.shape), (3, 2))
        expected = torch.stack(
            [
                reliability.modality_reliability[:, 0] * reliability.modality_reliability[:, 1],
                reliability.modality_reliability[:, 0] * reliability.modality_reliability[:, 2],
            ],
            dim=-1,
        )
        self.assertTrue(torch.allclose(reliability.pair_reliability, expected))

    def test_rceo_prior_ignores_query_semantic_features(self):
        import torch

        from dataclasses import replace

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder
        from moat_ovha_torch.models.multimodal.reliability_prior import RCEOReliabilityPrior

        torch.manual_seed(23)
        batch = _batch(torch)
        encoder = MultimodalEvidenceEncoder(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            d_model=10,
        )
        evidence = encoder(batch)
        shifted = replace(evidence, query_features=evidence.query_features + 100.0)
        prior = RCEOReliabilityPrior(d_model=10, candidate_names=("SPO", "LRIO"))

        reliability = prior(batch, evidence)
        shifted_reliability = prior(batch, shifted)

        self.assertTrue(torch.allclose(reliability.operator_logit_bias, shifted_reliability.operator_logit_bias, atol=1e-6))
        self.assertTrue(torch.allclose(reliability.features, shifted_reliability.features, atol=1e-6))

    def test_spo_diagnostics_include_prototype_diversity_and_collapse_warning(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(29)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=("SPO",),
        )

        with torch.no_grad():
            output = model(_batch(torch))

        spo = output.diagnostics["candidate_diagnostics"]["SPO"]
        self.assertIn("prototype_diversity", spo)
        self.assertIn("prototype_collapse_warning", spo)
        self.assertGreaterEqual(float(spo["prototype_diversity"]), 0.0)
        self.assertIn(bool(spo["prototype_collapse_warning"]), (False, True))

    def test_tleo_all_masked_source_has_zero_gate_and_no_nan(self):
        import torch

        from dataclasses import replace

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder
        from moat_ovha_torch.models.multimodal.primitives.typed_local_evidence import TLEOPrimitive

        batch = _batch(torch)
        text = batch.fields["text"]
        masked = replace(
            batch,
            fields={
                "text": replace(
                    text,
                    x=torch.zeros_like(text.x),
                    mask=torch.zeros_like(text.mask),
                    quality=torch.zeros(text.x.shape[0], 1),
                )
            },
        )
        encoder = MultimodalEvidenceEncoder(field_dims={"text": 5}, query_dim=6, d_model=8)
        evidence = encoder(masked)
        primitive = TLEOPrimitive(d_model=8, output_dim=1, modalities=("text",))
        params = {
            "lengthscale": torch.ones(3, 2, 1),
            "local_temperature": torch.ones(3, 2, 1),
            "scale": torch.ones(3, 2, 1),
            "bias": torch.zeros(3, 2, 1),
        }

        output = primitive(masked, torch.zeros(3, 2, 8), evidence, params, output_dim=1)

        self.assertFalse(torch.isnan(output.value).any())
        self.assertFalse(torch.isnan(output.feature).any())
        self.assertAlmostEqual(float(output.diagnostics["modality_gate"]["text"].detach()), 0.0, places=6)
        self.assertAlmostEqual(float(output.diagnostics["valid_source_rate"]["text"].detach()), 0.0, places=6)

    def test_cato_all_masked_source_has_zero_gate_and_token_alignment_diagnostics(self):
        import torch

        from dataclasses import replace

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder
        from moat_ovha_torch.models.multimodal.primitives.alignment_transport import CATOPrimitive

        batch = _batch(torch)
        vision = batch.fields["vision"]
        masked = replace(
            batch,
            fields={
                "vision": replace(
                    vision,
                    x=torch.zeros_like(vision.x),
                    mask=torch.zeros_like(vision.mask),
                    quality=torch.zeros(vision.x.shape[0], 1),
                )
            },
        )
        encoder = MultimodalEvidenceEncoder(field_dims={"vision": 3}, query_dim=6, d_model=8)
        evidence = encoder(masked)
        primitive = CATOPrimitive(d_model=8, output_dim=1, modalities=("vision",))
        params = {
            "alignment_temperature": torch.ones(3, 2, 1),
            "transport_scale": torch.ones(3, 2, 1),
            "scale": torch.ones(3, 2, 1),
            "bias": torch.zeros(3, 2, 1),
        }

        output = primitive(masked, torch.zeros(3, 2, 8), evidence, params, output_dim=1)

        self.assertFalse(torch.isnan(output.value).any())
        self.assertAlmostEqual(float(output.diagnostics["source_gate"]["vision"].detach()), 0.0, places=6)
        self.assertAlmostEqual(float(output.diagnostics["valid_source_rate"]["vision"].detach()), 0.0, places=6)
        self.assertIn("token_alignment_entropy", output.diagnostics)
        self.assertIn("source_token_marginal", output.diagnostics)
        self.assertIn("query_token_marginal", output.diagnostics)
        self.assertAlmostEqual(float(output.diagnostics["null_mass"].detach()), 1.0, places=6)
        self.assertIn("learned_null_mass", output.diagnostics)

    def test_cato_null_dustbin_is_learnable(self):
        import torch

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder
        from moat_ovha_torch.models.multimodal.primitives.alignment_transport import CATOPrimitive

        torch.manual_seed(53)
        batch = _batch(torch)
        evidence = MultimodalEvidenceEncoder(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            d_model=8,
        )(batch)
        primitive = CATOPrimitive(d_model=8, output_dim=1)
        params = {
            "alignment_temperature": torch.ones(3, 2, 1),
            "transport_scale": torch.ones(3, 2, 1),
            "scale": torch.ones(3, 2, 1),
            "bias": torch.zeros(3, 2, 1),
        }
        output = primitive(batch, torch.zeros(3, 2, 8), evidence, params, output_dim=1)
        output.value.square().mean().backward()

        self.assertTrue(hasattr(primitive, "null_value"))
        self.assertIsNotNone(primitive.null_value.grad)
        self.assertGreater(float(primitive.null_value.grad.norm()), 0.0)

    def test_lrio_diagnostics_include_pair_load_reliability_and_rank_entropy(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(17)
        batch = _batch(torch)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=("SPO", "LRIO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )

        with torch.no_grad():
            output = model(batch)

        lrio = output.diagnostics["candidate_diagnostics"]["LRIO"]
        self.assertEqual(set(lrio["pair_load"]), {"text__audio", "text__vision"})
        self.assertEqual(set(lrio["pair_reliability"]), {"text__audio", "text__vision"})
        self.assertEqual(set(lrio["pair_rank_entropy"]), {"text__audio", "text__vision"})
        self.assertGreaterEqual(lrio["pair_reliability"]["text__audio"], 0.0)
        self.assertLessEqual(lrio["pair_reliability"]["text__audio"], 1.0)

    def test_lrio_pair_specific_rank_logits_are_not_global_broadcast(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(37)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=("SPO", "LRIO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )
        adapter = model.joint_router_adapter.hyper_adapter

        with torch.no_grad():
            adapter.lrio_pair_rank_bias["text__audio"].copy_(torch.tensor([6.0, 0.0, 0.0, 0.0]))
            adapter.lrio_pair_rank_bias["text__vision"].zero_()

        with torch.no_grad():
            output = model(_batch(torch))

        lrio = output.diagnostics["candidate_diagnostics"]["LRIO"]
        self.assertIn("rank_logits_by_pair_mean", output.diagnostics["adapter_params_detail"]["LRIO"])
        self.assertLess(
            float(lrio["pair_rank_entropy"]["text__audio"]),
            float(lrio["pair_rank_entropy"]["text__vision"]),
        )

    def test_evidence_encoder_exposes_pair_specific_bank(self):
        import torch

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder

        torch.manual_seed(41)
        evidence = MultimodalEvidenceEncoder(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            d_model=10,
        )(_batch(torch))

        self.assertEqual(set(evidence.pair_features), {"text__audio", "text__vision", "audio__vision"})
        self.assertIn("pair_feature_count", evidence.diagnostics)
        self.assertFalse(torch.allclose(evidence.pair_features["text__audio"], evidence.pair_features["text__vision"]))

    def test_configured_lrio_pair_filter_blocks_audio_vision_from_evidence_memory_and_adapter(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(42)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=("SPO", "LRIO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )

        with torch.no_grad():
            output = model(_batch(torch))

        evidence = output.evidence
        self.assertEqual(set(evidence.pair_features), {"text__audio", "text__vision"})
        self.assertNotIn("audio__vision", evidence.pair_features)
        self.assertEqual(float(evidence.diagnostics["configured_pair_feature_count"]), 2.0)
        self.assertEqual(float(evidence.diagnostics["all_pair_feature_count"]), 3.0)

        lrio = output.diagnostics["candidate_diagnostics"]["LRIO"]
        self.assertEqual(set(lrio["active_pair_names"]), {"text__audio", "text__vision"})
        self.assertNotIn("audio__vision", output.diagnostics["adapter_params_detail"]["LRIO"])

    def test_lrio_output_invariant_to_batch_companions(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(44)
        base = _batch(torch)
        batch_a = _batch_with_first_sample_companions(torch, base, companion_seed=101)
        batch_b = _batch_with_first_sample_companions(torch, base, companion_seed=202)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=("LRIO",),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )

        with torch.no_grad():
            value_a = model(batch_a).candidate_outputs["LRIO"].value[0]
            value_b = model(batch_b).candidate_outputs["LRIO"].value[0]

        self.assertTrue(
            torch.allclose(value_a, value_b, atol=1e-6),
            "LRIO output for a sample must not depend on unrelated batch companions",
        )

    def test_rceo_clean_all_present_logit_bias_is_zero(self):
        import torch

        from dataclasses import replace

        from moat_ovha_torch.models.multimodal.evidence import MultimodalEvidenceEncoder
        from moat_ovha_torch.models.multimodal.reliability_prior import RCEOReliabilityPrior

        batch = _batch(torch)
        clean_fields = {
            name: replace(field, quality=torch.ones(field.x.shape[0], 1))
            for name, field in batch.fields.items()
        }
        clean_batch = replace(batch, fields=clean_fields)
        encoder = MultimodalEvidenceEncoder(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            d_model=10,
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )
        prior = RCEOReliabilityPrior(
            d_model=10,
            candidate_names=("SPO", "LRIO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )

        reliability = prior(clean_batch, encoder(clean_batch))

        self.assertTrue(torch.allclose(reliability.operator_logit_bias, torch.zeros_like(reliability.operator_logit_bias), atol=1e-7))
        self.assertEqual(float(reliability.diagnostics["operator_logit_bias_norm"].detach()), 0.0)

    def test_base_plus_residual_composition_uses_spo_base_and_lrio_delta(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(45)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=("SPO", "LRIO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
            composition_mode="base_plus_residual",
            base_candidate="SPO",
            residual_candidates=("LRIO",),
        )

        with torch.no_grad():
            output = model(_batch(torch))

        spo = output.candidate_outputs["SPO"].value
        lrio_delta = output.candidate_outputs["LRIO"].value
        lrio_gate = output.diagnostics["composition"]["residual_gate_tensor_by_candidate"]["LRIO"]
        expected = spo + lrio_gate * lrio_delta

        self.assertTrue(torch.allclose(output.y_hat, expected, atol=1e-6))
        self.assertTrue(torch.allclose(output.candidate_values[..., 1, :], expected, atol=1e-6))
        self.assertEqual(output.diagnostics["composition"]["mode"], "base_plus_residual")
        self.assertIn("raw_delta_by_candidate", output.diagnostics["composition"])
        self.assertIn("gated_delta_by_candidate", output.diagnostics["composition"])
        self.assertIn("ungated_corrected_candidate_values_by_candidate", output.diagnostics["composition"])

    def test_cmu_configs_disable_evidence_router_and_use_residual_composition(self):
        import json

        for name in ("multimodal_cmu_mosei_public_main.json", "multimodal_cmu_mosei_public_smoke.json"):
            with self.subTest(config=name):
                payload = json.loads((ROOT / "configs" / name).read_text())

                self.assertFalse(payload["use_evidence_router"])
                self.assertEqual(payload["composition_mode"], "base_plus_residual")
                self.assertEqual(payload["base_candidate"], "SPO")
                self.assertEqual(payload["residual_candidates"], ["LRIO", "TANSO"])

    def test_public_cmu_t5_uses_spo_diversity_and_router_utility_losses(self):
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA
        from scripts.multimodal.run_public_smoke import _public_loss_components

        torch.manual_seed(43)
        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_main.json")
        self.assertIn("spo_prototype_diversity", config.losses_by_stage["T5"])
        self.assertIn("router_marginal_utility", config.losses_by_stage["T5"])
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=config.candidate_names,
            lrio_pairs=config.lrio_pairs,
        )
        batch = _batch(torch)
        output = model(batch)
        components = _public_loss_components(output, batch, config)

        self.assertIn("spo_prototype_diversity", components)
        self.assertIn("router_marginal_utility", components)
        self.assertGreater(float(components["spo_prototype_diversity"].detach()), 0.0)
        self.assertGreaterEqual(float(components["router_marginal_utility"].detach()), 0.0)

    def test_public_loss_components_skip_diagnostic_only_zero_weight_candidate_loss(self):
        import torch
        from types import SimpleNamespace

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_smoke import _public_loss_components

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_main.json")
        batch = _batch(torch)
        output = SimpleNamespace(
            y_hat=torch.zeros_like(batch.target_y),
            candidate_values=torch.full((3, 2, len(config.candidate_names), 1), float("nan")),
            candidate_outputs={name: object() for name in config.candidate_names},
            diagnostics={},
            router_weights=torch.full((3, 2, len(config.candidate_names)), 1.0 / len(config.candidate_names)),
        )

        components = _public_loss_components(output, batch, config)

        self.assertIn("task_loss", components)
        self.assertNotIn("candidate_individual_loss", components)
        self.assertTrue(torch.isfinite(components["task_loss"]))

    def test_sentiment_public_diagnostics_include_candidate_gate_and_residual_oracles(self):
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA
        from scripts.multimodal.run_public_smoke import _public_training_diagnostics_row

        torch.manual_seed(49)
        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_main.json")
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=config.candidate_names,
            use_evidence_router=config.use_evidence_router,
            lrio_pairs=config.lrio_pairs,
            composition_mode=config.composition_mode,
            base_candidate=config.base_candidate,
            residual_candidates=config.residual_candidates,
        )

        batch = _batch(torch)
        with torch.no_grad():
            output = model(batch)
        row = _public_training_diagnostics_row(output, config, batch, step=1, seed=301)
        public = row["public_diagnostics"]

        self.assertIn("candidate_oracle_selection", public)
        self.assertIn("gate_sweep", public)
        self.assertIn("residual_oracle", public)
        self.assertIn("full_minus_oracle_min_loss", public["candidate_oracle_selection"])
        self.assertIn("best_alpha", public["gate_sweep"])
        self.assertIn("best_gamma", public["residual_oracle"])

    def test_router_marginal_utility_uses_per_sample_candidate_losses(self):
        import torch
        from types import SimpleNamespace

        from scripts.multimodal.run_public_smoke import _router_marginal_utility_loss

        candidate_outputs = {"SPO": object(), "LRIO": object()}
        diagnostics = {
            "candidate_loss": {"SPO": torch.tensor(5.0), "LRIO": torch.tensor(5.0)},
            "candidate_loss_by_sample": {
                "SPO": torch.tensor([[0.0, 10.0]]),
                "LRIO": torch.tensor([[10.0, 0.0]]),
            },
        }
        aligned = SimpleNamespace(
            y_hat=torch.zeros(1, 2, 1),
            candidate_outputs=candidate_outputs,
            diagnostics=diagnostics,
            router_weights=torch.tensor([[[0.99, 0.01], [0.01, 0.99]]]),
        )
        uniform = SimpleNamespace(
            y_hat=torch.zeros(1, 2, 1),
            candidate_outputs=candidate_outputs,
            diagnostics=diagnostics,
            router_weights=torch.full((1, 2, 2), 0.5),
        )

        self.assertLess(
            float(_router_marginal_utility_loss(aligned).detach()),
            float(_router_marginal_utility_loss(uniform).detach()),
        )

    def test_operator_admission_and_memory_differentiation_diagnostics(self):
        import torch

        from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA

        torch.manual_seed(47)
        model = MultimodalOVHA(
            field_dims={"text": 5, "audio": 4, "vision": 3},
            query_dim=6,
            output_dim=1,
            d_model=12,
            memory_tokens=2,
            candidate_names=("SPO", "LRIO"),
            lrio_pairs=(("text", "audio"), ("text", "vision")),
        )
        model.eval()

        with torch.no_grad():
            diagnostics = model(_batch(torch)).diagnostics

        self.assertEqual(set(diagnostics["operator_admission_gate"]), {"SPO", "LRIO"})
        self.assertIn("pair_admission_gate", diagnostics["candidate_diagnostics"]["LRIO"])
        self.assertIn("memory_slot_orthogonality", diagnostics)
        self.assertIn("memory_zero_out_delta", diagnostics)
        self.assertIn("memory_swap_delta", diagnostics)
        self.assertGreaterEqual(float(diagnostics["memory_slot_orthogonality"]), 0.0)
        self.assertGreaterEqual(float(diagnostics["memory_zero_out_delta"]), 0.0)
        self.assertGreaterEqual(float(diagnostics["memory_swap_delta"]), 0.0)

    def test_rceo_calibration_uses_multi_bin_ece(self):
        import torch

        from scripts.multimodal.run_public_main import _rceo_calibration

        batch = _batch(torch)
        prediction = batch.target_y + torch.tensor([[[0.0]], [[0.5]], [[1.0]]])
        diagnostics = {
            "reliability": {
                "sample_modality_reliability_mean": torch.tensor([0.1, 0.6, 0.9]),
                "modality_reliability_mean": torch.tensor(0.5333),
            }
        }

        calibration = _rceo_calibration(batch, prediction, diagnostics)

        self.assertEqual(calibration["bin_count"], 10)
        self.assertEqual(len(calibration["calibration_curve"]), 10)
        self.assertGreaterEqual(calibration["expected_calibration_error"], 0.0)

    def test_public_diagnostics_reject_lrio_pairs_outside_config(self):
        import torch
        from types import SimpleNamespace

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_smoke import _public_training_diagnostics_row

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_main.json")
        batch = _batch(torch)
        output = SimpleNamespace(
            router_logit_parts={},
            diagnostics={
                "router_entropy": torch.tensor(0.0),
                "router_load_by_candidate": {"SPO": torch.tensor(0.5), "LRIO": torch.tensor(0.5)},
                "router_memory_logit_norm": torch.tensor(0.0),
                "router_evidence_logit_norm": torch.tensor(0.0),
                "router_reliability_logit_norm": torch.tensor(0.0),
                "candidate_loss": {"SPO": torch.tensor(0.1), "LRIO": torch.tensor(0.2)},
                "adapter_params": {},
                "memory_slot_norm": {"SPO": torch.tensor(1.0), "LRIO": torch.tensor(1.0)},
                "stackability_passed": True,
                "candidate_diagnostics": {
                    "LRIO": {
                        "pair_load": {"audio__vision": torch.tensor(1.0)},
                        "pair_reliability": {"audio__vision": torch.tensor(1.0)},
                        "pair_rank_entropy": {"audio__vision": torch.tensor(0.5)},
                    }
                },
                "reliability": {},
            },
        )

        with self.assertRaisesRegex(ValueError, "LRIO diagnostics contain unconfigured modality pairs"):
            _public_training_diagnostics_row(output, config, batch, 0, 301)

    def test_cache_root_override_preserves_lrio_pair_contract(self):
        from pathlib import Path

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_smoke import _replace_cache_root

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_main.json")
        overridden = _replace_cache_root(config, Path("data/alternate_cache"))

        self.assertEqual(overridden.cache_root, Path("data/alternate_cache"))
        self.assertEqual(overridden.lrio_pairs, config.lrio_pairs)
        self.assertEqual(overridden.adapter_params_by_candidate, config.adapter_params_by_candidate)
        self.assertEqual(overridden.loss_metadata, config.loss_metadata)

    def test_sentiment_public_batch_uses_neutral_query_not_text_mean(self):
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
        from scripts.multimodal.run_public_smoke import _load_public_batch

        with tempfile.TemporaryDirectory() as tmp:
            layout = _write_public_cache_fixture(Path(tmp), task_type="sentiment_regression")
            config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_smoke.json")

            batch = _load_public_batch(layout, config, "train", torch.device("cpu"))

        text_mean = batch.fields["text"].x.mean(dim=1, keepdim=True).expand_as(batch.query.x)
        self.assertTrue(torch.all(batch.query.x == 0.0))
        self.assertFalse(torch.allclose(batch.query.x, text_mean))

    def test_public_main_sentiment_metrics_use_model_diagnostics_not_router_proxy(self):
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_main import _public_main_metrics

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_main.json")
        batch = _batch(torch)
        prediction = batch.target_y.clone()
        diagnostics = {
            "candidate_diagnostics": {
                "LRIO": {"rank_entropy": torch.tensor(0.42)},
                "SPO": {"prototype_entropy": torch.tensor(0.73)},
            },
            "reliability": {"modality_reliability_mean": torch.tensor(0.66)},
        }

        metrics = _public_main_metrics(
            config,
            batch,
            prediction=prediction,
            router_load_by_candidate={"SPO": 0.95, "LRIO": 0.05},
            router_entropy=torch.tensor(0.0),
            candidate_loss={},
            diagnostics=diagnostics,
        )

        self.assertAlmostEqual(metrics["lrio_rank_entropy"], 0.42, places=5)
        self.assertAlmostEqual(metrics["spo_prototype_entropy"], 0.73, places=5)
        self.assertAlmostEqual(metrics["rceo_reliability_calibration"]["mean_predicted_reliability"], 0.66, places=5)
        self.assertEqual(metrics["rceo_reliability_calibration"]["source"], "model_reliability_prior")

    def test_robustness_row_uses_model_reliability_not_corruption_strength_proxy(self):
        import torch

        from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
        from scripts.multimodal.run_public_main import _robustness_row

        config = MultimodalExperimentConfig.from_file(ROOT / "configs" / "multimodal_cmu_mosei_public_main.json")
        row = _robustness_row(
            config,
            _batch(torch),
            model_name="ovha_full",
            seed=201,
            raw_metric_path=Path("raw_metrics.jsonl"),
            corruption_type="audio_noise",
            score=0.8,
            router_load_by_candidate={"SPO": 0.5, "LRIO": 0.5},
            candidate_loss={},
            model_reliability=0.33,
        )

        self.assertAlmostEqual(row["rceo_reliability"], 0.33)
        self.assertNotEqual(row["rceo_reliability"], 1.0 - row["corruption_strength"])

    def test_diagnostic_validator_respects_active_candidate_subset(self):
        from moat_ovha_torch.eval.multimodal_diagnostics import validate_diagnostic_row

        row = {
            "active_candidate_names": ["SPO", "LRIO"],
            "router_entropy": 0.5,
            "router_load_by_candidate": {"SPO": 0.6, "LRIO": 0.4},
            "router_memory_logit_norm": 0.1,
            "router_evidence_logit_norm": 0.2,
            "router_reliability_logit_norm": 0.3,
            "router_logit_parts": {"memory": 0.0, "evidence": 0.0, "reliability": 0.0},
            "candidate_loss": {"SPO": 0.2, "LRIO": 0.3},
            "adapter_params": {"SPO_temperature": 1.0, "LRIO_rank_entropy": 0.5},
            "memory_slot_norm": {"SPO": 1.1, "LRIO": 1.2},
            "candidate_diagnostics": {
                "SPO": {
                    "prototype_entropy": 0.5,
                    "top_prototype": 1.0,
                    "prototype_temperature": 1.0,
                    "candidate_loss": 0.2,
                },
                "LRIO": {
                    "rank_entropy": 0.6,
                    "rank_top_k": 1.0,
                    "pair_interaction_strength": 0.4,
                    "candidate_loss": 0.3,
                },
                "RCEO": {
                    "modality_reliability": 0.8,
                    "reliability_bias_norm": 0.1,
                    "corruption_response": 0.2,
                },
            },
            "stackability_passed": True,
        }

        report = validate_diagnostic_row(row)

        self.assertTrue(report.ok, report.errors)


def _batch(torch, *, task_type: str = "sentiment_regression"):
    from moat_ovha_torch.data.multimodal.typed_batch import (
        MultimodalEpisodeBatch,
        ProvenanceBank,
        QueryField,
        SupervisionBank,
        TokenField,
    )

    batch_size = 3
    query_count = 2
    fields = {
        "text": TokenField(
            modality="text",
            x=torch.randn(batch_size, 4, 5),
            pos=torch.linspace(0.0, 1.0, 4).view(1, 4, 1).repeat(batch_size, 1, 1),
            mask=torch.ones(batch_size, 4, dtype=torch.bool),
            quality=torch.tensor([[1.0], [0.8], [0.6]]),
        ),
        "audio": TokenField(
            modality="audio",
            x=torch.randn(batch_size, 3, 4),
            pos=torch.linspace(0.0, 1.0, 3).view(1, 3, 1).repeat(batch_size, 1, 1),
            mask=torch.ones(batch_size, 3, dtype=torch.bool),
            quality=torch.tensor([[0.9], [0.7], [0.5]]),
        ),
        "vision": TokenField(
            modality="vision",
            x=torch.randn(batch_size, 5, 3),
            pos=torch.linspace(0.0, 1.0, 5).view(1, 5, 1).repeat(batch_size, 1, 1),
            mask=torch.ones(batch_size, 5, dtype=torch.bool),
            quality=torch.tensor([[0.7], [0.9], [0.4]]),
        ),
    }
    return MultimodalEpisodeBatch(
        fields=fields,
        query=QueryField(
            x=torch.randn(batch_size, query_count, 6),
            pos=torch.linspace(0.2, 0.8, query_count).view(1, query_count, 1).repeat(batch_size, 1, 1),
            query_type=torch.zeros(batch_size, query_count, 1),
            mask=torch.ones(batch_size, query_count, dtype=torch.bool),
        ),
        target_y=torch.randn(batch_size, query_count, 1),
        target_mask=torch.ones(batch_size, query_count, dtype=torch.bool),
        task_type=task_type,
        split="train",
        source_dataset="unit_test",
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
            source_id=[f"sample_{idx}" for idx in range(batch_size)],
            original_split=["train"] * batch_size,
            raw_ref=[f"raw-{idx}" for idx in range(batch_size)],
            license_tag=["unit-test"] * batch_size,
            preprocessing_version="unit-test",
            feature_extractor_version={"text": "unit", "audio": "unit", "vision": "unit"},
            pseudo_label_version={},
        ),
        hidden={"true_active_operator": torch.zeros(batch_size, query_count, dtype=torch.long)},
    )


def _batch_with_first_sample_companions(torch, base, *, companion_seed: int):
    from dataclasses import replace

    companion = _batch(torch)
    generator = torch.Generator()
    generator.manual_seed(companion_seed)
    fields = {}
    for name, field in base.fields.items():
        replacement = base.fields[name].x.clone()
        replacement[1:] = torch.randn(
            replacement[1:].shape,
            generator=generator,
            dtype=replacement.dtype,
            device=replacement.device,
        )
        companion_field = companion.fields[name]
        fields[name] = replace(
            field,
            x=replacement,
            pos=torch.cat([field.pos[:1], companion_field.pos[1:]], dim=0),
            mask=torch.cat([field.mask[:1], companion_field.mask[1:]], dim=0),
            quality=torch.cat([field.quality[:1], companion_field.quality[1:]], dim=0),
        )
    query = replace(
        base.query,
        x=torch.cat([base.query.x[:1], companion.query.x[1:]], dim=0),
        pos=torch.cat([base.query.pos[:1], companion.query.pos[1:]], dim=0),
        query_type=torch.cat([base.query.query_type[:1], companion.query.query_type[1:]], dim=0),
        mask=torch.cat([base.query.mask[:1], companion.query.mask[1:]], dim=0),
    )
    return replace(
        base,
        fields=fields,
        query=query,
        target_y=torch.cat([base.target_y[:1], companion.target_y[1:]], dim=0),
        target_mask=torch.cat([base.target_mask[:1], companion.target_mask[1:]], dim=0),
        provenance=replace(
            base.provenance,
            source_id=[base.provenance.source_id[0], *companion.provenance.source_id[1:]],
            original_split=[base.provenance.original_split[0], *companion.provenance.original_split[1:]],
            raw_ref=[base.provenance.raw_ref[0], *companion.provenance.raw_ref[1:]],
            license_tag=[base.provenance.license_tag[0], *companion.provenance.license_tag[1:]],
        ),
    )


def _write_public_cache_fixture(root: Path, *, task_type: str):
    import numpy as np

    from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout

    layout = MultimodalCacheLayout(root, "cmu_mosei", "unit")
    cache_root = layout.root
    for directory in ("token_fields", "positions", "masks", "supervision", "provenance"):
        (cache_root / directory).mkdir(parents=True, exist_ok=True)

    batch_size = 3
    specs = {
        "text": (4, 6),
        "audio": (3, 5),
        "vision": (2, 4),
    }
    manifest: dict[str, dict[str, str]] = {}
    for offset, (modality, (token_count, dim)) in enumerate(specs.items(), start=1):
        x = np.arange(batch_size * token_count * dim, dtype=np.float32).reshape(batch_size, token_count, dim) + offset
        pos = np.linspace(0.0, 1.0, token_count, dtype=np.float32).reshape(1, token_count, 1).repeat(batch_size, axis=0)
        mask = np.ones((batch_size, token_count), dtype=bool)
        x_path = f"token_fields/{modality}_x_train.npy"
        pos_path = f"positions/{modality}_pos_train.npy"
        mask_path = f"masks/{modality}_mask_train.npy"
        np.save(cache_root / x_path, x)
        np.save(cache_root / pos_path, pos)
        np.save(cache_root / mask_path, mask)
        manifest[modality] = {"x": x_path, "pos": pos_path, "mask": mask_path}

    np.save(cache_root / "supervision" / "task_labels_train.npy", np.array([[-1.0], [0.0], [1.0]], dtype=np.float32))
    np.save(cache_root / "supervision" / "missing_modality_mask_train.npy", np.zeros((batch_size, 3), dtype=np.float32))
    (cache_root / "token_fields" / "manifest_train.json").write_text(json.dumps(manifest, sort_keys=True))
    (cache_root / "data_card.json").write_text(
        json.dumps(
            {
                "dataset_name": "cmu_mosei",
                "cache_version": "unit",
                "modalities": ["text", "audio", "vision"],
                "tasks": [task_type],
                "operator_supervision": {
                    "TLEO": "not used",
                    "SPO": "semantic prototype",
                    "LRIO": "paired modality interaction",
                    "CATO": "not used",
                    "RCEO": "reliability",
                },
                "leakage_controls": {"hidden_metadata_excluded": True},
            },
            sort_keys=True,
        )
    )
    records = [
        {
            "source_id": f"sample_{index}",
            "split": "train",
            "original_split": "train",
            "raw_ref": f"raw_{index}",
            "license_tag": "unit-test",
            "preprocessing_version": "unit-test",
        }
        for index in range(batch_size)
    ]
    (cache_root / "provenance" / "sample_records_train.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    )
    (cache_root / "provenance" / "feature_versions.json").write_text(
        json.dumps({"text": "unit", "audio": "unit", "vision": "unit"}, sort_keys=True)
    )
    (cache_root / "provenance" / "pseudo_label_versions.json").write_text(json.dumps({"version": "none"}))
    return layout


if __name__ == "__main__":
    unittest.main()
