import importlib.util
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


class Phase17AdapterRouterStaticContractTests(unittest.TestCase):
    def test_new_modules_and_adapter_collapse_gate_script_exist(self):
        expected = [
            ROOT / "moat_ovha_torch" / "models" / "evidence.py",
            ROOT / "moat_ovha_torch" / "models" / "joint_router_adapter.py",
            ROOT / "moat_ovha_torch" / "train" / "controlled_aux_losses.py",
            ROOT / "scripts" / "run_phase1_7_adapter_collapse_gate.py",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)

    def test_phase15_config_exposes_staged_training_controls(self):
        source = (ROOT / "moat_ovha_torch" / "config.py").read_text()
        for field in (
            "adapter_auxiliary_loss_weight",
            "primitive_output_loss_weight",
            "oracle_routed_prediction_loss_weight",
            "oracle_route_warmup_steps",
            "train_active_primitive_only",
            "freeze_adapter",
            "freeze_primitives",
            "router_query_residual_loss_weight",
        ):
            with self.subTest(field=field):
                self.assertIn(field, source)

    def test_trainer_records_loss_components_and_grad_norms(self):
        source = (ROOT / "moat_ovha_torch" / "train" / "trainer.py").read_text()
        for token in (
            "controlled_v2_adapter_losses",
            "route_override",
            "active_primitive_mask",
            "loss_components",
            "grad_norms",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed; adapter/router patch tests skipped.")
class Phase17AdapterRouterPatchTests(unittest.TestCase):
    def test_router_returns_logits_prior_and_query_residual(self):
        import torch

        from moat_ovha_torch.models.router import PrimitiveRouter, RouterOutput

        router = PrimitiveRouter(("spectral", "local", "separable"), d_model=8)
        memory = torch.zeros(2, 3, 8)
        target_q = torch.linspace(0.0, 1.0, 5).view(1, 5, 1).repeat(2, 1, 1)

        output = router(memory, target_q)

        self.assertIsInstance(output, RouterOutput)
        self.assertEqual(tuple(output.weights.shape), (2, 5, 3))
        self.assertEqual(tuple(output.logits.shape), (2, 5, 3))
        self.assertEqual(tuple(output.context_prior_logits.shape), (2, 3))
        self.assertEqual(tuple(output.query_residual_logits.shape), (2, 5, 3))
        self.assertTrue(torch.allclose(output.weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6))

    def test_hyper_adapter_globalizes_model_aligned_operator_params(self):
        import torch

        from moat_ovha_torch.models.hyper_adapter import HyperAdapter

        adapter = HyperAdapter(
            ("spectral", "local", "separable"),
            d_model=8,
            controlled_generator_variant="model_aligned",
        )
        memory = torch.zeros(2, 3, 8)
        target_q = torch.linspace(0.0, 1.0, 5).view(1, 5, 1).repeat(2, 1, 1)

        params = adapter(memory, target_q)

        spectral = params["spectral"]
        self.assertIsNone(spectral.spectral_frequency)
        self.assertIsNone(spectral.spectral_phase)
        self.assertEqual(spectral.scope["scale"], "episode")
        self.assertEqual(spectral.scope["spectral_mode_logits"], "episode")
        self.assertIn("selected", spectral.raw)
        self.assertTrue(torch.allclose(spectral.scale, torch.ones(2, 5, 1), atol=1e-6))
        self.assertTrue(torch.allclose(spectral.spectral_mode_logits.var(dim=1), torch.zeros(2, 4), atol=1e-8))

        local = params["local"]
        self.assertEqual(local.scope["local_lengthscale"], "episode")
        self.assertTrue(torch.allclose(local.local_lengthscale.var(dim=1), torch.zeros(2, 1), atol=1e-8))

        separable = params["separable"]
        self.assertEqual(separable.scope["separable_rank_logits"], "episode")
        self.assertTrue(torch.allclose(separable.separable_rank_logits.var(dim=1), torch.zeros(2, 4), atol=1e-8))

    def test_hyper_adapter_accepts_direct_primitive_evidence_features(self):
        import torch

        from moat_ovha_torch.models.evidence import EvidenceBank
        from moat_ovha_torch.models.hyper_adapter import HyperAdapter

        adapter = HyperAdapter(("spectral", "local", "separable"), d_model=8)
        memory = {name: torch.zeros(2, 2, 8) for name in ("global", "spectral", "local", "separable")}
        target_q = torch.linspace(0.0, 1.0, 5).view(1, 5, 1).repeat(2, 1, 1)
        evidence = _evidence_bank(torch, batch_size=2)
        captured = {}

        def capture_global(features):
            captured["separable_tail"] = features[:, -12:].detach().clone()
            return torch.zeros(features.shape[0], 1, adapter._output_dim("separable"))

        adapter.global_heads["separable"].forward = capture_global

        adapter(memory, target_q, evidence_bank=evidence)

        expected_tail = torch.cat(
            [
                evidence.ls_coeff["separable"],
                evidence.residual_energy["separable"],
                evidence.uncertainty["separable"],
            ],
            dim=-1,
        )
        self.assertTrue(torch.equal(captured["separable_tail"], expected_tail))

    def test_hyper_adapter_scale_range_covers_parameter_holdout_gain(self):
        import torch

        from moat_ovha_torch.models.hyper_adapter import HyperAdapter

        adapter = HyperAdapter(("separable",), d_model=8)
        memory = torch.zeros(2, 2, 8)
        target_q = torch.linspace(0.0, 1.0, 5).view(1, 5, 1).repeat(2, 1, 1)

        with torch.no_grad():
            adapter.global_heads["separable"].net[-1].bias[0] = 2.0

        params = adapter(memory, target_q)

        self.assertGreater(float(params["separable"].scale.detach().max()), 1.5)

    def test_local_lengthscale_uses_public_evidence_candidate_prior(self):
        import torch

        from moat_ovha_torch.models.hyper_adapter import HyperAdapter

        adapter = HyperAdapter(("local",), d_model=8)
        memory = torch.zeros(2, 2, 8)
        target_q = torch.linspace(0.0, 1.0, 5).view(1, 5, 1).repeat(2, 1, 1)
        evidence = _evidence_bank(torch, batch_size=2)
        evidence.gram["local"] = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        evidence.corr["local"] = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=target_q.dtype,
        )

        params = adapter(memory, target_q, evidence_bank=evidence)
        lengthscale = torch.nn.functional.softplus(params["local"].local_lengthscale.detach()) + 1e-3

        self.assertGreater(float(lengthscale[0].mean()), 0.18)
        self.assertLess(float(lengthscale[1].mean()), 0.10)

    def test_ovha_forward_accepts_oracle_route_and_active_primitive_mask(self):
        import torch

        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
        from moat_ovha_torch.models.ovha import OVHAMetaOperator

        zoo = MetadataFreeOperatorZoo(seed=123)
        batch, hidden = zoo.sample_batch(
            batch_size=2,
            num_demos=1,
            context_points=4,
            support_points=12,
            query_points=5,
            family="single_primitive_spectral",
            split="iid",
            mode="operator_transfer",
            device="cpu",
            episode_id=7,
        )
        true_weights = hidden.oracle_hints["true_component_weight_by_q"]
        model = OVHAMetaOperator(d_model=16, memory_tokens=2)

        output = model(batch, route_override=true_weights, active_primitive_mask=true_weights > 0.5)

        self.assertTrue(torch.equal(output.primitive_weights, true_weights))
        self.assertEqual(tuple(output.router_logits.shape), (2, 5, 3))
        self.assertIsNotNone(output.adapter_params)
        self.assertIsNotNone(output.memory_bank)
        per_primitive = output.diagnostics["per_primitive_outputs_train"]
        self.assertEqual(tuple(per_primitive.shape), (2, 5, 3, 1))
        self.assertGreater(float(per_primitive[..., 0, :].detach().abs().sum()), 0.0)
        self.assertEqual(float(per_primitive[..., 1:, :].detach().abs().sum()), 0.0)

    def test_route_override_conditions_hyper_adapter_posterior_features(self):
        import torch

        from moat_ovha_torch.models.joint_router_adapter import JointRouterAdapter
        from moat_ovha_torch.models.primitives.base import PrimitiveParams

        adapter = JointRouterAdapter(("spectral", "local", "separable"), d_model=8)
        memory_bank = {name: torch.zeros(2, 2, 8) for name in adapter.router.primitive_names}
        memory_bank["global"] = torch.zeros(2, 2, 8)
        target_q = torch.linspace(0.0, 1.0, 5).view(1, 5, 1).repeat(2, 1, 1)
        route_override = torch.zeros(2, 5, 3)
        route_override[..., 2] = 1.0
        captured = {}

        def capture_forward(memory, target_q, router_out=None, evidence_bank=None):
            captured["weights"] = router_out.weights.detach().clone()
            return {name: PrimitiveParams() for name in adapter.router.primitive_names}

        adapter.hyper_adapter.forward = capture_forward

        adapter(memory_bank, target_q, route_override=route_override)

        self.assertTrue(torch.equal(captured["weights"], route_override))

    def test_controlled_v2_adapter_losses_expose_required_components(self):
        import torch

        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
        from moat_ovha_torch.models.ovha import OVHAMetaOperator
        from moat_ovha_torch.train.controlled_aux_losses import controlled_v2_adapter_losses

        zoo = MetadataFreeOperatorZoo(seed=456)
        batch, hidden = zoo.sample_batch(
            batch_size=2,
            num_demos=1,
            context_points=4,
            support_points=12,
            query_points=5,
            family="query_piecewise_router",
            split="iid",
            mode="operator_transfer",
            device="cpu",
            episode_id=9,
        )
        model = OVHAMetaOperator(d_model=16, memory_tokens=2)
        output = model(batch, route_override=hidden.oracle_hints["true_component_weight_by_q"])

        losses = controlled_v2_adapter_losses(output, hidden, model.primitive_names, batch)

        for key in (
            "adapter_param_loss",
            "primitive_output_loss",
            "oracle_routed_prediction_loss",
            "param_scope_loss",
            "spectral_mode_kl",
            "separable_rank_kl",
            "gain_huber",
            "bias_huber",
            "local_lengthscale_log_huber",
        ):
            with self.subTest(key=key):
                self.assertIn(key, losses)
                self.assertTrue(torch.isfinite(losses[key]))

    def test_controlled_v2_aux_losses_mask_inactive_single_primitive_terms(self):
        import torch

        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
        from moat_ovha_torch.models.ovha import OVHAMetaOperator
        from moat_ovha_torch.train.controlled_aux_losses import controlled_v2_adapter_losses

        zoo = MetadataFreeOperatorZoo(seed=654)
        batch, hidden = zoo.sample_batch(
            batch_size=2,
            num_demos=1,
            context_points=4,
            support_points=12,
            query_points=5,
            family="single_primitive_spectral",
            split="iid",
            mode="operator_transfer",
            device="cpu",
            episode_id=21,
        )
        true_weights = hidden.oracle_hints["true_component_weight_by_q"]
        model = OVHAMetaOperator(d_model=16, memory_tokens=2)
        output = model(batch, route_override=true_weights, active_primitive_mask=true_weights > 0.5)

        losses = controlled_v2_adapter_losses(output, hidden, model.primitive_names, batch)

        learned = output.diagnostics["per_primitive_outputs_train"]
        true_outputs = hidden.oracle_hints["true_primitive_outputs_by_q"].to(learned.device)
        expected_spectral_only = _relative_mse(learned[..., 0, :], true_outputs[..., 0, :])

        self.assertTrue(torch.allclose(losses["separable_rank_kl"], torch.zeros_like(losses["separable_rank_kl"])))
        self.assertTrue(
            torch.allclose(losses["local_lengthscale_log_huber"], torch.zeros_like(losses["local_lengthscale_log_huber"]))
        )
        self.assertTrue(torch.allclose(losses["primitive_output_loss"], expected_spectral_only, atol=1e-6))

    def test_mlp_expert_moe_builds_two_distinct_experts(self):
        from moat_ovha_torch.models.baselines import build_model

        model = build_model("mlp_expert_moe", d_model=8, memory_tokens=2)

        self.assertEqual(len(model.primitive_names), 2)
        self.assertEqual(len(set(model.primitive_names)), 2)
        self.assertEqual(len(model.primitives), 2)

    def test_ablation_wrappers_accept_trainer_route_kwargs(self):
        import torch

        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
        from moat_ovha_torch.models.baselines import build_model

        zoo = MetadataFreeOperatorZoo(seed=321)
        batch, hidden = zoo.sample_batch(
            batch_size=2,
            num_demos=1,
            context_points=4,
            support_points=12,
            query_points=5,
            family="query_piecewise_router",
            split="iid",
            mode="operator_transfer",
            device="cpu",
            episode_id=13,
        )
        true_weights = hidden.oracle_hints["true_component_weight_by_q"]

        for model_name in ("simple_stack", "ovha_shuffled_context_memory"):
            with self.subTest(model=model_name):
                model = build_model(model_name, d_model=16, memory_tokens=2)
                output = model(batch, route_override=true_weights, active_primitive_mask=true_weights > 0.5)
                self.assertEqual(tuple(output.y_hat.shape), (2, 5, 1))
                self.assertTrue(torch.isfinite(output.y_hat).all())

    def test_oracle_metrics_include_router_adapter_matrix_and_adapter_stats(self):
        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
        from moat_ovha_torch.eval.oracle_metrics import controlled_oracle_metrics
        from moat_ovha_torch.models.ovha import OVHAMetaOperator

        zoo = MetadataFreeOperatorZoo(seed=789)
        batch, hidden = zoo.sample_batch(
            batch_size=2,
            num_demos=1,
            context_points=4,
            support_points=12,
            query_points=5,
            family="context_identifiable_mixture",
            split="iid",
            mode="operator_transfer",
            device="cpu",
            episode_id=11,
        )
        model = OVHAMetaOperator(d_model=16, memory_tokens=2)
        output = model(batch)

        metrics = controlled_oracle_metrics(output, batch.target_y, hidden, model.primitive_names, batch=batch)

        for key in (
            "true_router_learned_adapter_relative_l2",
            "learned_router_true_adapter_relative_l2",
            "true_router_true_adapter_relative_l2",
            "learned_primitive_true_param_gap_spectral",
            "adapter_spectral_mode_true_kl",
            "adapter_separable_rank_true_kl",
            "adapter_local_lengthscale_log_mae",
        ):
            with self.subTest(key=key):
                self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()


def _relative_mse(prediction, target):
    numerator = (prediction - target).square().mean()
    denominator = target.square().mean().clamp_min(1e-8)
    return numerator / denominator


def _evidence_bank(torch, batch_size):
    from moat_ovha_torch.models.evidence import EvidenceBank

    zeros = torch.zeros(batch_size, 4)
    return EvidenceBank(
        basis_outputs={},
        residuals={},
        gram={},
        corr={},
        ls_coeff={
            "spectral": zeros + 1.0,
            "local": zeros + 2.0,
            "separable": zeros + 3.0,
        },
        residual_energy={
            "spectral": zeros + 4.0,
            "local": zeros + 5.0,
            "separable": zeros + 6.0,
        },
        uncertainty={
            "spectral": zeros + 7.0,
            "local": zeros + 8.0,
            "separable": zeros + 9.0,
        },
        point_features=torch.zeros(batch_size, 1, 1, 36),
    )
