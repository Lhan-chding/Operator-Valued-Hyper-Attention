import importlib.util
import tempfile
import unittest
from pathlib import Path

from moat_ovha_torch.config import Phase15Config


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class Phase17ProtocolContractTests(unittest.TestCase):
    def test_config_exposes_fixed_eval_episode_controls(self):
        config = Phase15Config(eval_episode_count=2, eval_episode_base=1_000_000)

        self.assertEqual(config.eval_episode_count, 2)
        self.assertEqual(config.eval_episode_base, 1_000_000)

    def test_sampling_interfaces_accept_episode_id(self):
        import inspect

        from moat_ovha_torch.data.field_episode_adapter import FieldToEpisodeAdapter
        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo

        self.assertIn("episode_id", inspect.signature(MetadataFreeOperatorZoo.sample_batch).parameters)
        self.assertIn("episode_id", inspect.signature(FieldToEpisodeAdapter.sample_episode).parameters)

    def test_controlled_v2_dataset_and_phase17_configs_exist(self):
        root = Path(__file__).resolve().parents[1]
        expected = [
            root / "data" / "ovha_controlled_v2" / "train_episode_ids.json",
            root / "data" / "ovha_controlled_v2" / "val_episode_ids.json",
            root / "data" / "ovha_controlled_v2" / "test_episode_ids.json",
            root / "data" / "ovha_controlled_v2" / "data_card.yaml",
            root / "data" / "ovha_controlled_v2" / "generator_config.json",
            root / "configs" / "phase1_7_controlled_v2_sanity.json",
            root / "configs" / "phase1_7_controlled_v2_main.json",
            root / "configs" / "phase1_7_g1_single_iid_spectral.json",
            root / "configs" / "phase1_7_g1_single_iid_local.json",
            root / "configs" / "phase1_7_g1_single_iid_separable.json",
            root / "configs" / "phase1_7_g2_g3_component_iid.json",
            root / "scripts" / "run_phase1_7_controlled_v2_sanity.sh",
            root / "docs" / "phases" / "phase_1_7.md",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)

        sanity = Phase15Config.from_file(root / "configs" / "phase1_7_controlled_v2_sanity.json")
        main = Phase15Config.from_file(root / "configs" / "phase1_7_controlled_v2_main.json")
        self.assertEqual(sanity.steps, 5000)
        self.assertGreaterEqual(sanity.eval_episode_count, 128)
        self.assertGreaterEqual(main.eval_episode_count, 512)
        self.assertIn("ovha_vector_value_big", sanity.training_model_names())
        for family in ("single_primitive_spectral", "single_primitive_local", "single_primitive_separable"):
            self.assertIn(family, sanity.families)

    def test_phase17_gated_configs_follow_root_cause_audit_order(self):
        root = Path(__file__).resolve().parents[1]

        sanity = Phase15Config.from_file(root / "configs" / "phase1_7_controlled_v2_sanity.json")
        self.assertEqual(sanity.eval_splits, ("iid",))
        self.assertEqual(sanity.router_auxiliary_loss_weight, 0.05)
        self.assertEqual(sanity.oracle_route_single_primitive_probability, 1.0)

        for name, family in (
            ("phase1_7_g1_single_iid_spectral.json", "single_primitive_spectral"),
            ("phase1_7_g1_single_iid_local.json", "single_primitive_local"),
            ("phase1_7_g1_single_iid_separable.json", "single_primitive_separable"),
        ):
            with self.subTest(config=name):
                config = Phase15Config.from_file(root / "configs" / name)
                self.assertEqual(config.families, (family,))
                self.assertEqual(config.eval_splits, ("iid",))
                self.assertLessEqual(config.steps, 2000)
                self.assertIn("ovha_full", config.training_model_names())
                self.assertIn(family.replace("single_primitive_", "") + "_only", config.training_model_names())
                self.assertEqual(config.router_auxiliary_loss_weight, 0.0)
                self.assertEqual(config.oracle_route_warmup_steps, config.steps)
                self.assertTrue(config.train_active_primitive_only)
                self.assertTrue(config.freeze_router)
                self.assertGreater(config.adapter_auxiliary_loss_weight, 0.0)

        components = Phase15Config.from_file(root / "configs" / "phase1_7_g2_g3_component_iid.json")
        self.assertEqual(components.eval_splits, ("iid",))
        self.assertEqual(components.families, ("query_piecewise_router", "context_identifiable_mixture"))
        self.assertIn("ovha_no_query_router", components.training_model_names())
        self.assertIn("ovha_shuffled_context_memory", components.training_model_names())


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed; Phase 1.7 tensor protocol tests skipped.")
class Phase17ControlledV2ProtocolTests(unittest.TestCase):
    def test_episode_id_sampling_is_reproducible_without_repeating_train_batches(self):
        import torch

        from moat_ovha_torch.data.episodes import hash_model_inputs
        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo

        zoo = MetadataFreeOperatorZoo(seed=123)
        kwargs = dict(
            batch_size=2,
            num_demos=2,
            context_points=5,
            support_points=12,
            query_points=7,
            family="single_primitive_representable",
            split="iid",
            mode="operator_transfer",
            device="cpu",
        )

        first, hidden_first = zoo.sample_batch(**kwargs, episode_id=0)
        repeated, hidden_repeated = zoo.sample_batch(**kwargs, episode_id=0)
        next_batch, hidden_next = zoo.sample_batch(**kwargs, episode_id=1)

        self.assertEqual(hash_model_inputs(first), hash_model_inputs(repeated))
        self.assertNotEqual(hash_model_inputs(first), hash_model_inputs(next_batch))
        self.assertTrue(torch.equal(first.target_y, repeated.target_y))
        self.assertFalse(torch.equal(first.target_y, next_batch.target_y))
        self.assertEqual(hidden_first.oracle_hints["episode_id"], 0)
        self.assertEqual(hidden_repeated.oracle_hints["episode_id"], 0)
        self.assertEqual(hidden_next.oracle_hints["episode_id"], 1)

    def test_controlled_v2_hidden_oracles_are_aligned_to_model_order(self):
        import torch

        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo

        zoo = MetadataFreeOperatorZoo(seed=8)
        batch, hidden = zoo.sample_batch(
            batch_size=2,
            num_demos=2,
            context_points=4,
            support_points=16,
            query_points=9,
            family="query_piecewise_router",
            split="iid",
            mode="operator_transfer",
            device="cpu",
            episode_id=77,
        )

        hints = hidden.oracle_hints
        self.assertEqual(hints["primitive_order"], ("spectral", "local", "separable"))
        weights = hints["true_component_weight_by_q"]
        outputs = hints["true_primitive_outputs_by_q"]
        self.assertEqual(tuple(weights.shape), (2, 9, 3))
        self.assertEqual(tuple(outputs.shape), (2, 9, 3, 1))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 9), atol=1e-5))
        reconstructed = (weights.unsqueeze(-1) * outputs).sum(dim=-2)
        self.assertTrue(torch.allclose(reconstructed, batch.target_y, atol=1e-5))

    def test_single_primitive_families_are_not_mixed(self):
        import torch

        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo

        expected = {
            "single_primitive_spectral": torch.tensor([1.0, 0.0, 0.0]),
            "single_primitive_local": torch.tensor([0.0, 1.0, 0.0]),
            "single_primitive_separable": torch.tensor([0.0, 0.0, 1.0]),
        }
        zoo = MetadataFreeOperatorZoo(seed=12)
        for family, expected_weight in expected.items():
            with self.subTest(family=family):
                batch, hidden = zoo.sample_batch(
                    batch_size=3,
                    num_demos=1,
                    context_points=4,
                    support_points=8,
                    query_points=5,
                    family=family,
                    split="iid",
                    mode="operator_transfer",
                    device="cpu",
                    episode_id=4,
                )
                weights = hidden.oracle_hints["true_component_weight_by_q"]
                expected_tensor = expected_weight.view(1, 1, 3).expand_as(weights)
                self.assertTrue(torch.equal(weights, expected_tensor))
                outputs = hidden.oracle_hints["true_primitive_outputs_by_q"]
                reconstructed = (weights.unsqueeze(-1) * outputs).sum(dim=-2)
                self.assertTrue(torch.allclose(reconstructed, batch.target_y, atol=1e-5))

    def test_model_aligned_primitives_reproduce_generator_with_true_params(self):
        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
        from moat_ovha_torch.eval.oracle_metrics import model_primitive_outputs_from_true_params
        from moat_ovha_torch.train.metrics import relative_l2

        zoo = MetadataFreeOperatorZoo(seed=34)
        for family in (
            "single_primitive_spectral",
            "single_primitive_local",
            "single_primitive_separable",
            "query_piecewise_router",
            "context_identifiable_mixture",
        ):
            with self.subTest(family=family):
                batch, hidden = zoo.sample_batch(
                    batch_size=3,
                    num_demos=2,
                    context_points=5,
                    support_points=24,
                    query_points=11,
                    family=family,
                    split="iid",
                    mode="operator_transfer",
                    device="cpu",
                    episode_id=19,
                    controlled_generator_variant="model_aligned",
                )

                model_outputs = model_primitive_outputs_from_true_params(
                    batch,
                    hidden,
                    ("spectral", "local", "separable"),
                )
                self.assertIsNotNone(model_outputs)
                weights = hidden.oracle_hints["true_component_weight_by_q"]
                reconstructed = (weights.unsqueeze(-1) * model_outputs).sum(dim=-2)
                self.assertLess(float(relative_l2(reconstructed, batch.target_y).max()), 1e-5)

    def test_model_aligned_iid_params_stay_inside_sanity_adapter_domain(self):
        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo

        zoo = MetadataFreeOperatorZoo(seed=35)
        batch, hidden = zoo.sample_batch(
            batch_size=16,
            num_demos=1,
            context_points=4,
            support_points=16,
            query_points=6,
            family="context_identifiable_mixture",
            split="iid",
            mode="operator_transfer",
            device="cpu",
            episode_id=23,
            controlled_generator_variant="model_aligned",
        )
        self.assertEqual(tuple(batch.target_y.shape), (16, 6, 1))
        params = hidden.oracle_hints["true_operator_tensors"]

        self.assertGreaterEqual(float(params["gain"].min()), 0.9)
        self.assertLessEqual(float(params["gain"].max()), 1.1)
        self.assertGreaterEqual(float(params["lengthscale"].min()), 0.08)
        self.assertLessEqual(float(params["lengthscale"].max()), 0.20)

    def test_hyper_adapter_starts_from_identity_primitive_defaults(self):
        import torch

        from moat_ovha_torch.models.hyper_adapter import HyperAdapter

        adapter = HyperAdapter(("spectral", "local", "separable"), d_model=8)
        memory = torch.zeros(2, 3, 8)
        target_q = torch.linspace(0.0, 1.0, 5).view(1, 5, 1).repeat(2, 1, 1)

        params = adapter(memory, target_q)

        self.assertTrue(torch.allclose(params["spectral"].scale, torch.ones(2, 5, 1), atol=1e-6))
        self.assertTrue(torch.allclose(params["spectral"].bias, torch.zeros(2, 5, 1), atol=1e-6))
        self.assertIsNone(params["spectral"].spectral_frequency)
        self.assertIsNone(params["spectral"].spectral_phase)
        self.assertTrue(torch.allclose(params["spectral"].spectral_mode_logits, torch.zeros(2, 5, 4), atol=1e-6))
        self.assertTrue(torch.allclose(params["local"].local_lengthscale, torch.full((2, 5, 1), -2.0), atol=1e-6))
        self.assertIsNone(params["local"].local_shift)
        self.assertTrue(torch.allclose(params["separable"].separable_rank_logits, torch.zeros(2, 5, 4), atol=1e-6))

    def test_context_encoder_uses_primitive_aligned_feature_bank(self):
        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
        from moat_ovha_torch.models.context_encoder import ContextTokenEncoder

        zoo = MetadataFreeOperatorZoo(seed=44)
        batch, _ = zoo.sample_batch(
            batch_size=2,
            num_demos=3,
            context_points=5,
            support_points=16,
            query_points=7,
            family="query_piecewise_router",
            split="iid",
            mode="operator_transfer",
            device="cpu",
            episode_id=5,
        )
        encoder = ContextTokenEncoder(d_model=12)
        tokens = encoder(batch)

        self.assertEqual(tuple(tokens.shape), (2, 3, 5, 12))
        self.assertEqual(encoder.input_dim, 66)
        self.assertEqual(encoder.candidate_feature_count, 12)
        self.assertEqual(encoder.evidence_feature_count, 36)

    def test_training_records_router_auxiliary_loss_when_oracles_are_available(self):
        import json

        from moat_ovha_torch.train.trainer import run_training_for_model

        with tempfile.TemporaryDirectory() as tmp:
            config = Phase15Config(
                seed=45,
                output_dir=Path(tmp),
                steps=1,
                batch_size=2,
                support_points=12,
                query_points=5,
                num_demos=1,
                context_points=4,
                d_model=16,
                memory_tokens=2,
                families=("query_piecewise_router",),
                train_models=("ovha_full",),
                eval_models=("ovha_full",),
                router_auxiliary_loss_weight=0.05,
            )
            run_training_for_model(config, "ovha_full")
            row = json.loads((Path(tmp) / "train_metrics" / "ovha_full" / "seed_45.jsonl").read_text().splitlines()[0])

        self.assertGreater(row["router_auxiliary_loss"], 0.0)
        self.assertAlmostEqual(row["loss"], row["prediction_loss"] + 0.05 * row["router_auxiliary_loss"], places=6)

    def test_training_rows_record_episode_and_batch_hashes(self):
        from moat_ovha_torch.train.trainer import run_training_for_model

        with tempfile.TemporaryDirectory() as tmp:
            config = Phase15Config(
                seed=5,
                output_dir=Path(tmp),
                steps=2,
                batch_size=1,
                support_points=8,
                query_points=4,
                num_demos=1,
                context_points=3,
                d_model=16,
                memory_tokens=2,
                families=("single_primitive_representable",),
                train_models=("ovha_full",),
                eval_models=("ovha_full",),
            )
            run_training_for_model(config, "ovha_full")
            rows = [
                __import__("json").loads(line)
                for line in (Path(tmp) / "train_metrics" / "ovha_full" / "seed_5.jsonl").read_text().splitlines()
            ]

        self.assertEqual([row["episode_id"] for row in rows], [0, 1])
        self.assertNotEqual(rows[0]["batch_hash"], rows[1]["batch_hash"])
        for row in rows:
            self.assertRegex(row["context_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(row["target_hash"], r"^[0-9a-f]{64}$")

    def test_evaluation_runs_fixed_episode_list_and_oracle_diagnostics(self):
        from moat_ovha_torch.eval.evaluator import run_evaluation

        with tempfile.TemporaryDirectory() as tmp:
            config = Phase15Config(
                seed=9,
                output_dir=Path(tmp),
                steps=1,
                batch_size=1,
                support_points=8,
                query_points=4,
                num_demos=1,
                context_points=3,
                d_model=16,
                memory_tokens=2,
                families=("query_piecewise_router",),
                eval_splits=("iid",),
                train_models=("ovha_full",),
                eval_models=("ovha_full",),
                require_checkpoint=False,
                eval_episode_count=2,
                eval_episode_base=1_000_000,
            )
            run_evaluation(config)
            rows = [
                __import__("json").loads(line)
                for line in (Path(tmp) / "eval_metrics" / "ovha_full" / "seed_9.jsonl").read_text().splitlines()
            ]

        self.assertEqual([row["episode_id"] for row in rows], [1_000_000, 1_000_001])
        for row in rows:
            self.assertIn("router_true_weight_mae", row)
            self.assertIn("router_true_weight_kl", row)
            self.assertIn("memory_swap_delta", row)
            self.assertIn("oracle_router_upper_bound_relative_l2", row)
            self.assertIn("model_primitive_true_param_relative_l2", row)
            self.assertRegex(row["batch_hash"], r"^[0-9a-f]{64}$")

    def test_field_adapter_operator_transfer_uses_grouped_non_target_demos(self):
        import torch

        from moat_ovha_torch.data.field_episode_adapter import FieldToEpisodeAdapter

        inputs = torch.arange(48, dtype=torch.float32).view(4, 12, 1)
        outputs = inputs * 10.0
        coordinates = torch.linspace(0.0, 1.0, 12).view(12, 1)
        sample_batch = {
            "input_field": inputs,
            "output_field": outputs,
            "coordinates": coordinates,
            "operator_group_id": torch.tensor([0, 0, 1, 1]),
        }
        adapter = FieldToEpisodeAdapter(seed=0)

        episode = adapter.sample_episode(
            sample_batch,
            num_demos=1,
            context_points=3,
            query_points=4,
            mode="operator_transfer",
            context_sampling="stratified",
            episode_id=3,
        )

        demo_indices = adapter.last_record.demo_indices
        target_indices = adapter.last_record.target_indices
        self.assertTrue(torch.equal(sample_batch["operator_group_id"][demo_indices[:, 0]], sample_batch["operator_group_id"][target_indices]))
        self.assertTrue(torch.all(demo_indices[:, 0] != target_indices))
        self.assertFalse(torch.equal(episode.context_u[:, 0], episode.target_u))


if __name__ == "__main__":
    unittest.main()
