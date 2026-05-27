import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from moat_ovha_torch.config import Phase15Config


ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class Phase16BenchmarkProtocolArtifactTests(unittest.TestCase):
    def test_phase16_configs_docs_and_scripts_exist(self):
        expected = [
            ROOT / "docs" / "phases" / "phase_1_6.md",
            ROOT / "docs" / "benchmark_protocol.md",
            ROOT / "docs" / "execution_policy_mac_vs_a800.md",
            ROOT / "configs" / "phase1_6_cpu_integrity_smoke.json",
            ROOT / "configs" / "phase1_6_gpu_integrity_short.json",
            ROOT / "configs" / "phase1_6_gpu_component_main.json",
            ROOT / "configs" / "phase1_6_gpu_public_pilot.json",
            ROOT / "configs" / "phase1_6_gpu_public_full_optional.json",
            ROOT / "scripts" / "run_phase1_6_cpu_integrity_smoke.sh",
            ROOT / "scripts" / "run_phase1_6_gpu_integrity_short.sh",
            ROOT / "scripts" / "run_phase1_6_gpu_component_main.sh",
            ROOT / "scripts" / "run_phase1_6_gpu_public_pilot.sh",
            ROOT / "scripts" / "summarize_phase1_6.py",
        ]

        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), path)

    def test_phase16_cpu_config_stays_mac_safe(self):
        config = Phase15Config.from_file(ROOT / "configs" / "phase1_6_cpu_integrity_smoke.json")

        self.assertEqual(config.device, "cpu")
        self.assertLessEqual(config.steps, 50)
        self.assertLessEqual(config.batch_size, 4)
        self.assertLessEqual(config.support_points, 32)
        self.assertLessEqual(config.query_points, 32)
        self.assertEqual(config.training_model_names(), ("ovha_full", "transformer_only"))
        self.assertEqual(config.evaluation_model_names(), ("ovha_full", "transformer_only"))
        self.assertTrue(config.require_checkpoint)

    def test_benchmark_registry_contains_required_cards(self):
        from moat_ovha_torch.data.benchmark_registry import get_default_benchmark_registry

        registry = get_default_benchmark_registry()

        for name in (
            "pdebench_burgers_1d",
            "pdebench_darcy_2d",
            "fno_burgers",
            "fno_darcy",
            "mechanical_mnist_small",
        ):
            with self.subTest(name=name):
                spec = registry.get(name)
                self.assertEqual(spec.name, name)
                self.assertTrue(spec.requires_download)
                self.assertIn(spec.domain, {"pde", "material"})
                self.assertTrue(spec.supports_context_episodes)

    def test_missing_public_benchmark_data_fails_clearly(self):
        from moat_ovha_torch.data.public_benchmarks.pdebench_loader import PDEBenchSubsetLoader

        with tempfile.TemporaryDirectory() as tmp:
            loader = PDEBenchSubsetLoader(Path(tmp) / "missing")

            with self.assertRaisesRegex(FileNotFoundError, "PDEBench subset not found"):
                loader.load_split("train")


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed; Phase 1.6 tensor protocol tests skipped.")
class Phase16BenchmarkProtocolTorchTests(unittest.TestCase):
    def test_controlled_stress_hidden_component_weights_do_not_enter_inputs(self):
        import torch

        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo

        zoo = MetadataFreeOperatorZoo(seed=7)
        batch, hidden = zoo.sample_batch(
            batch_size=2,
            num_demos=1,
            context_points=4,
            support_points=8,
            query_points=9,
            family="query_piecewise_composition_family",
            split="iid",
            mode="operator_transfer",
            device="cpu",
        )

        model_inputs = batch.model_inputs()
        self.assertNotIn("true_component_weight_by_q", model_inputs)
        weights = hidden.oracle_hints["true_component_weight_by_q"]
        self.assertEqual(tuple(weights.shape), (2, 9, 3))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 9), atol=1e-5))

    def test_field_to_episode_adapter_deterministic_stratified_sampling(self):
        import torch

        from moat_ovha_torch.data.field_episode_adapter import FieldToEpisodeAdapter

        inputs = torch.arange(24, dtype=torch.float32).view(2, 12, 1)
        outputs = inputs * 2.0
        coordinates = torch.linspace(0.0, 1.0, 12).view(12, 1)
        sample_batch = {"input_field": inputs, "output_field": outputs, "coordinates": coordinates}
        adapter = FieldToEpisodeAdapter(seed=11)

        first = adapter.sample_episode(
            sample_batch,
            num_demos=1,
            context_points=4,
            query_points=5,
            mode="same_sample",
            context_sampling="stratified",
        )
        second = adapter.sample_episode(
            sample_batch,
            num_demos=1,
            context_points=4,
            query_points=5,
            mode="same_sample",
            context_sampling="stratified",
        )

        self.assertTrue(torch.equal(first.context_q, second.context_q))
        self.assertTrue(torch.equal(first.target_q, second.target_q))
        self.assertEqual(tuple(first.context_q.shape), (2, 1, 4, 1))
        self.assertEqual(tuple(first.target_q.shape), (2, 5, 1))


if __name__ == "__main__":
    unittest.main()
