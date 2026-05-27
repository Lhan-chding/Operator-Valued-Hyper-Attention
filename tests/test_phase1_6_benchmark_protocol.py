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
            ROOT / "configs" / "phase1_6_gpu_diagnostic_short.json",
            ROOT / "configs" / "phase1_6_gpu_component_main.json",
            ROOT / "configs" / "phase1_6_gpu_public_pilot.json",
            ROOT / "configs" / "phase1_6_gpu_public_full_optional.json",
            ROOT / "scripts" / "run_phase1_6_cpu_integrity_smoke.sh",
            ROOT / "scripts" / "run_phase1_6_gpu_integrity_short.sh",
            ROOT / "scripts" / "run_phase1_6_gpu_diagnostic_short.sh",
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

    def test_phase16_gpu_scripts_default_to_gpu_three(self):
        for script in (
            ROOT / "scripts" / "run_phase1_6_gpu_integrity_short.sh",
            ROOT / "scripts" / "run_phase1_6_gpu_diagnostic_short.sh",
            ROOT / "scripts" / "run_phase1_6_gpu_component_main.sh",
            ROOT / "scripts" / "run_phase1_6_gpu_public_pilot.sh",
        ):
            with self.subTest(script=script):
                self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"', script.read_text())

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

    def test_phase16_summary_separates_checkpoint_integrity_from_scientific_go(self):
        from scripts.summarize_phase1_6 import _build_summary

        eval_rows = [
            {
                "family": "query_piecewise_composition_family",
                "model_name": model,
                "seed": 41,
                "relative_l2": relative_l2,
                "checkpoint_loaded": True,
                "checkpoint_train_steps": 2000,
            }
            for model, relative_l2 in (
                ("ovha_full", 1.35),
                ("local_only", 1.19),
                ("separable_only", 1.30),
                ("spectral_only", 1.55),
                ("ovha_vector_value_big", 1.21),
                ("ovha_no_memory", 0.98),
                ("ovha_no_query_router", 1.33),
                ("ovha_no_hyper_adapter", 1.03),
            )
        ]
        train_rows = [
            {
                "model": row["model_name"],
                "seed": row["seed"],
                "relative_l2": 0.75,
            }
            for row in eval_rows
        ]

        summary = _build_summary(eval_rows, train_rows)

        self.assertTrue(all(row["checkpoint_loaded"] for row in summary["checkpoint_integrity"]))
        self.assertEqual(summary["controlled_stress"][0]["conclusion"], "negative component signal")
        self.assertIn("Scientific No-Go", summary["go_no_go"])
        self.assertIn("checkpoint path is wired", summary["go_no_go"])

    def test_phase16_summary_includes_diagnostic_signals_from_jsonl(self):
        from scripts.summarize_phase1_6 import summarize

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "environment.json").write_text(json.dumps({"torch_available": True, "device": "cuda", "config_hash": "abc"}))
            eval_row = {
                "family": "query_piecewise_composition_family",
                "model_name": "ovha_full",
                "seed": 41,
                "relative_l2": 1.0,
                "checkpoint_loaded": True,
                "checkpoint_train_steps": 2000,
            }
            train_row = {"model": "ovha_full", "seed": 41, "relative_l2": 0.8}
            diagnostic_row = {
                "family": "query_piecewise_composition_family",
                "model_name": "ovha_full",
                "seed": 41,
                "primitive_entropy": 1.0,
                "memory_norm": 2.0,
                "adapter_norms": {"spectral": 0.1, "separable": 0.2, "local": 0.0},
                "primitive_load": {"spectral": 0.5, "separable": 0.3, "local": 0.2},
            }
            (root / "eval_metrics.jsonl").write_text(json.dumps(eval_row) + "\n")
            (root / "train_metrics.jsonl").write_text(json.dumps(train_row) + "\n")
            (root / "diagnostics.jsonl").write_text(json.dumps(diagnostic_row) + "\n")

            report = summarize(root)
            summary = json.loads((root / "phase1_6_summary.json").read_text())
            report_text = report.read_text()

        self.assertEqual(summary["diagnostic_signals"][0]["model"], "ovha_full")
        self.assertEqual(summary["diagnostic_signals"][0]["family"], "query_piecewise_composition_family")
        self.assertAlmostEqual(summary["diagnostic_signals"][0]["mean_adapter_norm"], 0.1)
        self.assertIn("## Diagnostic Signals", report_text)
        self.assertIn("spectral=0.500000", report_text)


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

    def test_evaluator_diagnostic_row_exposes_router_memory_and_adapter_signals(self):
        import types
        import torch

        from moat_ovha_torch.eval.evaluator import _diagnostic_row

        output = types.SimpleNamespace(
            primitive_weights=torch.tensor([[[0.7, 0.3], [0.5, 0.5]]]),
            diagnostics={
                "primitive_entropy": torch.tensor(0.61),
                "memory_norms": torch.tensor(2.5),
                "adapter_norms": {"spectral": torch.tensor(1.2), "separable": torch.tensor(0.3)},
            },
        )

        row = _diagnostic_row(
            split="iid",
            family="query_piecewise_composition_family",
            model_name="ovha_full",
            seed=41,
            eval_seed=1041,
            checkpoint_loaded=True,
            checkpoint_path="checkpoint.pt",
            output=output,
            primitive_names=("spectral", "separable"),
        )

        self.assertAlmostEqual(row["primitive_load"]["spectral"], 0.6)
        self.assertAlmostEqual(row["primitive_entropy"], 0.61, places=6)
        self.assertAlmostEqual(row["memory_norm"], 2.5)
        self.assertEqual(row["adapter_norms"], {"spectral": 1.2, "separable": 0.3})


if __name__ == "__main__":
    unittest.main()
