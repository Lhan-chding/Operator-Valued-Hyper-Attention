import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moat_ovha_torch.config import Phase15Config


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class Phase16CheckpointLoaderContractTests(unittest.TestCase):
    def test_checkpoint_loader_exposes_map_location_parameter(self):
        from moat_ovha_torch.train.checkpoints import load_checkpoint_for_eval

        signature = inspect.signature(load_checkpoint_for_eval)

        self.assertIn("map_location", signature.parameters)


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed; checkpoint integrity tests skipped.")
class Phase16CheckpointEvaluationTests(unittest.TestCase):
    def _config(self, output_dir: Path) -> Phase15Config:
        return Phase15Config.from_mapping(
            {
                "seed": 101,
                "output_dir": str(output_dir),
                "device": "cpu",
                "steps": 1,
                "batch_size": 1,
                "support_points": 8,
                "query_points": 6,
                "num_demos": 1,
                "context_points": 4,
                "d_model": 16,
                "memory_tokens": 2,
                "lr": 0.001,
                "mode": "operator_transfer",
                "train_split": "iid",
                "eval_splits": ["iid"],
                "families": ["spectral_family"],
                "train_models": ["ovha_full"],
                "eval_models": ["ovha_full"],
                "require_checkpoint": True,
                "allow_metadata_inputs": False,
            }
        )

    def test_evaluator_requires_checkpoint(self):
        from moat_ovha_torch.eval.evaluator import run_evaluation

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))

            with self.assertRaises(FileNotFoundError):
                run_evaluation(config)

    def test_strict_checkpoint_load(self):
        from moat_ovha_torch.models.baselines import build_model
        from moat_ovha_torch.train.checkpoints import load_checkpoint_for_eval
        from moat_ovha_torch.train.trainer import run_training_for_model

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            checkpoint = run_training_for_model(config, "ovha_full")
            model = build_model("ovha_full", d_model=config.d_model, memory_tokens=config.memory_tokens)

            payload = load_checkpoint_for_eval(model, checkpoint, strict=True)

            self.assertEqual(payload["model_name"], "ovha_full")
            self.assertEqual(payload["seed"], config.seed)
            self.assertEqual(payload["train_steps"], config.steps)
            self.assertIn("config_hash", payload)

    def test_checkpoint_loader_accepts_explicit_map_location(self):
        import torch

        from moat_ovha_torch.models.baselines import build_model
        from moat_ovha_torch.train.checkpoints import load_checkpoint_for_eval
        from moat_ovha_torch.train.trainer import run_training_for_model

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            checkpoint = run_training_for_model(config, "ovha_full")
            model = build_model("ovha_full", d_model=config.d_model, memory_tokens=config.memory_tokens)
            calls = []
            original_load = torch.load

            def spy_load(*args, **kwargs):
                calls.append(kwargs.get("map_location"))
                return original_load(*args, **kwargs)

            with patch("torch.load", side_effect=spy_load):
                load_checkpoint_for_eval(model, checkpoint, strict=True, map_location="cpu")

        self.assertEqual(calls, ["cpu"])

    def test_changing_checkpoint_weights_changes_eval_output(self):
        import torch

        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
        from moat_ovha_torch.models.baselines import build_model
        from moat_ovha_torch.train.checkpoints import load_checkpoint_for_eval
        from moat_ovha_torch.train.trainer import run_training_for_model

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            checkpoint = run_training_for_model(config, "ovha_full")
            zoo = MetadataFreeOperatorZoo(seed=config.seed + 1000)
            batch, _ = zoo.sample_batch(
                batch_size=config.batch_size,
                num_demos=config.num_demos,
                context_points=config.context_points,
                support_points=config.support_points,
                query_points=config.query_points,
                family="spectral_family",
                split="iid",
                mode=config.mode,
                device="cpu",
            )

            model = build_model("ovha_full", d_model=config.d_model, memory_tokens=config.memory_tokens)
            load_checkpoint_for_eval(model, checkpoint, strict=True)
            with torch.no_grad():
                original = model(batch).y_hat.detach().clone()

            payload = torch.load(checkpoint, map_location="cpu")
            payload["model_state_dict"] = {
                key: value + 0.5 if torch.is_floating_point(value) else value
                for key, value in payload["model_state_dict"].items()
            }
            altered_checkpoint = checkpoint.parent / "model_altered.pt"
            torch.save(payload, altered_checkpoint)

            altered = build_model("ovha_full", d_model=config.d_model, memory_tokens=config.memory_tokens)
            load_checkpoint_for_eval(altered, altered_checkpoint, strict=True)
            with torch.no_grad():
                changed = altered(batch).y_hat.detach().clone()

            self.assertFalse(torch.allclose(original, changed))

    def test_eval_metrics_records_checkpoint_loaded(self):
        from moat_ovha_torch.eval.evaluator import run_evaluation
        from moat_ovha_torch.train.trainer import run_training_for_model

        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            checkpoint = run_training_for_model(config, "ovha_full")

            metrics_path = run_evaluation(config)

            rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            self.assertEqual(rows[0]["checkpoint_loaded"], True)
            self.assertEqual(rows[0]["checkpoint_path"], str(checkpoint))
            self.assertEqual(rows[0]["checkpoint_train_steps"], config.steps)
            self.assertEqual(rows[0]["model_name"], "ovha_full")
            self.assertEqual(rows[0]["seed"], config.seed)
            self.assertEqual(rows[0]["eval_seed"], config.seed + 1000)

    def test_simple_stack_loss_keeps_gradient_path(self):
        import torch

        from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
        from moat_ovha_torch.models.baselines import build_model
        from moat_ovha_torch.train.losses import prediction_loss

        zoo = MetadataFreeOperatorZoo(seed=13)
        batch, _ = zoo.sample_batch(
            batch_size=1,
            num_demos=1,
            context_points=4,
            support_points=8,
            query_points=6,
            family="query_piecewise_composition_family",
            split="iid",
            mode="operator_transfer",
            device="cpu",
        )
        model = build_model("simple_stack", d_model=16, memory_tokens=2)

        output = model(batch)
        loss = prediction_loss(output.y_hat, batch.target_y)

        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertTrue(any(param.grad is not None for param in model.parameters() if param.requires_grad))


if __name__ == "__main__":
    unittest.main()
