import ast
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "phase2"
    / "ovha_rod_swin_t_3e_refcoco_full.py"
)
RUNNER = ROOT / "scripts" / "run_phase2_full_server.sh"
FREEZER = ROOT / "scripts" / "freeze_phase2_checkpoint.py"


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )
    return _config_literal(assignment.value)


def _config_literal(node):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and not node.args
    ):
        return {
            keyword.arg: _config_literal(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
    if isinstance(node, ast.Dict):
        return {
            _config_literal(key): _config_literal(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, (ast.Tuple, ast.List)):
        values = tuple(_config_literal(value) for value in node.elts)
        return values if isinstance(node, ast.Tuple) else list(values)
    return ast.literal_eval(node)


class Phase2FullSurfaceTests(unittest.TestCase):
    def _dry_run(self, checkpoint="/private/epoch_2.pth"):
        return subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--mmdet-root",
                "/reviewed/mmdetection",
                "--data-root",
                "/private/coco",
                "--epoch2-checkpoint",
                checkpoint,
                "--bert-root",
                "/private/bert",
                "--work-root",
                "/private/runs",
                "--python",
                sys.executable,
                "--gpus",
                "1",
                "--per-device-batch",
                "8",
                "--master-port",
                "29631",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )

    def test_phase2_files_exist_and_parse(self):
        for path in (CONFIG, RUNNER, FREEZER):
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
        ast.parse(CONFIG.read_text(encoding="utf-8"))
        ast.parse(FREEZER.read_text(encoding="utf-8"))
        subprocess.run(
            ["bash", "-n", str(RUNNER)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_config_enables_the_complete_bounded_bank(self):
        self.assertEqual(
            _literal_assignment(CONFIG, "_base_"),
            "../ovha_rod_swin_t_5e_refcoco.py",
        )
        model = _literal_assignment(CONFIG, "model")
        self.assertIs(model["train_decoder_operator_only"], True)
        bank = model["decoder_operator_cfg"]
        self.assertEqual(
            bank,
            {
                "enabled": True,
                "enabled_operators": ("qsro", "tq_cato", "ms_tleo"),
                "router_hidden_dim": 128,
                "adapter_rank": 16,
                "qsro_query_chunk_size": 128,
                "use_router": True,
                "use_memory": True,
                "use_hyper_adapter": True,
                "use_rceo": True,
            },
        )

    def test_config_is_a_fresh_three_epoch_nondeterministic_phase(self):
        self.assertIsNone(_literal_assignment(CONFIG, "load_from"))
        self.assertEqual(_literal_assignment(CONFIG, "max_epochs"), 3)
        train_cfg = _literal_assignment(CONFIG, "train_cfg")
        self.assertEqual(train_cfg["max_epochs"], 3)
        self.assertEqual(train_cfg["val_interval"], 1)
        self.assertEqual(
            _literal_assignment(CONFIG, "randomness"),
            {"seed": 2026, "deterministic": False},
        )
        hooks = _literal_assignment(CONFIG, "custom_hooks")
        self.assertEqual(
            tuple(hook["type"] for hook in hooks),
            ("OperatorDiagnosticsHook", "CheckpointProvenanceHook"),
        )
        optim = _literal_assignment(CONFIG, "optim_wrapper")
        self.assertEqual(
            optim["paramwise_cfg"]["custom_keys"]["decoder_operator"],
            {"lr_mult": 1.0},
        )

    def test_config_and_runner_do_not_use_training_oracles(self):
        forbidden = (
            "ground_truth_box",
            "gt_box",
            "oracle_candidate",
            "proposal_cache",
            "region_features.npy",
        )
        source = (
            CONFIG.read_text(encoding="utf-8")
            + RUNNER.read_text(encoding="utf-8")
        ).lower()
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_dry_run_loads_epoch2_without_resuming_optimizer_state(self):
        result = self._dry_run()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(CONFIG.name, result.stdout)
        self.assertIn("load_from=/private/epoch_2.pth", result.stdout)
        self.assertIn("model.train_decoder_operator_only=True", result.stdout)
        self.assertIn("randomness.deterministic=False", result.stdout)
        self.assertIn("optim_wrapper.accumulative_counts=4", result.stdout)
        command_line = next(
            line for line in result.stdout.splitlines()
            if line.startswith("Command:")
        )
        self.assertNotIn("--resume", command_line)

    def test_runner_rejects_any_checkpoint_other_than_epoch2(self):
        result = self._dry_run("/private/epoch_1.pth")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("epoch_2.pth", result.stderr)


if __name__ == "__main__":
    unittest.main()
