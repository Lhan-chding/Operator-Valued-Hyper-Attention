import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_phase1_server.sh"
CHECKPOINT_SHA256 = (
    "b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a"
)


class Phase1Epoch2HandoffTests(unittest.TestCase):
    def _dry_run(self, *extra: str, dataset: str = "refcoco",
                 variant: str = "rqgo") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                "bash",
                str(RUNNER),
                "--dataset", dataset,
                "--variant", variant,
                "--mmdet-root", "/reviewed/mmdetection",
                "--data-root", "/private/coco",
                "--checkpoint", "/private/official.pth",
                "--bert-root", "/private/bert",
                "--work-root", "/private/runs",
                "--gpus", "1",
                "--per-device-batch", "8",
                "--seed", "2026",
                "--python", sys.executable,
                "--checkpoint-sha256", CHECKPOINT_SHA256,
                "--master-port", "29627",
                *extra,
                "--dry-run",
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )

    def test_resume_can_stop_naturally_after_exact_epoch2(self):
        result = self._dry_run("--resume", "--stop-after-epoch2")

        self.assertEqual(result.returncode, 0, result.stderr)
        command = next(
            line for line in result.stdout.splitlines()
            if line.startswith("Command:"))
        self.assertIn("train_cfg.max_epochs=2", command)

    def test_default_protocol_keeps_the_five_epoch_budget(self):
        result = self._dry_run("--resume")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("train_cfg.max_epochs=2", result.stdout)

    def test_epoch2_handoff_requires_resume(self):
        result = self._dry_run("--stop-after-epoch2")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --resume", result.stderr)

    def test_epoch2_handoff_is_limited_to_refcoco_rqgo(self):
        result = self._dry_run(
            "--resume", "--stop-after-epoch2", variant="generic")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only supports refcoco/rqgo", result.stderr)

    def test_resource_guards_are_rechecked_under_the_run_lock(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn('--guard-gpus "${GPUS}"', source)
        self.assertIn('--guard-port "${MASTER_PORT}"', source)


if __name__ == "__main__":
    unittest.main()
