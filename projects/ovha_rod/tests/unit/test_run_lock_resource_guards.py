import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load_script():
    path = SCRIPTS / "run_lock.py"
    spec = importlib.util.spec_from_file_location("phase2_run_lock", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunLockResourceGuardTests(unittest.TestCase):
    def test_private_run_lock_and_command_log_contracts(self):
        run_lock = _load_script()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            work_dir = Path(directory)
            work_dir.chmod(0o700)
            descriptor = run_lock.open_run_lock(work_dir)
            try:
                lock = work_dir / ".run.lock"
                self.assertTrue(lock.is_file())
                self.assertEqual(lock.stat().st_mode & 0o777, 0o600)

                environment = {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": "5",
                    "MASTER_ADDR": "127.0.0.1",
                    "PORT": "29631",
                    "TRANSFORMERS_OFFLINE": "1",
                }
                with mock.patch.dict(os.environ, environment, clear=False):
                    run_lock._write_command_log(
                        work_dir, ["python", "train.py"], "fresh")
                    run_lock._write_command_log(
                        work_dir, ["python", "train.py", "--resume"], "resume")
                command_log = (work_dir / "command.txt").read_text(
                    encoding="utf-8")
                self.assertIn("CUDA_VISIBLE_DEVICES=5", command_log)
                self.assertIn("# guarded epoch-boundary resume", command_log)
                self.assertIn("--resume", command_log)
                self.assertEqual(
                    (work_dir / "command.txt").stat().st_mode & 0o777,
                    0o600,
                )
            finally:
                os.close(descriptor)

    def test_resource_guards_run_under_lock_before_fresh_identity(self):
        run_lock = _load_script()
        order = []
        descriptor = os.open(os.devnull, os.O_RDONLY)
        arguments = [
            "run_lock.py",
            "--work-dir", "/private/run",
            "--mode", "fresh",
            "--freeze-dir", "/private/frozen",
            "--cwd", "/private/mmdetection",
            "--guard-gpus", "1",
            "--guard-port", "29631",
            "--expected-identity", "dataset=refcoco",
            "--", "/bin/true",
        ]
        with (
            mock.patch.object(sys, "argv", arguments),
            mock.patch.object(
                run_lock, "_open_run_lock_with_state",
                return_value=(descriptor, None)),
            mock.patch.object(
                run_lock, "require_visible_device_ids", return_value=(5,)),
            mock.patch.object(
                run_lock, "require_selected_gpus_idle",
                side_effect=lambda _: order.append("gpu")),
            mock.patch.object(
                run_lock, "require_port_available",
                side_effect=lambda _: order.append("port")),
            mock.patch.object(
                run_lock, "initialize_identity",
                side_effect=lambda *_: order.append("identity")),
            mock.patch.object(run_lock, "_write_command_log"),
            mock.patch.object(
                run_lock.subprocess, "run",
                return_value=SimpleNamespace(returncode=0)),
        ):
            self.assertEqual(run_lock.main(), 0)

        self.assertEqual(order, ["gpu", "port", "identity"])

    def test_guard_failure_removes_only_a_new_run_lock(self):
        run_lock = _load_script()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            work_dir = Path(directory)
            work_dir.chmod(0o700)
            arguments = [
                "run_lock.py",
                "--work-dir", str(work_dir),
                "--mode", "fresh",
                "--freeze-dir", str(work_dir / "frozen"),
                "--cwd", str(work_dir),
                "--guard-gpus", "1",
                "--expected-identity", "dataset=refcoco",
                "--", "/bin/true",
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(
                    run_lock, "require_visible_device_ids",
                    return_value=(5,)),
                mock.patch.object(
                    run_lock, "require_selected_gpus_idle",
                    side_effect=(RuntimeError("GPU is busy"), None)),
                mock.patch.object(run_lock, "initialize_identity") as identity,
                mock.patch.object(run_lock, "_write_command_log"),
                mock.patch.object(
                    run_lock.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0)),
            ):
                with self.assertRaisesRegex(RuntimeError, "GPU is busy"):
                    run_lock.main()

                identity.assert_not_called()
                self.assertFalse((work_dir / ".run.lock").exists())

                self.assertEqual(run_lock.main(), 0)
                identity.assert_called_once()
                self.assertTrue((work_dir / ".run.lock").is_file())

    def test_guard_failure_preserves_a_preexisting_run_lock(self):
        run_lock = _load_script()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            work_dir = Path(directory)
            work_dir.chmod(0o700)
            lock_path = work_dir / ".run.lock"
            lock_path.write_bytes(b"preexisting\n")
            lock_path.chmod(0o600)
            arguments = [
                "run_lock.py",
                "--work-dir", str(work_dir),
                "--mode", "fresh",
                "--freeze-dir", str(work_dir / "frozen"),
                "--cwd", str(work_dir),
                "--guard-port", "29631",
                "--expected-identity", "dataset=refcoco",
                "--", "/bin/true",
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(
                    run_lock, "require_port_available",
                    side_effect=RuntimeError("port is busy")),
                mock.patch.object(run_lock, "initialize_identity") as identity,
            ):
                with self.assertRaisesRegex(RuntimeError, "port is busy"):
                    run_lock.main()

            identity.assert_not_called()
            self.assertEqual(lock_path.read_bytes(), b"preexisting\n")

    def test_failed_flock_does_not_unlink_the_shared_lock_path(self):
        run_lock = _load_script()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            work_dir = Path(directory)
            work_dir.chmod(0o700)
            with mock.patch.object(
                run_lock.fcntl,
                "flock",
                side_effect=BlockingIOError("lock already held"),
            ):
                with self.assertRaisesRegex(
                    BlockingIOError, "lock already held"
                ):
                    run_lock._open_run_lock_with_state(work_dir)

            self.assertTrue((work_dir / ".run.lock").is_file())


if __name__ == "__main__":
    unittest.main()
