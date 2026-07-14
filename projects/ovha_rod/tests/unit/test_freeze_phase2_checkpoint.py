import fcntl
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from ovha_rod.runtime_contracts import write_checkpoint_provenance


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SOURCE_COMMIT = "d93d2db25c6156b4ca5ebc352a3c89a4c6d06c96"
MMDET_COMMIT = "cfd5d3a985b0249de009b67d04f37263e11cdf3d"
INITIAL_SHA256 = "b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a"
DATA_ROOT = "/private/data/COCO2014"
BERT_ROOT = "/private/models/bert-base-uncased"


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resume_guard = _load_script("resume_guard")
freezer = _load_script("freeze_phase2_checkpoint")


class FreezePhase2CheckpointTests(unittest.TestCase):
    def _phase1_run(self, parent: Path) -> tuple[Path, Path]:
        work_dir = parent / "phase1"
        work_dir.mkdir(mode=0o700)
        lock = work_dir / ".run.lock"
        lock.touch(mode=0o600)
        lock.chmod(0o600)
        identity = resume_guard.initialize_identity(
            work_dir,
            {
                "dataset": "refcoco",
                "variant": "rqgo",
                "seed": "2026",
                "gpus": "1",
                "per_device_batch": "8",
                "accumulative_counts": "4",
                "global_batch": "32",
                "amp": "false",
                "amp_dtype": "none",
                "physical_cuda_devices": "4",
                "data_root": DATA_ROOT,
                "bert_root": BERT_ROOT,
                "initial_checkpoint_sha256": INITIAL_SHA256,
                "project_commit": SOURCE_COMMIT,
                "mmdetection_commit": MMDET_COMMIT,
                "environment_profile": "cu121-wheel",
            },
        )
        checkpoint = work_dir / "epoch_2.pth"
        checkpoint.write_bytes(b"complete epoch two checkpoint")
        checkpoint.chmod(0o600)
        write_checkpoint_provenance(checkpoint, identity)
        return work_dir, checkpoint

    def _freeze(self, checkpoint: Path, freeze_dir: Path):
        return freezer.freeze_phase2_checkpoint(
            checkpoint,
            freeze_dir,
            expected_source_project_commit=SOURCE_COMMIT,
            expected_seed=2026,
            expected_data_root=DATA_ROOT,
            expected_bert_root=BERT_ROOT,
        )

    def test_freezes_inactive_epoch2_as_private_digest_bound_copy(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            _, checkpoint = self._phase1_run(parent)
            source_bytes = checkpoint.read_bytes()
            source_digest = hashlib.sha256(source_bytes).hexdigest()

            frozen, digest, identity_digest, provenance_digest = self._freeze(
                checkpoint, parent / "frozen")

            self.assertEqual(digest, source_digest)
            self.assertEqual(frozen.read_bytes(), source_bytes)
            self.assertEqual(frozen.stat().st_mode & 0o777, 0o400)
            self.assertEqual(frozen.stat().st_nlink, 1)
            manifest = json.loads(
                frozen.with_name(frozen.name + ".sha256.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(manifest["sha256"], source_digest)
            self.assertEqual(manifest["source"], str(checkpoint.resolve()))
            receipt_path = frozen.with_name(frozen.name + ".phase2-source.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["source_identity_sha256"], identity_digest)
            self.assertEqual(receipt["source_provenance_sha256"], provenance_digest)
            self.assertEqual(receipt["source_project_commit"], SOURCE_COMMIT)
            self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o400)

            repeated = self._freeze(checkpoint, parent / "frozen")
            self.assertEqual(repeated, (
                frozen, digest, identity_digest, provenance_digest))
            self.assertEqual(len(tuple((parent / "frozen").glob("*.pth"))), 1)

    def test_rejects_any_checkpoint_other_than_exact_epoch2(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            _, checkpoint = self._phase1_run(parent)
            wrong = checkpoint.with_name("epoch_1.pth")
            wrong.write_bytes(checkpoint.read_bytes())
            wrong.chmod(0o600)

            with self.assertRaisesRegex(ValueError, "exact epoch_2"):
                self._freeze(wrong, parent / "frozen")

    def test_rejects_an_active_phase1_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            work_dir, checkpoint = self._phase1_run(parent)

            with (work_dir / ".run.lock").open("rb") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "still active"):
                    self._freeze(checkpoint, parent / "frozen")

    def test_rejects_checkpoint_tampering_after_provenance(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            _, checkpoint = self._phase1_run(parent)
            checkpoint.write_bytes(b"x" * checkpoint.stat().st_size)
            checkpoint.chmod(0o600)

            with self.assertRaisesRegex(ValueError, "digest does not match"):
                self._freeze(checkpoint, parent / "frozen")

    def test_rejects_missing_or_nonprivate_run_lock(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            work_dir, checkpoint = self._phase1_run(parent)
            lock = work_dir / ".run.lock"
            lock.unlink()
            with self.assertRaisesRegex(ValueError, "lock is missing"):
                self._freeze(checkpoint, parent / "missing-lock")

            lock.touch(mode=0o600)
            lock.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "lock must be private"):
                self._freeze(checkpoint, parent / "public-lock")

    def test_rejects_unsafe_checkpoint_and_missing_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            work_dir, checkpoint = self._phase1_run(parent)

            checkpoint.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "must be private"):
                self._freeze(checkpoint, parent / "public-checkpoint")

            checkpoint.chmod(0o600)
            (work_dir / "run_identity.json").unlink()
            with self.assertRaisesRegex(ValueError, "identity is missing"):
                self._freeze(checkpoint, parent / "missing-identity")

    def test_rejects_symlinked_epoch2(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            work_dir = parent / "phase1"
            work_dir.mkdir(mode=0o700)
            target = work_dir / "payload.pth"
            target.write_bytes(b"payload")
            target.chmod(0o600)
            checkpoint = work_dir / "epoch_2.pth"
            checkpoint.symlink_to(target)

            with self.assertRaisesRegex(ValueError, "non-symlink"):
                self._freeze(checkpoint, parent / "frozen")

    def test_main_prints_exact_frozen_path_and_digest(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            _, checkpoint = self._phase1_run(parent)
            output = io.StringIO()
            arguments = [
                "freeze_phase2_checkpoint.py",
                "--checkpoint", str(checkpoint),
                "--freeze-dir", str(parent / "frozen"),
                "--expected-source-project-commit", SOURCE_COMMIT,
                "--expected-seed", "2026",
                "--expected-data-root", DATA_ROOT,
                "--expected-bert-root", BERT_ROOT,
            ]

            with mock.patch.object(sys, "argv", arguments), redirect_stdout(output):
                self.assertEqual(freezer.main(), 0)

            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertTrue(Path(lines[0]).is_file())
            self.assertRegex(lines[1], r"^[0-9a-f]{64}$")
            self.assertRegex(lines[2], r"^[0-9a-f]{64}$")
            self.assertRegex(lines[3], r"^[0-9a-f]{64}$")

    def test_rejects_a_self_signed_but_wrong_source_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            _, checkpoint = self._phase1_run(parent)

            with self.assertRaisesRegex(ValueError, "source project commit"):
                freezer.freeze_phase2_checkpoint(
                    checkpoint,
                    parent / "frozen",
                    expected_source_project_commit="e" * 40,
                    expected_seed=2026,
                    expected_data_root=DATA_ROOT,
                    expected_bert_root=BERT_ROOT,
                )


if __name__ == "__main__":
    unittest.main()
