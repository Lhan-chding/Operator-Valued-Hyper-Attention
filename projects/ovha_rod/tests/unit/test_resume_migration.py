import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ovha_rod.runtime_contracts import (
    validate_private_epoch_checkpoint,
    write_checkpoint_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resume_guard = _load_script("resume_guard")


class ResumeMigrationTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.migration = _load_script("migrate_epoch_resume")
        self.source_commit = "a" * 40
        self.target_commit = "b" * 40
        self.identity = {
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
            "data_root": "/private/data/COCO2014",
            "bert_root": "/private/models/bert-base-uncased",
            "initial_checkpoint_sha256": "c" * 64,
            "project_commit": self.source_commit,
            "mmdetection_commit": "d" * 40,
            "environment_profile": "cu121",
        }

    def _source_run(self, parent: Path) -> Path:
        source = parent / "source"
        source.mkdir(mode=0o700)
        identity_path = resume_guard.initialize_identity(source, self.identity)
        checkpoint = source / "epoch_1.pth"
        checkpoint.write_bytes(b"complete epoch one checkpoint")
        checkpoint.chmod(0o600)
        write_checkpoint_provenance(checkpoint, identity_path)
        pointer = source / "last_checkpoint"
        pointer.write_text("epoch_1.pth\n", encoding="utf-8")
        pointer.chmod(0o600)
        return source

    @staticmethod
    def _fingerprint(path: Path) -> dict[str, tuple[int, int, str]]:
        return {
            item.name: (
                item.stat().st_ino,
                item.stat().st_mtime_ns,
                hashlib.sha256(item.read_bytes()).hexdigest(),
            )
            for item in path.iterdir()
            if item.is_file()
        }

    def test_migration_preserves_source_and_builds_guarded_target(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            source = self._source_run(parent)
            before = self._fingerprint(source)
            target = parent / "target"

            result = self.migration.migrate_epoch_resume(
                source,
                target,
                expected_source_project_commit=self.source_commit,
                target_project_commit=self.target_commit,
            )

            self.assertEqual(result, target.resolve())
            self.assertEqual(before, self._fingerprint(source))
            checkpoint = validate_private_epoch_checkpoint(target)
            self.assertEqual(checkpoint.name, "epoch_1.pth")
            self.assertEqual(checkpoint.read_bytes(), b"complete epoch one checkpoint")
            self.assertEqual(checkpoint.stat().st_mode & 0o777, 0o400)
            self.assertEqual(checkpoint.stat().st_nlink, 1)

            target_identity = json.loads(
                (target / "run_identity.json").read_text(encoding="utf-8"))
            expected_identity = dict(self.identity)
            expected_identity["project_commit"] = self.target_commit
            self.assertEqual(target_identity, {
                "contract": resume_guard.CONTRACT,
                "identity": expected_identity,
            })
            resume_guard.validate_identity(target, expected_identity)
            digest, size = resume_guard._validated_provenance(
                checkpoint, target / "run_identity.json")
            self.assertEqual(digest, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(size, checkpoint.stat().st_size)

            receipt_path = target / "resume_migration.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["contract"], "ovha_rod_resume_migration_v1")
            self.assertEqual(receipt["source_project_commit"], self.source_commit)
            self.assertEqual(receipt["target_project_commit"], self.target_commit)
            self.assertEqual(receipt["checkpoint_sha256"], digest)
            self.assertEqual(receipt["checkpoint_size"], size)
            self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o400)
            self.assertEqual(
                (target / "last_checkpoint").read_text(encoding="utf-8"),
                "epoch_1.pth\n",
            )

    def test_migration_rejects_unexpected_identity_or_existing_target(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            source = self._source_run(parent)

            with self.assertRaisesRegex(ValueError, "source project commit"):
                self.migration.migrate_epoch_resume(
                    source,
                    parent / "wrong-source",
                    expected_source_project_commit="e" * 40,
                    target_project_commit=self.target_commit,
                )
            with self.assertRaisesRegex(ValueError, "must differ"):
                self.migration.migrate_epoch_resume(
                    source,
                    parent / "same-commit",
                    expected_source_project_commit=self.source_commit,
                    target_project_commit=self.source_commit,
                )

            existing = parent / "existing"
            existing.mkdir(mode=0o700)
            (existing / "keep.txt").write_text("do not overwrite")
            with self.assertRaisesRegex(ValueError, "must not exist"):
                self.migration.migrate_epoch_resume(
                    source,
                    existing,
                    expected_source_project_commit=self.source_commit,
                    target_project_commit=self.target_commit,
                )
            self.assertEqual((existing / "keep.txt").read_text(), "do not overwrite")

    def test_migration_rejects_tampered_source_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            source = self._source_run(parent)
            (source / "epoch_1.pth").write_bytes(b"tampered after provenance")

            with self.assertRaisesRegex(ValueError, "digest"):
                self.migration.migrate_epoch_resume(
                    source,
                    parent / "target",
                    expected_source_project_commit=self.source_commit,
                    target_project_commit=self.target_commit,
                )
            self.assertFalse((parent / "target").exists())

    def test_runtime_checkout_must_be_clean_and_exact(self):
        with mock.patch.object(
                self.migration, "_git_output",
                side_effect=(self.target_commit, "")):
            self.migration.validate_runtime_checkout(ROOT, self.target_commit)

        with mock.patch.object(
                self.migration, "_git_output",
                side_effect=(self.source_commit, "")):
            with self.assertRaisesRegex(ValueError, "target commit"):
                self.migration.validate_runtime_checkout(ROOT, self.target_commit)

        with mock.patch.object(
                self.migration, "_git_output",
                side_effect=(self.target_commit, " M changed.py")):
            with self.assertRaisesRegex(ValueError, "clean"):
                self.migration.validate_runtime_checkout(ROOT, self.target_commit)


if __name__ == "__main__":
    unittest.main()
