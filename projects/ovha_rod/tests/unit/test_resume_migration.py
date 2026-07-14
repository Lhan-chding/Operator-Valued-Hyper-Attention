import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ovha_rod.runtime_contracts import (
    validate_private_epoch_checkpoint,
    write_checkpoint_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parents[1]
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
        self.real_validate_runtime_checkout = (
            self.migration.validate_runtime_checkout)
        runtime_patcher = mock.patch.object(
            self.migration, "validate_runtime_checkout", return_value=ROOT)
        self.runtime_validator = runtime_patcher.start()
        self.addCleanup(runtime_patcher.stop)
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
        lock = source / ".run.lock"
        lock.touch(mode=0o600)
        lock.chmod(0o600)
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
                runtime_project_dir=ROOT,
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
            self.assertEqual(receipt["runtime_project_dir"], str(ROOT))
            self.assertEqual(receipt["checkpoint_sha256"], digest)
            self.assertEqual(receipt["checkpoint_size"], size)
            self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o400)
            self.assertEqual(
                (target / "last_checkpoint").read_text(encoding="utf-8"),
                "epoch_1.pth\n",
            )

            command = [
                sys.executable,
                str(SCRIPTS / "run_lock.py"),
                "--work-dir", str(target),
                "--mode", "resume",
                "--freeze-dir", str(parent / "frozen"),
                "--cwd", str(ROOT),
            ]
            for key, value in expected_identity.items():
                command.extend(("--expected-identity", f"{key}={value}"))
            command.extend((
                "--",
                sys.executable,
                "-c",
                "import pathlib,sys; assert sys.argv[1] == '--resume'; "
                "assert pathlib.Path(sys.argv[2]).is_file()",
            ))
            environment = dict(__import__("os").environ)
            environment["PYTHONPATH"] = str(ROOT)
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True,
                env=environment)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("# guarded epoch-boundary resume", (
                target / "command.txt").read_text(encoding="utf-8"))

    def test_migration_rejects_unexpected_identity_or_existing_target(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            source = self._source_run(parent)

            with self.assertRaisesRegex(ValueError, "source project commit"):
                self.migration.migrate_epoch_resume(
                    source,
                    parent / "wrong-source",
                    runtime_project_dir=ROOT,
                    expected_source_project_commit="e" * 40,
                    target_project_commit=self.target_commit,
                )
            with self.assertRaisesRegex(ValueError, "must differ"):
                self.migration.migrate_epoch_resume(
                    source,
                    parent / "same-commit",
                    runtime_project_dir=ROOT,
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
                    runtime_project_dir=ROOT,
                    expected_source_project_commit=self.source_commit,
                    target_project_commit=self.target_commit,
                )
            self.assertEqual((existing / "keep.txt").read_text(), "do not overwrite")

    def test_migration_rejects_an_active_source_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            source = self._source_run(parent)
            with (source / ".run.lock").open("rb") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "still active"):
                    self.migration.migrate_epoch_resume(
                        source,
                        parent / "target",
                        runtime_project_dir=ROOT,
                        expected_source_project_commit=self.source_commit,
                        target_project_commit=self.target_commit,
                    )
            self.assertFalse((parent / "target").exists())

    def test_migration_rejects_tampered_source_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            source = self._source_run(parent)
            (source / "epoch_1.pth").write_bytes(b"tampered after provenance")
            (source / "last_checkpoint").touch()

            with self.assertRaisesRegex(ValueError, "provenance|digest"):
                self.migration.migrate_epoch_resume(
                    source,
                    parent / "target",
                    runtime_project_dir=ROOT,
                    expected_source_project_commit=self.source_commit,
                    target_project_commit=self.target_commit,
                )
            self.assertFalse((parent / "target").exists())

    def test_runtime_checkout_must_be_clean_and_exact(self):
        with (
            mock.patch.object(
                self.migration, "_validate_detached_runtime_metadata") as metadata,
            mock.patch.object(self.migration, "_attest_runtime_tree") as attest,
        ):
            self.real_validate_runtime_checkout(ROOT, self.target_commit)
        attest.assert_called_once_with(ROOT, REPO_ROOT, self.target_commit)
        self.assertEqual(metadata.call_count, 2)

        with mock.patch.object(
                self.migration, "_validate_detached_runtime_metadata",
                side_effect=ValueError("not detached at the target commit")):
            with self.assertRaisesRegex(ValueError, "target commit"):
                self.real_validate_runtime_checkout(ROOT, self.target_commit)

        with (
            mock.patch.object(
                self.migration, "_validate_detached_runtime_metadata",
                side_effect=(None, ValueError("changed")),
            ),
            mock.patch.object(self.migration, "_attest_runtime_tree"),
        ):
            with self.assertRaisesRegex(ValueError, "changed during validation"):
                self.real_validate_runtime_checkout(ROOT, self.target_commit)

        self.assertEqual(
            self.migration._git_output(ROOT, "rev-parse", "HEAD"),
            __import__("subprocess").check_output(
                ("git", "-C", str(ROOT), "rev-parse", "HEAD"),
                text=True).strip(),
        )

    def test_invalid_identity_and_commit_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "full lowercase Git commit"):
            self.migration._require_commit("d3bad6a", "target project commit")

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            source = self._source_run(parent)
            identity_path = source / "run_identity.json"
            malformed = json.loads(identity_path.read_text(encoding="utf-8"))
            malformed["identity"]["unexpected"] = "value"
            identity_path.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
            identity_path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "incomplete or unexpected"):
                self.migration.migrate_epoch_resume(
                    source,
                    parent / "target",
                    runtime_project_dir=ROOT,
                    expected_source_project_commit=self.source_commit,
                    target_project_commit=self.target_commit,
                )

    def test_runtime_tree_attestation_hashes_raw_read_only_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            top = Path(directory) / "runtime"
            root = top / "projects" / "ovha_rod"
            root.mkdir(parents=True)
            git_dir = top / ".git"
            info_dir = git_dir / "objects" / "info"
            info_dir.mkdir(parents=True)
            head = git_dir / "HEAD"
            head.write_text(f"{self.target_commit}\n")
            tracked = root / "script.py"
            tracked.write_bytes(b"print('reviewed')\n")
            digest = hashlib.sha1(
                b"blob 18\0" + tracked.read_bytes()).hexdigest()
            for path in (tracked, head):
                path.chmod(0o444)
            for path in (info_dir, info_dir.parent, git_dir,
                         root, root.parent, top):
                path.chmod(0o555)
            tree = (
                f"100644 blob {digest}\tprojects/ovha_rod/script.py\0")

            with mock.patch.object(
                    self.migration, "_git_output", return_value=tree):
                self.migration._attest_runtime_tree(
                    root, top, self.target_commit)
            self.migration._validate_detached_runtime_metadata(
                top, self.target_commit)

            tracked.chmod(0o644)
            with mock.patch.object(
                    self.migration, "_git_output", return_value=tree):
                with self.assertRaisesRegex(ValueError, "file is unsafe"):
                    self.migration._attest_runtime_tree(
                        root, top, self.target_commit)
            tracked.chmod(0o444)

            unsupported = (
                f"160000 commit {self.target_commit}"
                "\tprojects/ovha_rod/submodule\0")
            with mock.patch.object(
                    self.migration, "_git_output", return_value=unsupported):
                with self.assertRaisesRegex(ValueError, "unsupported entry"):
                    self.migration._attest_runtime_tree(
                        root, top, self.target_commit)

            wrong_digest = (
                f"100644 blob {'0' * 40}"
                "\tprojects/ovha_rod/script.py\0")
            with mock.patch.object(
                    self.migration, "_git_output", return_value=wrong_digest):
                with self.assertRaisesRegex(ValueError, "does not match commit"):
                    self.migration._attest_runtime_tree(
                        root, top, self.target_commit)

            root.chmod(0o755)
            extra = root / "untracked.py"
            extra.write_text("unreviewed\n")
            extra.chmod(0o444)
            root.chmod(0o555)
            with mock.patch.object(
                    self.migration, "_git_output", return_value=tree):
                with self.assertRaisesRegex(ValueError, "missing or extra"):
                    self.migration._attest_runtime_tree(
                        root, top, self.target_commit)

    def test_source_metadata_change_during_copy_removes_incomplete_target(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            source = self._source_run(parent)
            target = parent / "target"
            original_copy = self.migration._copy_checkpoint

            def changing_copy(*args, **kwargs):
                result = original_copy(*args, **kwargs)
                identity_path = source / "run_identity.json"
                identity_path.write_bytes(identity_path.read_bytes() + b" ")
                return result

            with mock.patch.object(
                    self.migration, "_copy_checkpoint",
                    side_effect=changing_copy):
                with self.assertRaisesRegex(ValueError, "changed during migration"):
                    self.migration.migrate_epoch_resume(
                        source,
                        target,
                        runtime_project_dir=ROOT,
                        expected_source_project_commit=self.source_commit,
                        target_project_commit=self.target_commit,
                    )
            self.assertFalse(target.exists())

    def test_cli_parser_and_main_dispatch(self):
        arguments = SimpleNamespace(
            source_work_dir=Path("source"),
            target_work_dir=Path("target"),
            project_dir=ROOT,
            expected_source_project_commit=self.source_commit,
            target_project_commit=self.target_commit,
        )
        with (
            mock.patch.object(self.migration, "parse_args", return_value=arguments),
            mock.patch.object(self.migration, "validate_runtime_checkout") as checkout,
            mock.patch.object(
                self.migration, "migrate_epoch_resume",
                return_value=Path("/private/target"),
            ) as migrate,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(self.migration.main(), 0)
        checkout.assert_not_called()
        migrate.assert_called_once_with(
            Path("source"),
            Path("target"),
            runtime_project_dir=ROOT,
            expected_source_project_commit=self.source_commit,
            target_project_commit=self.target_commit,
        )

    def test_migrated_resume_launcher_executes_only_attested_runtime(self):
        launcher = _load_script("run_migrated_resume")
        completed = SimpleNamespace(returncode=0)
        with (
            mock.patch.object(
                launcher, "validate_runtime_checkout", return_value=ROOT,
            ) as validate,
            mock.patch.object(
                launcher, "_validate_locked_python",
                return_value=(Path(sys.executable), Path(sys.executable).parent),
            ),
            mock.patch.object(
                launcher.subprocess, "run", return_value=completed,
            ) as run,
        ):
            self.assertEqual(launcher.launch_migrated_resume(
                ROOT,
                self.target_commit,
                Path(sys.executable),
                ("--dataset", "refcoco", "--resume"),
            ), 0)
        validate.assert_called_once_with(ROOT, self.target_commit)
        command = run.call_args.args[0]
        self.assertEqual(command[1], str(ROOT / "scripts/run_phase1_server.sh"))
        self.assertIn("--resume", command)
        self.assertEqual(command[-2:], ("--python", str(Path(sys.executable))))
        environment = run.call_args.kwargs["env"]
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("BASH_ENV", environment)
        self.assertNotIn("LD_LIBRARY_PATH", environment)
        self.assertFalse(any(key.startswith("NCCL_") for key in environment))
        self.assertFalse(any(key.startswith("OMP_") for key in environment))
        self.assertTrue(environment["PATH"].endswith(os.defpath))
        self.assertNotIn(str(Path(sys.executable).parent), environment["PATH"])

        with self.assertRaisesRegex(ValueError, "requires --resume"):
            launcher.launch_migrated_resume(
                ROOT, self.target_commit, Path(sys.executable),
                ("--dataset", "refcoco"))
        with self.assertRaisesRegex(ValueError, "cannot use --dry-run"):
            launcher.launch_migrated_resume(
                ROOT, self.target_commit, Path(sys.executable),
                ("--resume", "--dry-run"))
        with self.assertRaisesRegex(ValueError, "--python-bin"):
            launcher.launch_migrated_resume(
                ROOT, self.target_commit, Path(sys.executable),
                ("--resume", "--python", "/tmp/python"))
        with (
            mock.patch.object(
                launcher, "validate_runtime_checkout", return_value=ROOT),
            mock.patch.object(
                launcher, "_validate_locked_python",
                return_value=(Path(sys.executable), Path(sys.executable).parent),
            ),
            mock.patch.object(launcher.shutil, "which", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "Bash"):
                launcher.launch_migrated_resume(
                    ROOT, self.target_commit, Path(sys.executable),
                    ("--resume",))

        arguments = SimpleNamespace(
            runtime_project_dir=ROOT,
            target_project_commit=self.target_commit,
            python_bin=Path(sys.executable),
            runner_arguments=("--", "--resume"),
        )
        with (
            mock.patch.object(launcher, "parse_args", return_value=arguments),
            mock.patch.object(
                launcher, "launch_migrated_resume", return_value=7,
            ) as launch,
        ):
            self.assertEqual(launcher.main(), 7)
        launch.assert_called_once_with(
            ROOT, self.target_commit, Path(sys.executable), ("--resume",))

        argv = [
            "run_migrated_resume.py",
            "--runtime-project-dir", str(ROOT),
            "--target-project-commit", self.target_commit,
            "--python-bin", sys.executable,
            "--", "--resume",
        ]
        with mock.patch.object(sys, "argv", argv):
            parsed = launcher.parse_args()
        self.assertEqual(parsed.runtime_project_dir, ROOT)
        self.assertEqual(parsed.python_bin, Path(sys.executable))
        self.assertEqual(parsed.runner_arguments, ["--", "--resume"])

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            environment_root = Path(directory) / "locked-env"
            bin_dir = environment_root / "bin"
            bin_dir.mkdir(parents=True)
            python_link = bin_dir / "python"
            python_link.symlink_to(Path(sys.executable).resolve())
            for path in (bin_dir, environment_root):
                path.chmod(0o555)
            self.assertEqual(
                launcher._validate_locked_python(python_link),
                (python_link, bin_dir),
            )
            environment_root.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "directory is unsafe"):
                launcher._validate_locked_python(python_link)
            bin_dir.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
