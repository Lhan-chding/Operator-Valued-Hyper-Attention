import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_resume_guard():
    script = ROOT / "scripts/resume_guard.py"
    spec = importlib.util.spec_from_file_location("ovha_resume_guard_observer", script)
    module = importlib.util.module_from_spec(spec)
    contracts = types.ModuleType("ovha_rod.runtime_contracts")
    contracts.validate_private_epoch_checkpoint = lambda path: Path(path)
    package = types.ModuleType("ovha_rod")
    package.runtime_contracts = contracts
    previous_package = sys.modules.get("ovha_rod")
    previous_contracts = sys.modules.get("ovha_rod.runtime_contracts")
    sys.modules["ovha_rod"] = package
    sys.modules["ovha_rod.runtime_contracts"] = contracts
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_package is None:
            sys.modules.pop("ovha_rod", None)
        else:
            sys.modules["ovha_rod"] = previous_package
        if previous_contracts is None:
            sys.modules.pop("ovha_rod.runtime_contracts", None)
        else:
            sys.modules["ovha_rod.runtime_contracts"] = previous_contracts
    return module


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ObserverResumeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.guard = _load_resume_guard()

    def test_identity_exception_only_allows_project_commit_change(self):
        source = {
            "dataset": "refcoco",
            "variant": "rqgo",
            "seed": "2026",
            "gpus": "1",
            "per_device_batch": "8",
            "accumulative_counts": "4",
            "global_batch": "32",
            "amp": "false",
            "amp_dtype": "none",
            "project_commit": "a" * 40,
        }
        target = {**source, "project_commit": "b" * 40}

        self.guard.validate_observer_identity_transition(source, target)

        for field, value in (
            ("per_device_batch", "16"),
            ("amp", "true"),
            ("amp_dtype", "float16"),
        ):
            with self.subTest(field=field):
                changed = {**target, field: value}
                with self.assertRaisesRegex(ValueError, field):
                    self.guard.validate_observer_identity_transition(
                        source, changed)

    def test_repository_transition_requires_exact_reviewed_blobs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            repo = Path(directory)
            _git(repo, "init", "-q")
            _git(repo, "config", "user.email", "tests@example.invalid")
            _git(repo, "config", "user.name", "OVHA tests")
            _write(repo / "train.py", "TRAINING = 'locked'\n")
            _write(repo / "observer.py", "SYNC = 'many'\n")
            _git(repo, "add", "train.py", "observer.py")
            _git(repo, "commit", "-qm", "source")
            source_commit = _git(repo, "rev-parse", "HEAD")

            _write(repo / "observer.py", "SYNC = 'one'\n")
            _write(repo / "observer_helper.py", "DETACHED = True\n")
            _git(repo, "add", "observer.py", "observer_helper.py")
            _git(repo, "commit", "-qm", "observer only")
            target_commit = _git(repo, "rev-parse", "HEAD")
            policy = self.guard.ObserverCompatibilityPolicy(
                source_commit=source_commit,
                target_blobs={
                    "observer.py": _sha256(repo / "observer.py"),
                    "observer_helper.py": _sha256(repo / "observer_helper.py"),
                },
            )

            result = self.guard.validate_observer_repository_transition(
                repo, source_commit, target_commit, policy)
            self.assertEqual(result["source_commit"], source_commit)
            self.assertEqual(result["target_commit"], target_commit)
            self.assertEqual(
                set(result["changed_paths"]),
                {"observer.py", "observer_helper.py"},
            )

            _write(repo / "train.py", "TRAINING = 'changed'\n")
            _git(repo, "add", "train.py")
            _git(repo, "commit", "-qm", "unauthorized training change")
            unauthorized = _git(repo, "rev-parse", "HEAD")
            with self.assertRaisesRegex(ValueError, "unauthorized"):
                self.guard.validate_observer_repository_transition(
                    repo, source_commit, unauthorized, policy)

    def test_continuation_manifest_binds_source_without_rewriting_it(self):
        source_identity = b'{"contract":"source"}\n'
        source_provenance = b'{"contract":"provenance"}\n'
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            root.chmod(0o700)
            source = root / "source"
            target = root / "target"
            source.mkdir(mode=0o700)
            target.mkdir(mode=0o700)
            identity_path = source / "run_identity.json"
            provenance_path = source / "epoch_1.pth.provenance.json"
            identity_path.write_bytes(source_identity)
            provenance_path.write_bytes(source_provenance)
            identity_path.chmod(0o600)
            provenance_path.chmod(0o600)

            manifest = self.guard.write_observer_continuation_manifest(
                target,
                source_identity_path=identity_path,
                source_provenance_path=provenance_path,
                source_checkpoint_sha256="1" * 64,
                source_commit="a" * 40,
                target_commit="b" * 40,
                transition={"changed_paths": ["observer.py"]},
            )

            self.assertEqual(identity_path.read_bytes(), source_identity)
            self.assertEqual(provenance_path.read_bytes(), source_provenance)
            self.assertEqual(manifest.stat().st_mode & 0o777, 0o600)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["source_identity_sha256"],
                hashlib.sha256(source_identity).hexdigest(),
            )
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.guard.write_observer_continuation_manifest(
                    target,
                    source_identity_path=identity_path,
                    source_provenance_path=provenance_path,
                    source_checkpoint_sha256="1" * 64,
                    source_commit="a" * 40,
                    target_commit="b" * 40,
                    transition={"changed_paths": ["observer.py"]},
                )

    def test_server_observer_resume_is_explicit_and_separate(self):
        runner = (ROOT / "scripts/run_phase1_server.sh").read_text()
        lock = (ROOT / "scripts/run_lock.py").read_text()
        self.assertIn("--observer-resume-from", runner)
        self.assertIn('LOCK_MODE="observer-resume"', runner)
        self.assertIn('--source-work-dir "${OBSERVER_RESUME_FROM}"', runner)
        self.assertIn('choices=("fresh", "resume", "observer-resume")', lock)
        self.assertIn("prepare_observer_compatible_resume", lock)
        self.assertIn("observer_continuation.json", lock)


if __name__ == "__main__":
    unittest.main()
