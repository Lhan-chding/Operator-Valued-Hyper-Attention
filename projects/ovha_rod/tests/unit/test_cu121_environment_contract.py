import hashlib
import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _load_preflight_module():
    path = ROOT / "scripts/server_preflight.py"
    spec = importlib.util.spec_from_file_location("ovha_server_preflight", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Cu121EnvironmentContractTests(unittest.TestCase):
    def test_lock_declares_exact_cu121_wheel_profile(self):
        values = {}
        for line in (ROOT / "environment/mmdetection.lock").read_text().splitlines():
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value

        self.assertEqual(values["profile"], "cu121-wheel")
        self.assertEqual(values["torch_version"], "2.1.0")
        self.assertEqual(values["torchvision_version"], "0.16.0")
        self.assertEqual(values["torch_cuda_runtime"], "12.1")
        self.assertEqual(values["default_cuda_wheel"], "cu121")
        self.assertEqual(values["mmcv_version"], "2.1.0")
        self.assertEqual(values["mmcv_install"], "official-prebuilt-wheel")
        self.assertTrue(values["mmcv_wheel_url"].startswith(
            "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/"))
        self.assertIn("cp310-cp310-manylinux1_x86_64.whl", values["mmcv_wheel_url"])

    def test_setup_forbids_source_fallback_and_arbitrary_version_overrides(self):
        source = (ROOT / "scripts/setup_mmdetection.sh").read_text()
        self.assertIn("--only-binary=:all:", source)
        self.assertIn('MMCV_WHEEL_URL=', source)
        self.assertNotIn("mim install 'mmcv==2.1.0'", source)
        self.assertNotIn("--torch-version", source)
        self.assertNotIn("--torchvision-version", source)
        self.assertIn("Python 3.10", source)
        self.assertIn("x86_64", source)

    def test_lock_setup_and_preflight_profiles_are_synchronized(self):
        lock = {}
        for line in (ROOT / "environment/mmdetection.lock").read_text().splitlines():
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                lock[key] = value
        setup = (ROOT / "scripts/setup_mmdetection.sh").read_text()

        def shell_value(name):
            match = re.search(rf'^{name}="([^"]+)"$', setup, re.MULTILINE)
            self.assertIsNotNone(match, name)
            return match.group(1)

        preflight = _load_preflight_module()
        self.assertEqual(shell_value("ENVIRONMENT_PROFILE"), lock["profile"])
        self.assertEqual(shell_value("TORCH_VERSION"), lock["torch_version"])
        self.assertEqual(
            shell_value("TORCHVISION_VERSION"), lock["torchvision_version"])
        self.assertEqual(shell_value("MMCV_WHEEL_URL"), lock["mmcv_wheel_url"])
        self.assertEqual(
            shell_value("MMCV_WHEEL_SHA256"), lock["mmcv_wheel_sha256"])
        self.assertEqual(
            shell_value("MMDET_COMMIT"), lock["mmdetection_commit"])
        self.assertEqual(preflight.ENVIRONMENT_PROFILE, lock["profile"])
        self.assertEqual(
            preflight.EXPECTED_VERSIONS["torch"], lock["torch_version"])
        self.assertEqual(
            preflight.EXPECTED_VERSIONS["torchvision"],
            lock["torchvision_version"])
        self.assertEqual(
            preflight.EXPECTED_TORCH_CUDA, lock["torch_cuda_runtime"])
        self.assertEqual(
            preflight.EXPECTED_CHECKPOINT_SIZE,
            int(lock["checkpoint_size_bytes"]))
        self.assertEqual(
            preflight.EXPECTED_CHECKPOINT_SHA256,
            lock["checkpoint_sha256"])
        self.assertEqual(
            preflight.EXPECTED_BERT_SAFETENSORS_SHA256,
            lock["bert_safetensors_sha256"])
        self.assertEqual(
            preflight.EXPECTED_BERT_SAFETENSORS_SIZE,
            int(lock["bert_safetensors_size_bytes"]))

    def test_preflight_requires_exact_cu121_runtime(self):
        preflight = _load_preflight_module()
        self.assertEqual(preflight.ENVIRONMENT_PROFILE, "cu121-wheel")
        self.assertEqual(preflight.EXPECTED_VERSIONS["torch"], "2.1.0")
        self.assertEqual(preflight.EXPECTED_VERSIONS["torchvision"], "0.16.0")
        self.assertEqual(preflight.EXPECTED_TORCH_CUDA, "12.1")

    def test_checkpoint_path_rejects_symlinks_before_resolution(self):
        preflight = _load_preflight_module()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            target = root / "official.pth"
            target.write_bytes(b"checkpoint")
            link = root / "linked.pth"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                preflight._resolve_trusted_checkpoint(link)

    def test_checkpoint_checks_reject_public_permissions_and_wrong_size(self):
        preflight = _load_preflight_module()
        payload = b"locked checkpoint"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            checkpoint = Path(directory) / "official.pth"
            checkpoint.write_bytes(payload)
            checkpoint.chmod(0o600)
            old_size = preflight.EXPECTED_CHECKPOINT_SIZE
            old_digest = preflight.EXPECTED_CHECKPOINT_SHA256
            try:
                preflight.EXPECTED_CHECKPOINT_SIZE = len(payload)
                preflight.EXPECTED_CHECKPOINT_SHA256 = digest
                checks = preflight._checkpoint_checks(checkpoint, digest)
                self.assertTrue(all(check.ok for check in checks), checks)

                checkpoint.chmod(0o644)
                checks = preflight._checkpoint_checks(checkpoint, digest)
                permission = next(
                    check for check in checks
                    if check.name == "checkpoint_private_permissions")
                self.assertFalse(permission.ok)

                checkpoint.chmod(0o600)
                preflight.EXPECTED_CHECKPOINT_SIZE += 1
                checks = preflight._checkpoint_checks(checkpoint, digest)
                size = next(
                    check for check in checks if check.name == "checkpoint_size")
                self.assertFalse(size.ok)
            finally:
                preflight.EXPECTED_CHECKPOINT_SIZE = old_size
                preflight.EXPECTED_CHECKPOINT_SHA256 = old_digest

    def test_locked_checkpoint_loader_rehashes_before_compatibility_load(self):
        preflight = _load_preflight_module()
        payload = b"locked official checkpoint"
        digest = hashlib.sha256(payload).hexdigest()
        fake_load = mock.Mock(return_value={"state_dict": {}})
        fake_torch = SimpleNamespace(load=fake_load)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            checkpoint = Path(directory) / "official.pth"
            checkpoint.write_bytes(payload)
            checkpoint.chmod(0o600)
            old_size = preflight.EXPECTED_CHECKPOINT_SIZE
            old_digest = preflight.EXPECTED_CHECKPOINT_SHA256
            try:
                preflight.EXPECTED_CHECKPOINT_SIZE = len(payload)
                preflight.EXPECTED_CHECKPOINT_SHA256 = digest
                with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                    loaded = preflight._load_locked_checkpoint(
                        checkpoint, checkpoint_trusted=True)
                self.assertEqual(loaded, {"state_dict": {}})
                fake_load.assert_called_once()
                self.assertEqual(fake_load.call_args.kwargs["map_location"], "cpu")
                self.assertFalse(fake_load.call_args.kwargs["weights_only"])
            finally:
                preflight.EXPECTED_CHECKPOINT_SIZE = old_size
                preflight.EXPECTED_CHECKPOINT_SHA256 = old_digest

    def test_locked_checkpoint_loader_rejects_digest_mismatch_before_torch_load(self):
        preflight = _load_preflight_module()
        payload = b"tampered checkpoint"
        fake_load = mock.Mock()
        fake_torch = SimpleNamespace(load=fake_load)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            checkpoint = Path(directory) / "official.pth"
            checkpoint.write_bytes(payload)
            checkpoint.chmod(0o600)
            old_size = preflight.EXPECTED_CHECKPOINT_SIZE
            old_digest = preflight.EXPECTED_CHECKPOINT_SHA256
            try:
                preflight.EXPECTED_CHECKPOINT_SIZE = len(payload)
                preflight.EXPECTED_CHECKPOINT_SHA256 = "0" * 64
                with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                    with self.assertRaisesRegex(ValueError, "digest"):
                        preflight._load_locked_checkpoint(
                            checkpoint, checkpoint_trusted=True)
                fake_load.assert_not_called()
            finally:
                preflight.EXPECTED_CHECKPOINT_SIZE = old_size
                preflight.EXPECTED_CHECKPOINT_SHA256 = old_digest

    def test_runner_only_resumes_private_guarded_checkpoints(self):
        source = (ROOT / "scripts/run_phase1_server.sh").read_text()
        self.assertIn("--resume", source)
        self.assertIn("resume_guard.py", source)
        self.assertIn('RESUME_CHECKPOINT=', source)
        self.assertIn('COMMAND+=(--resume "${RESUME_CHECKPOINT}")', source)
        self.assertIn("run_identity.json", source)
        self.assertIn("--expected-identity", source)
        self.assertIn("/trusted_inputs", source)
        self.assertIn("LOCKED_CHECKPOINT_SIZE=1093815743", source)

    def test_preflight_requires_safetensors_only_bert(self):
        source = (ROOT / "scripts/server_preflight.py").read_text()
        self.assertIn("EXPECTED_BERT_SAFETENSORS_SHA256", source)
        self.assertIn("and not pytorch_weights.exists()", source)
        self.assertIn("model.safetensors", source)

    def test_executable_tree_allows_internal_links_but_rejects_escape(self):
        preflight = _load_preflight_module()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            tree = Path(directory) / "trusted"
            tree.mkdir(mode=0o700)
            target = tree / "configs"
            target.mkdir(mode=0o700)
            (target / "model.py").write_text("MODEL = True\n")
            (target / "model.py").chmod(0o600)
            internal = tree / "config-link"
            internal.symlink_to(target, target_is_directory=True)

            check = preflight._executable_tree_check(tree, "tree")
            self.assertTrue(check.ok, check)

            external = tree / "escape-link"
            external.symlink_to(Path.home(), target_is_directory=True)
            check = preflight._executable_tree_check(tree, "tree")
            self.assertFalse(check.ok)
            self.assertIn(str(external), check.observed)

    def test_docs_disclose_compatibility_exception(self):
        documentation = "\n".join([
            (ROOT / "README.md").read_text(),
            (ROOT / "RUNBOOK.md").read_text(),
        ]).lower()
        self.assertIn("cu121-wheel", documentation)
        self.assertIn("compatibility", documentation)
        self.assertIn("sha-256", documentation)
        self.assertNotIn("do not downgrade below 2.6.0", documentation)
        self.assertNotRegex(documentation, r"/home/(?!user)[a-z0-9._-]+/")


if __name__ == "__main__":
    unittest.main()
