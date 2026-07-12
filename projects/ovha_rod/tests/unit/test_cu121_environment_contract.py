import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


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

    def test_preflight_requires_exact_cu121_runtime(self):
        preflight = _load_preflight_module()
        self.assertEqual(preflight.ENVIRONMENT_PROFILE, "cu121-wheel")
        self.assertEqual(preflight.EXPECTED_VERSIONS["torch"], "2.1.0")
        self.assertEqual(preflight.EXPECTED_VERSIONS["torchvision"], "0.16.0")
        self.assertEqual(preflight.EXPECTED_TORCH_CUDA, "12.1")

    def test_checkpoint_path_rejects_symlinks_before_resolution(self):
        preflight = _load_preflight_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "official.pth"
            target.write_bytes(b"checkpoint")
            link = root / "linked.pth"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                preflight._resolve_trusted_checkpoint(link)

    def test_docs_disclose_compatibility_exception(self):
        documentation = "\n".join([
            (ROOT / "README.md").read_text(),
            (ROOT / "RUNBOOK.md").read_text(),
        ]).lower()
        self.assertIn("cu121-wheel", documentation)
        self.assertIn("compatibility", documentation)
        self.assertIn("sha-256", documentation)
        self.assertNotIn("do not downgrade below 2.6.0", documentation)


if __name__ == "__main__":
    unittest.main()
