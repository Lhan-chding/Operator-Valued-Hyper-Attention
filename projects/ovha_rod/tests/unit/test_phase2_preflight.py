import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "phase2_preflight.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("phase2_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase2PreflightTests(unittest.TestCase):
    def setUp(self):
        self.preflight = _load_script()
        self.commit = "a" * 40

    @staticmethod
    def _ok(name: str):
        return SimpleNamespace(
            name=name, ok=True, detail="reviewed", observed="ok")

    def test_validates_read_only_runtime_mmdet_data_bert_and_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "projects" / "ovha_rod"
            mmdet = root / "mmdetection"
            data = root / "data"
            bert = root / "bert"
            work = root / "runs"
            for path in (project, mmdet, data, bert):
                path.mkdir(parents=True, exist_ok=True)

            with (
                mock.patch.object(
                    self.preflight, "validate_runtime_checkout",
                    return_value=project.resolve()) as runtime,
                mock.patch.object(
                    self.preflight, "validate_local_bert",
                    return_value=bert.resolve()) as local_bert,
                mock.patch.object(
                    self.preflight, "_mmdet_checks",
                    return_value=(self._ok("mmdet"),)),
                mock.patch.object(
                    self.preflight, "_data_checks",
                    return_value=(self._ok("data"),)),
                mock.patch.object(
                    self.preflight, "_storage_checks",
                    return_value=(self._ok("storage"),)),
            ):
                payload = self.preflight.validate_phase2_preflight(
                    project,
                    target_project_commit=self.commit,
                    mmdet_root=mmdet,
                    data_root=data,
                    bert_root=bert,
                    work_root=work,
                )

            runtime.assert_called_once_with(project, self.commit)
            local_bert.assert_called_once_with(bert)
            self.assertEqual(payload["target_project_commit"], self.commit)
            self.assertEqual(payload["contract"], "ovha_rod_phase2_preflight_v1")

    def test_fails_closed_when_any_reviewed_check_fails(self):
        failed = SimpleNamespace(
            name="mmdetection_clean",
            ok=False,
            detail="checkout is dirty",
            observed="dirty",
        )
        with (
            mock.patch.object(
                self.preflight, "validate_runtime_checkout",
                return_value=ROOT),
            mock.patch.object(
                self.preflight, "validate_local_bert",
                return_value=ROOT),
            mock.patch.object(
                self.preflight, "_mmdet_checks", return_value=(failed,)),
            mock.patch.object(
                self.preflight, "_data_checks", return_value=()),
            mock.patch.object(
                self.preflight, "_storage_checks", return_value=()),
        ):
            with self.assertRaisesRegex(RuntimeError, "mmdetection_clean"):
                self.preflight.validate_phase2_preflight(
                    ROOT,
                    target_project_commit=self.commit,
                    mmdet_root=ROOT,
                    data_root=ROOT,
                    bert_root=ROOT,
                    work_root=ROOT,
                )


if __name__ == "__main__":
    unittest.main()
