import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ovha_rod.runtime_contracts import (
    audit_smoke_outputs,
    collect_scalar_diagnostics,
    find_remote_weight_values,
    prepare_fresh_private_work_dir,
    require_selected_gpus_idle,
    require_visible_device_ids,
)


ROOT = Path(__file__).resolve().parents[2]


class RuntimeContractTests(unittest.TestCase):
    def test_visible_devices_are_explicit_unique_and_counted(self):
        self.assertEqual(require_visible_device_ids("4", 1), (4,))
        self.assertEqual(require_visible_device_ids("4,5", 2), (4, 5))
        for value, count in ((None, 1), ("", 1), ("4,4", 2), ("4,5", 1)):
            with self.subTest(value=value, count=count):
                with self.assertRaises(ValueError):
                    require_visible_device_ids(value, count)

    def test_selected_gpu_guard_rejects_existing_compute_process(self):
        outputs = iter((
            "0, GPU-zero\n4, GPU-four\n",
            "GPU-four, 1234, python, 1024 MiB\n",
        ))

        def fake_run(*args, **kwargs):
            del args, kwargs
            return SimpleNamespace(returncode=0, stdout=next(outputs), stderr="")

        with self.assertRaisesRegex(RuntimeError, "GPU 4"):
            require_selected_gpus_idle((4,), run_command=fake_run)

    def test_fresh_private_work_dir_rejects_stale_or_symlinked_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            fresh = root / "fresh"
            prepared = prepare_fresh_private_work_dir(fresh)
            self.assertEqual(prepared, fresh.resolve())
            self.assertEqual(prepared.stat().st_mode & 0o777, 0o700)

            (prepared / "stale.txt").write_text("old run")
            with self.assertRaisesRegex(ValueError, "empty"):
                prepare_fresh_private_work_dir(prepared)

            target = root / "target"
            target.mkdir(mode=0o700)
            link = root / "linked"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                prepare_fresh_private_work_dir(link)

    def test_scalar_diagnostics_preserve_nonfinite_failures(self):
        finite, nonfinite = collect_scalar_diagnostics({
            "loss": torch.tensor(1.25),
            "nan_loss": torch.tensor(float("nan")),
            "vector": torch.ones(2),
        })
        self.assertEqual(finite, {"loss": 1.25})
        self.assertEqual(nonfinite, ("nan_loss",))

    def test_smoke_audit_requires_current_operator_rows_and_fields(self):
        required = {
            "seed_operator": "rqgo",
            "loss_seed_weight": 0.5,
            "seed_gradient_norm": 1.0,
            "seed_bias_abs_max": 0.1,
            "seed_invalid_bias_abs_max": 0.0,
            "nonfinite_scalar_count": 0,
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            work_dir = Path(directory)
            rows = [dict(required, iteration=index) for index in (1, 2)]
            (work_dir / "operator_diagnostics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            (work_dir / "iter_2.pth").write_bytes(b"checkpoint")
            self.assertTrue(audit_smoke_outputs(work_dir, "rqgo")["ok"])

            rows[0].pop("seed_gradient_norm")
            (work_dir / "operator_diagnostics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            failed = audit_smoke_outputs(work_dir, "rqgo")
            self.assertFalse(failed["ok"])
            self.assertIn("seed_gradient_norm", failed["missing_fields"])

    def test_smoke_audit_rejects_unknown_operator_and_nonfinite_marker(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            work_dir = Path(directory)
            rows = [{
                "iteration": index,
                "seed_operator": "unknown",
                "loss_seed_weight": 0.5,
                "seed_gradient_norm": 1.0,
                "seed_bias_abs_max": 0.1,
                "seed_invalid_bias_abs_max": 0.0,
                "nonfinite_scalar_count": 1,
            } for index in (1, 2)]
            (work_dir / "operator_diagnostics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            (work_dir / "iter_2.pth").write_bytes(b"checkpoint")
            self.assertFalse(audit_smoke_outputs(work_dir, "rqgo")["ok"])

    def test_remote_weight_values_are_found_recursively(self):
        config = {
            "model": {"backbone": {"init_cfg": {
                "checkpoint": "https://example.invalid/swin.pth"}}},
            "load_from": "/private/trusted/model.pth",
        }
        self.assertEqual(
            find_remote_weight_values(config),
            ("https://example.invalid/swin.pth",),
        )


if __name__ == "__main__":
    unittest.main()
