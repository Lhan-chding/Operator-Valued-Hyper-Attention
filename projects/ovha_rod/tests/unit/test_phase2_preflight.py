import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
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
                    self.preflight, "_python_checks",
                    return_value=(self._ok("python"),)),
                mock.patch.object(
                    self.preflight, "_package_checks",
                    return_value=(self._ok("packages"),)),
                mock.patch.object(
                    self.preflight, "_data_checks",
                    return_value=(self._ok("data"),)),
                mock.patch.object(
                    self.preflight, "_storage_checks",
                    return_value=(self._ok("storage"),)),
                mock.patch.object(
                    self.preflight, "_cuda_checks",
                    return_value=(self._ok("cuda"),)),
                mock.patch.object(
                    self.preflight, "_phase2_build_checks",
                    return_value=(self._ok("phase2_build"),)) as build,
            ):
                payload = self.preflight.validate_phase2_preflight(
                    project,
                    target_project_commit=self.commit,
                    mmdet_root=mmdet,
                    data_root=data,
                    bert_root=bert,
                    work_root=work,
                    per_device_batch=8,
                    accumulative_counts=4,
                    warmup_iters=2000,
                    seed=2026,
                )

            runtime.assert_called_once_with(project, self.commit)
            local_bert.assert_called_once_with(bert)
            self.assertEqual(payload["target_project_commit"], self.commit)
            self.assertEqual(payload["contract"], "ovha_rod_phase2_preflight_v1")
            build.assert_called_once_with(
                project.resolve(),
                data_root=data.resolve(),
                bert_root=bert.resolve(),
                work_root=work.resolve(),
                per_device_batch=8,
                accumulative_counts=4,
                warmup_iters=2000,
                seed=2026,
            )

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
                self.preflight, "_python_checks", return_value=()),
            mock.patch.object(
                self.preflight, "_package_checks", return_value=()),
            mock.patch.object(
                self.preflight, "_data_checks", return_value=()),
            mock.patch.object(
                self.preflight, "_storage_checks", return_value=()),
            mock.patch.object(
                self.preflight, "_cuda_checks", return_value=()),
            mock.patch.object(
                self.preflight, "_phase2_build_checks", return_value=()),
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

    def test_training_surface_requires_exact_operator_optimizer_identity(self):
        operator_weight = SimpleNamespace(requires_grad=True)
        frozen_weight = SimpleNamespace(requires_grad=False)
        model = SimpleNamespace(
            named_parameters=lambda: (
                ("decoder_operator.weight", operator_weight),
                ("backbone.weight", frozen_weight),
            ),
            decoder_operator=SimpleNamespace(
                parameters=lambda: (operator_weight,)),
        )
        wrapper = SimpleNamespace(
            optimizer=SimpleNamespace(
                param_groups=({"params": (operator_weight,)},)))

        checks = self.preflight._training_surface_checks(model, wrapper)

        self.assertTrue(all(check.ok for check in checks), checks)
        self.assertEqual(
            {check.name for check in checks},
            {
                "phase2_operator_trainable_only",
                "phase2_optimizer_exact_trainable_set",
            },
        )

    def test_training_surface_rejects_a_frozen_parameter_in_optimizer(self):
        operator_weight = SimpleNamespace(requires_grad=True)
        frozen_weight = SimpleNamespace(requires_grad=False)
        model = SimpleNamespace(
            named_parameters=lambda: (
                ("decoder_operator.weight", operator_weight),
                ("backbone.weight", frozen_weight),
            ),
            decoder_operator=SimpleNamespace(
                parameters=lambda: (operator_weight,)),
        )
        wrapper = SimpleNamespace(
            optimizer=SimpleNamespace(
                param_groups=({"params": (operator_weight, frozen_weight)},)))

        checks = self.preflight._training_surface_checks(model, wrapper)

        by_name = {check.name: check for check in checks}
        self.assertTrue(by_name["phase2_operator_trainable_only"].ok)
        self.assertFalse(
            by_name["phase2_optimizer_exact_trainable_set"].ok)

    def _fake_build_runtime(self, *, remote_weights=()):
        class AttrDict(dict):
            __getattr__ = dict.__getitem__

        operator_weight = SimpleNamespace(requires_grad=True)
        frozen_weight = SimpleNamespace(requires_grad=False)
        model = SimpleNamespace(
            named_parameters=lambda: (
                ("decoder_operator.weight", operator_weight),
                ("backbone.weight", frozen_weight),
            ),
            decoder_operator=SimpleNamespace(
                parameters=lambda: (operator_weight,)),
        )
        wrapper = SimpleNamespace(
            optimizer=SimpleNamespace(
                param_groups=({"params": (operator_weight,)},)))
        bank = AttrDict(
            enabled=True,
            enabled_operators=("qsro", "tq_cato", "ms_tleo"),
            qsro_query_chunk_size=128,
            use_router=True,
            use_memory=True,
            use_hyper_adapter=True,
            use_rceo=True,
        )
        config = AttrDict(
            load_from=None,
            resume=False,
            model=AttrDict(
                train_decoder_operator_only=True,
                decoder_operator_cfg=bank,
            ),
            val_evaluator=AttrDict(type="OVHARefExpMetric"),
            train_cfg=AttrDict(max_epochs=3),
            train_dataloader=AttrDict(batch_size=8),
            optim_wrapper=AttrDict(
                accumulative_counts=4,
                constructor="TrainableOnlyOptimWrapperConstructor",
            ),
            param_scheduler=[AttrDict(end=2000)],
            randomness=AttrDict(seed=2026, deterministic=False),
        )
        config.merge_from_dict = mock.Mock()

        config_module = types.ModuleType("mmengine.config")
        config_module.Config = SimpleNamespace(
            fromfile=mock.Mock(return_value=config))
        optim_module = types.ModuleType("mmengine.optim")
        optim_module.build_optim_wrapper = mock.Mock(return_value=wrapper)
        registry_module = types.ModuleType("mmengine.registry")
        registry_module.init_default_scope = mock.Mock()
        mmdet_registry = types.ModuleType("mmdet.registry")
        mmdet_registry.MODELS = SimpleNamespace(
            build=mock.Mock(return_value=model))
        mmdet_registry.METRICS = SimpleNamespace(
            build=mock.Mock(return_value=SimpleNamespace()))
        mmengine = types.ModuleType("mmengine")
        mmengine.__path__ = []
        mmdet = types.ModuleType("mmdet")
        mmdet.__path__ = []
        modules = {
            "mmengine": mmengine,
            "mmengine.config": config_module,
            "mmengine.optim": optim_module,
            "mmengine.registry": registry_module,
            "mmdet": mmdet,
            "mmdet.registry": mmdet_registry,
        }
        runtime_contracts = sys.modules["ovha_rod.runtime_contracts"]
        return (
            modules,
            runtime_contracts,
            config,
            tuple(remote_weights),
        )

    def test_builds_the_resolved_model_metric_and_exact_optimizer_surface(self):
        modules, contracts, config, remote_weights = (
            self._fake_build_runtime())
        with (
            mock.patch.dict(sys.modules, modules, clear=False),
            mock.patch.object(
                contracts,
                "find_remote_weight_values",
                return_value=remote_weights,
            ),
        ):
            checks = self.preflight._phase2_build_checks(
                ROOT,
                data_root=Path("/private/coco"),
                bert_root=Path("/private/bert"),
                work_root=Path("/private/runs"),
                per_device_batch=8,
                accumulative_counts=4,
                warmup_iters=2000,
                seed=2026,
            )

        self.assertTrue(all(check.ok for check in checks), checks)
        overrides = config.merge_from_dict.call_args.args[0]
        self.assertEqual(overrides["train_dataloader.batch_size"], 8)
        self.assertEqual(overrides["optim_wrapper.accumulative_counts"], 4)
        self.assertEqual(overrides["param_scheduler.0.end"], 2000)
        self.assertEqual(overrides["randomness.seed"], 2026)

    def test_resolved_build_rejects_remote_weights(self):
        modules, contracts, _, remote_weights = self._fake_build_runtime(
            remote_weights=("https://example.invalid/model.pth",))
        with (
            mock.patch.dict(sys.modules, modules, clear=False),
            mock.patch.object(
                contracts,
                "find_remote_weight_values",
                return_value=remote_weights,
            ),
        ):
            checks = self.preflight._phase2_build_checks(
                ROOT,
                data_root=Path("/private/coco"),
                bert_root=Path("/private/bert"),
                work_root=Path("/private/runs"),
                per_device_batch=8,
                accumulative_counts=4,
                warmup_iters=2000,
                seed=2026,
            )

        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0].ok)
        self.assertIn("remote weights", checks[0].detail)

    def test_rejects_invalid_numeric_launch_inputs_before_checks(self):
        base = dict(
            target_project_commit=self.commit,
            mmdet_root=ROOT,
            data_root=ROOT,
            bert_root=ROOT,
            work_root=ROOT,
        )
        for override, message in (
            ({"per_device_batch": 0}, "positive"),
            ({"accumulative_counts": 0}, "positive"),
            ({"warmup_iters": 0}, "positive"),
            ({"seed": -1}, "non-negative"),
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    self.preflight.validate_phase2_preflight(
                        ROOT, **base, **override)

    def test_main_emits_machine_readable_attestation(self):
        payload = {
            "contract": "ovha_rod_phase2_preflight_v1",
            "target_project_commit": self.commit,
        }
        arguments = [
            "phase2_preflight.py",
            "--runtime-project-dir", "/runtime/projects/ovha_rod",
            "--target-project-commit", self.commit,
            "--mmdet-root", "/runtime/mmdetection",
            "--data-root", "/private/data",
            "--bert-root", "/private/bert",
            "--work-root", "/private/runs",
            "--per-device-batch", "8",
            "--accumulative-counts", "4",
            "--warmup-iters", "2000",
            "--seed", "2026",
        ]
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", arguments),
            mock.patch.object(
                self.preflight, "validate_phase2_preflight",
                return_value=payload) as validate,
            redirect_stdout(output),
        ):
            self.assertEqual(self.preflight.main(), 0)

        validate.assert_called_once()
        self.assertEqual(json.loads(output.getvalue()), payload)


if __name__ == "__main__":
    unittest.main()
