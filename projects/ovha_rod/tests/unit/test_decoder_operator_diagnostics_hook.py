import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


class _RegistryStub:
    def register_module(self):
        return lambda target: target


def _load_hook_class():
    importlib.import_module("ovha_rod")
    mmengine = types.ModuleType("mmengine")
    mmengine_dist = types.ModuleType("mmengine.dist")
    mmengine_dist.is_main_process = lambda: True
    mmengine_hooks = types.ModuleType("mmengine.hooks")
    mmengine_hooks.Hook = object
    mmdet = types.ModuleType("mmdet")
    mmdet_registry = types.ModuleType("mmdet.registry")
    mmdet_registry.HOOKS = _RegistryStub()
    stubs = {
        "mmengine": mmengine,
        "mmengine.dist": mmengine_dist,
        "mmengine.hooks": mmengine_hooks,
        "mmdet": mmdet,
        "mmdet.registry": mmdet_registry,
    }
    with mock.patch.dict(sys.modules, stubs):
        for name in (
            "ovha_rod.hooks",
            "ovha_rod.hooks.checkpoint_provenance_hook",
            "ovha_rod.hooks.operator_diagnostics_hook",
            "ovha_rod.hooks.seed_loss_warmup_hook",
        ):
            sys.modules.pop(name, None)
        module = importlib.import_module(
            "ovha_rod.hooks.operator_diagnostics_hook")
    return module.OperatorDiagnosticsHook


class _ModelStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.seed_operator = None
        self.seed_operator_name = "rqgo"
        self.decoder_operator = nn.Linear(2, 1, bias=False)
        self.bbox_head = SimpleNamespace(
            last_seed_metrics={},
            loss_seed_weight=0.5,
        )
        self.last_seed_diagnostics = {}
        self.last_decoder_operator_diagnostics = {}


class DecoderOperatorDiagnosticsHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hook_class = _load_hook_class()

    def _runner(self, work_dir: str):
        return SimpleNamespace(
            model=_ModelStub(),
            iter=0,
            work_dir=work_dir,
        )

    @staticmethod
    def _read_row(work_dir: str):
        path = Path(work_dir) / "operator_diagnostics.jsonl"
        return json.loads(path.read_text(encoding="utf-8").strip())

    def test_persists_decoder_operator_diagnostics(self):
        with tempfile.TemporaryDirectory() as work_dir:
            runner = self._runner(work_dir)
            runner.model.last_decoder_operator_diagnostics = {
                "decoder_qsro_router_weight_mean": torch.tensor(0.25),
            }
            hook = self.hook_class(interval=1)

            hook.before_train(runner)
            hook.after_train_iter(runner, batch_idx=0, outputs={})

            row = self._read_row(work_dir)
            self.assertAlmostEqual(
                row["decoder_qsro_router_weight_mean"], 0.25)

    def test_records_aggregate_decoder_operator_gradient_norm(self):
        with tempfile.TemporaryDirectory() as work_dir:
            runner = self._runner(work_dir)
            hook = self.hook_class(interval=1)
            hook.before_train(runner)
            weighted_parameters = (
                runner.model.decoder_operator.weight
                * torch.tensor([[3.0, 4.0]])
            ).sum()

            weighted_parameters.backward()
            hook.after_train_iter(runner, batch_idx=0, outputs={})

            row = self._read_row(work_dir)
            self.assertAlmostEqual(
                row["decoder_operator_gradient_norm"], 5.0, places=6)

    def test_nonfinite_decoder_diagnostics_fail_fast_before_logging_gate(self):
        with tempfile.TemporaryDirectory() as work_dir:
            runner = self._runner(work_dir)
            runner.model.last_decoder_operator_diagnostics = {
                "decoder_memory_norm": torch.tensor(float("nan")),
            }
            runner.iter = 1
            hook = self.hook_class(interval=50)
            hook.before_train(runner)

            with self.assertRaisesRegex(
                    FloatingPointError, "decoder_memory_norm"):
                hook.after_train_iter(runner, batch_idx=1, outputs={})


if __name__ == "__main__":
    unittest.main()
