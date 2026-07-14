import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE = (
    ROOT
    / "ovha_rod"
    / "optim"
    / "trainable_only_optim_wrapper_constructor.py"
)
PACKAGE_INIT = ROOT / "ovha_rod" / "__init__.py"


class _FakeRegistry:
    def __init__(self):
        self.registered = {}

    def register_module(self):
        def decorator(module):
            self.registered[module.__name__] = module
            return module

        return decorator


class _FakeDefaultOptimWrapperConstructor:
    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        self.optim_wrapper_cfg = dict(optim_wrapper_cfg)
        self.paramwise_cfg = paramwise_cfg

    def add_params(
        self,
        params,
        module,
        prefix="",
        is_dcn_module=None,
    ):
        del prefix, is_dcn_module
        params.extend(dict(group) for group in module.param_groups)

    def __call__(self, module):
        if not self.paramwise_cfg:
            return list(module.parameters)
        params = []
        self.add_params(params, module)
        return params


@contextmanager
def _load_constructor_module():
    registry = _FakeRegistry()
    mmengine = types.ModuleType("mmengine")
    optim = types.ModuleType("mmengine.optim")
    optim.DefaultOptimWrapperConstructor = _FakeDefaultOptimWrapperConstructor
    optim.OPTIM_WRAPPER_CONSTRUCTORS = registry
    mmengine.optim = optim

    spec = importlib.util.spec_from_file_location(
        "test_trainable_only_optim_wrapper_constructor", MODULE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"mmengine": mmengine, "mmengine.optim": optim},
    ):
        spec.loader.exec_module(module)
        yield module, registry


class TrainableOnlyOptimWrapperConstructorTests(unittest.TestCase):
    def test_excludes_frozen_parameters_without_mutating_source_groups(self):
        trainable_a = SimpleNamespace(name="a", requires_grad=True)
        frozen = SimpleNamespace(name="frozen", requires_grad=False)
        trainable_b = SimpleNamespace(name="b", requires_grad=True)
        source_groups = [
            {"params": [trainable_a], "lr": 0.1},
            {"params": [frozen], "lr": 0.01},
            {
                "params": [frozen, trainable_b],
                "weight_decay": 0.001,
            },
        ]
        model = SimpleNamespace(param_groups=source_groups)

        with _load_constructor_module() as (module, registry):
            constructor = module.TrainableOnlyOptimWrapperConstructor(
                {"optimizer": {"type": "AdamW"}},
                None,
            )
            filtered = constructor(model)

        self.assertIs(
            registry.registered["TrainableOnlyOptimWrapperConstructor"],
            module.TrainableOnlyOptimWrapperConstructor,
        )
        self.assertTrue(constructor.paramwise_cfg)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0], {"params": [trainable_a], "lr": 0.1})
        self.assertEqual(
            filtered[1],
            {"params": [trainable_b], "weight_decay": 0.001},
        )
        self.assertEqual(source_groups[1]["params"], [frozen])
        self.assertEqual(source_groups[2]["params"], [frozen, trainable_b])

    def test_preserves_explicit_paramwise_configuration(self):
        paramwise = {
            "custom_keys": {
                "decoder_operator": {"lr_mult": 1.0},
            },
        }

        with _load_constructor_module() as (module, _):
            constructor = module.TrainableOnlyOptimWrapperConstructor(
                {"optimizer": {"type": "AdamW"}},
                paramwise,
            )

        self.assertEqual(constructor.paramwise_cfg, paramwise)
        self.assertIsNot(constructor.paramwise_cfg, paramwise)

    def test_top_level_package_imports_constructor_registry_module(self):
        source = PACKAGE_INIT.read_text(encoding="utf-8")

        self.assertIn("from . import optim as _optim", source)


if __name__ == "__main__":
    unittest.main()
