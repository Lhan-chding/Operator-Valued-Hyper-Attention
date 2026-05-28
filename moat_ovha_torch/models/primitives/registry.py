from __future__ import annotations

from torch import nn

from moat_ovha_torch.models.primitives.local_kernel import LocalKernelPrimitive
from moat_ovha_torch.models.primitives.mlp_expert import MLPExpertPrimitive
from moat_ovha_torch.models.primitives.separable import SeparableBasisPrimitive
from moat_ovha_torch.models.primitives.spectral import SpectralIntegralPrimitive
from moat_ovha_torch.models.primitives.vector_value import VectorValuePrimitive


def make_primitive_registry(names: tuple[str, ...]) -> nn.ModuleDict:
    builders = {
        "spectral": SpectralIntegralPrimitive,
        "separable": SeparableBasisPrimitive,
        "local": LocalKernelPrimitive,
        "vector_value": VectorValuePrimitive,
        "mlp_expert": MLPExpertPrimitive,
    }
    modules = {}
    for name in names:
        builder_name = _builder_name(name)
        modules[name] = builders[builder_name]()
    return nn.ModuleDict(modules)


def _builder_name(name: str) -> str:
    if name.startswith("mlp_expert"):
        return "mlp_expert"
    return name
