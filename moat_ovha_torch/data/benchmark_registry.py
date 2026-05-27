from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    domain: str
    input_type: str
    output_type: str
    resolution: str
    train_size: int | None
    val_size: int | None
    test_size: int | None
    supports_resolution_transfer: bool
    supports_parameter_holdout: bool
    supports_context_episodes: bool
    requires_download: bool
    local_path: str | None


class BenchmarkRegistry:
    def __init__(self, specs: Iterable[DatasetSpec]):
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> DatasetSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._specs))
            raise KeyError(f"unknown benchmark dataset {name!r}; available: {available}") from exc

    def list_specs(self) -> tuple[DatasetSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


def get_default_benchmark_registry() -> BenchmarkRegistry:
    return BenchmarkRegistry(
        (
            DatasetSpec(
                name="pdebench_burgers_1d",
                domain="pde",
                input_type="grid_field",
                output_type="field",
                resolution="1d_variable",
                train_size=None,
                val_size=None,
                test_size=None,
                supports_resolution_transfer=True,
                supports_parameter_holdout=True,
                supports_context_episodes=True,
                requires_download=True,
                local_path=None,
            ),
            DatasetSpec(
                name="pdebench_darcy_2d",
                domain="pde",
                input_type="grid_field",
                output_type="field",
                resolution="2d_variable",
                train_size=None,
                val_size=None,
                test_size=None,
                supports_resolution_transfer=True,
                supports_parameter_holdout=True,
                supports_context_episodes=True,
                requires_download=True,
                local_path=None,
            ),
            DatasetSpec(
                name="pdebench_shallow_water_2d",
                domain="pde",
                input_type="grid_field",
                output_type="field",
                resolution="2d_time_dependent",
                train_size=None,
                val_size=None,
                test_size=None,
                supports_resolution_transfer=True,
                supports_parameter_holdout=True,
                supports_context_episodes=True,
                requires_download=True,
                local_path=None,
            ),
            DatasetSpec(
                name="fno_burgers",
                domain="pde",
                input_type="grid_field",
                output_type="field",
                resolution="1d_canonical",
                train_size=None,
                val_size=None,
                test_size=None,
                supports_resolution_transfer=True,
                supports_parameter_holdout=False,
                supports_context_episodes=True,
                requires_download=True,
                local_path=None,
            ),
            DatasetSpec(
                name="fno_darcy",
                domain="pde",
                input_type="grid_field",
                output_type="field",
                resolution="2d_canonical",
                train_size=None,
                val_size=None,
                test_size=None,
                supports_resolution_transfer=True,
                supports_parameter_holdout=False,
                supports_context_episodes=True,
                requires_download=True,
                local_path=None,
            ),
            DatasetSpec(
                name="fno_navier_stokes",
                domain="pde",
                input_type="grid_field",
                output_type="field",
                resolution="2d_time_dependent",
                train_size=None,
                val_size=None,
                test_size=None,
                supports_resolution_transfer=True,
                supports_parameter_holdout=False,
                supports_context_episodes=True,
                requires_download=True,
                local_path=None,
            ),
            DatasetSpec(
                name="mechanical_mnist_small",
                domain="material",
                input_type="image",
                output_type="field",
                resolution="2d_small",
                train_size=None,
                val_size=None,
                test_size=None,
                supports_resolution_transfer=False,
                supports_parameter_holdout=True,
                supports_context_episodes=True,
                requires_download=True,
                local_path=None,
            ),
        )
    )
