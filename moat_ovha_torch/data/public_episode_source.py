from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from moat_ovha_torch.config import Phase15Config
from moat_ovha_torch.data.episodes import EpisodeHiddenInfo
from moat_ovha_torch.data.field_episode_adapter import FieldToEpisodeAdapter
from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
from moat_ovha_torch.data.public_benchmarks import MissingBenchmarkDataError, PublicBenchmarkLoader
from moat_ovha_torch.data.public_benchmarks.fno_classic_loader import FNOClassicSubsetLoader
from moat_ovha_torch.data.public_benchmarks.mechanical_mnist_loader import MechanicalMNISTSubsetLoader
from moat_ovha_torch.data.public_benchmarks.pdebench_loader import PDEBenchSubsetLoader
from moat_ovha_torch.runtime import require_torch


PUBLIC_DATA_ROOT_ENV = "OVHA_PUBLIC_BENCHMARK_ROOT"


class SyntheticEpisodeSource:
    def __init__(self, seed: int):
        self._zoo = MetadataFreeOperatorZoo(seed=seed)

    def sample_batch(self, **kwargs: Any):
        return self._zoo.sample_batch(**kwargs)


class PublicBenchmarkEpisodeSource:
    """Sample metadata-free episodes from local public benchmark field caches."""

    def __init__(self, config: Phase15Config, seed: int):
        self.config = config
        self.seed = seed
        self.root = _public_data_root(config)
        self.adapter = FieldToEpisodeAdapter(seed=seed)
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._source_splits: dict[tuple[str, str], str] = {}

    def sample_batch(
        self,
        *,
        batch_size: int,
        num_demos: int,
        context_points: int,
        support_points: int,
        query_points: int,
        family: str,
        split: str,
        mode: str,
        device: str,
        episode_id: int = 0,
        **_: Any,
    ):
        del support_points
        sample_batch = self._sample_field_batch(family, split, batch_size, episode_id, device)
        batch = self.adapter.sample_episode(
            sample_batch,
            num_demos=num_demos,
            context_points=context_points,
            query_points=query_points,
            mode=mode,
            context_sampling=self.config.public_context_sampling,
            episode_id=episode_id,
        )
        source_split = self._source_splits[(family, split)]
        hidden = EpisodeHiddenInfo(
            family=family,
            latent_params={},
            mixture_weights=None,
            oracle_hints={
                "dataset": family,
                "source_split": source_split,
                "public_benchmark": True,
                "public_cache_root": str(self.root),
            },
        )
        return batch, hidden

    def _sample_field_batch(self, family: str, split: str, batch_size: int, episode_id: int, device: str) -> dict[str, Any]:
        torch = require_torch()
        data = self._load_split(family, split)
        total = int(data["input_field"].shape[0])
        if total <= 0:
            raise ValueError(f"public benchmark split is empty: family={family!r} split={split!r}")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed * 1_000_003 + episode_id)
        if batch_size <= total:
            indices = torch.randperm(total, generator=generator, device="cpu")[:batch_size]
        else:
            indices = torch.randint(0, total, (batch_size,), generator=generator, device="cpu")
        selected: dict[str, Any] = {}
        for key, value in data.items():
            if key == "coordinates":
                selected[key] = value.to(device)
            elif key == "operator_group_id":
                selected[key] = value.index_select(0, indices).to(device)
            else:
                selected[key] = value.index_select(0, indices).to(device)
        if "operator_group_id" in selected and not _has_repeated_group(selected["operator_group_id"]):
            selected.pop("operator_group_id")
        return selected

    def _load_split(self, family: str, split: str) -> dict[str, Any]:
        cache_key = (family, split)
        if cache_key in self._cache:
            return self._cache[cache_key]
        loader = _loader_for_family(self.root, family, self.config.datasets)
        source_split = _resolve_split(loader, split)
        raw = loader.load_split(source_split)
        normalized = _normalize_public_split(raw, family=family, split=source_split)
        self._cache[cache_key] = normalized
        self._source_splits[cache_key] = source_split
        return normalized


def build_episode_source(config: Phase15Config, seed: int):
    if _should_use_public_source(config):
        return PublicBenchmarkEpisodeSource(config, seed)
    return SyntheticEpisodeSource(seed)


def _should_use_public_source(config: Phase15Config) -> bool:
    has_root = config.public_data_root is not None or bool(os.environ.get(PUBLIC_DATA_ROOT_ENV))
    return has_root and any(_is_public_family(family) for family in config.families)


def _public_data_root(config: Phase15Config) -> Path:
    raw = config.public_data_root or os.environ.get(PUBLIC_DATA_ROOT_ENV)
    if raw is None:
        raise MissingBenchmarkDataError(
            f"public benchmark data root is required; set public_data_root or {PUBLIC_DATA_ROOT_ENV}"
        )
    root = Path(raw)
    if not root.exists():
        raise MissingBenchmarkDataError(f"public benchmark data root does not exist: {root}")
    return root


def _loader_for_family(root: Path, family: str, datasets: tuple[str, ...]) -> PublicBenchmarkLoader:
    loader_cls = _loader_class_for_family(family)
    candidates = [root / family]
    for dataset in datasets:
        candidates.append(root / dataset / family)
        candidates.append(root / dataset)
    existing = next((path for path in candidates if path.exists()), candidates[0])
    return loader_cls(existing)


def _loader_class_for_family(family: str) -> type[PublicBenchmarkLoader]:
    if family.startswith("pdebench_"):
        return PDEBenchSubsetLoader
    if family.startswith("fno_"):
        return FNOClassicSubsetLoader
    if family.startswith("mechanical_mnist"):
        return MechanicalMNISTSubsetLoader
    if family.startswith("openfwi"):
        raise MissingBenchmarkDataError("OpenFWI cache support is not implemented yet; start with PDEBench or Mechanical-MNIST")
    raise MissingBenchmarkDataError(f"unsupported public benchmark family: {family}")


def _is_public_family(family: str) -> bool:
    return family.startswith(("pdebench_", "fno_", "mechanical_mnist", "openfwi"))


def _resolve_split(loader: PublicBenchmarkLoader, split: str) -> str:
    candidates = _split_candidates(split)
    for candidate in candidates:
        try:
            loader.split_path(candidate)
            return candidate
        except MissingBenchmarkDataError:
            continue
    searched = ", ".join(candidates)
    raise MissingBenchmarkDataError(
        f"{loader.dataset_label} subset not found at {loader.root} for split {split!r}; searched: {searched}"
    )


def _split_candidates(split: str) -> tuple[str, ...]:
    if split == "train":
        return ("train", "iid")
    if split == "iid":
        return ("iid", "test", "val", "valid", "validation")
    if split == "parameter_holdout":
        return ("parameter_holdout", "param_holdout", "ood", "test")
    if split == "resolution_transfer":
        return ("resolution_transfer", "resolution", "res_transfer", "test")
    if split == "sparse_context":
        return ("sparse_context", "iid", "test", "val")
    if split == "confusable_context":
        return ("confusable_context", "iid", "test", "val")
    return (split,)


def _normalize_public_split(raw: dict[str, Any], *, family: str, split: str) -> dict[str, Any]:
    torch = require_torch()
    coordinates = _optional_array(raw, ("coordinates", "coords", "grid", "points"))
    input_array = _required_array(raw, ("input_field", "input", "inputs", "u", "x", "a"), family, split)
    output_array = _required_array(raw, ("output_field", "output", "outputs", "target", "targets", "y", "solution"), family, split)
    coordinates_tensor = None if coordinates is None else _coordinates_to_tensor(coordinates, torch)
    input_tensor = _field_to_points(input_array, torch, coordinates_tensor, "input_field")
    output_tensor = _field_to_points(output_array, torch, coordinates_tensor, "output_field")
    if input_tensor.shape[:2] != output_tensor.shape[:2]:
        raise ValueError(
            f"public split {family}/{split} has mismatched input/output point dimensions: "
            f"{tuple(input_tensor.shape)} vs {tuple(output_tensor.shape)}"
        )
    if coordinates_tensor is None:
        coordinates_tensor = torch.linspace(0.0, 1.0, input_tensor.shape[1], dtype=input_tensor.dtype).view(-1, 1)
    if coordinates_tensor.shape[0] != input_tensor.shape[1]:
        raise ValueError(
            f"public split {family}/{split} coordinates length {coordinates_tensor.shape[0]} "
            f"does not match field points {input_tensor.shape[1]}"
        )
    normalized = {
        "input_field": input_tensor.float().contiguous(),
        "output_field": output_tensor.float().contiguous(),
        "coordinates": coordinates_tensor.float().contiguous(),
    }
    group = _optional_array(raw, ("operator_group_id", "operator_id", "group_id"))
    if group is not None:
        group_tensor = torch.as_tensor(group, dtype=torch.long).reshape(-1)
        if group_tensor.shape[0] != input_tensor.shape[0]:
            raise ValueError(
                f"public split {family}/{split} operator_group_id length {group_tensor.shape[0]} "
                f"does not match samples {input_tensor.shape[0]}"
            )
        normalized["operator_group_id"] = group_tensor
    return normalized


def _required_array(raw: dict[str, Any], names: tuple[str, ...], family: str, split: str) -> Any:
    value = _optional_array(raw, names)
    if value is None:
        expected = ", ".join(names)
        available = ", ".join(sorted(raw))
        raise KeyError(f"public split {family}/{split} missing one of [{expected}]; available keys: {available}")
    return value


def _optional_array(raw: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in raw:
            return raw[name]
    return None


def _coordinates_to_tensor(value: Any, torch: Any):
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim >= 3:
        return tensor.reshape(-1, tensor.shape[-1])
    raise ValueError(f"coordinates must have at least one dimension, got shape {tuple(tensor.shape)}")


def _field_to_points(value: Any, torch: Any, coordinates: Any | None, name: str):
    tensor = torch.as_tensor(value)
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have shape [samples, points, ...], got {tuple(tensor.shape)}")
    if coordinates is not None:
        point_count = int(coordinates.shape[0])
        if tensor.ndim >= 3 and int(_prod(tensor.shape[1:-1])) == point_count:
            return tensor.reshape(tensor.shape[0], point_count, tensor.shape[-1])
        if int(_prod(tensor.shape[1:])) == point_count:
            return tensor.reshape(tensor.shape[0], point_count, 1)
    if tensor.ndim == 2:
        return tensor.unsqueeze(-1)
    if tensor.ndim == 3 and tensor.shape[-1] <= 16:
        return tensor
    if tensor.ndim >= 3:
        return tensor.reshape(tensor.shape[0], int(_prod(tensor.shape[1:])), 1)
    return tensor


def _prod(values: Any) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _has_repeated_group(operator_group_id: Any) -> bool:
    torch = require_torch()
    values, counts = torch.unique(operator_group_id.detach().cpu(), return_counts=True)
    return bool(values.numel() > 0 and (counts >= 2).all())
