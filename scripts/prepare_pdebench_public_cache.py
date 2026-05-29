#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class SourceSpec:
    family: str
    role: str
    filename: str
    split_prefix: str


SOURCES = (
    SourceSpec("pdebench_burgers_1d", "base", "1D_Burgers_Sols_Nu0.01.hdf5", "base"),
    SourceSpec("pdebench_burgers_1d", "parameter_holdout", "1D_Burgers_Sols_Nu0.001.hdf5", "parameter_holdout"),
    SourceSpec("pdebench_advection_1d", "base", "1D_Advection_Sols_beta0.4.hdf5", "base"),
    SourceSpec("pdebench_advection_1d", "parameter_holdout", "1D_Advection_Sols_beta1.0.hdf5", "parameter_holdout"),
    SourceSpec("pdebench_darcy_2d", "base", "2D_DarcyFlow_beta1.0_Train.hdf5", "base"),
    SourceSpec("pdebench_darcy_2d", "parameter_holdout", "2D_DarcyFlow_beta10.0_Train.hdf5", "parameter_holdout"),
    SourceSpec("pdebench_shallow_water_2d", "base", "2D_rdb_NA_NA.h5", "base"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert downloaded PDEBench HDF5/H5 files into OVHA public benchmark NPZ caches."
    )
    parser.add_argument("--raw-root", default="data/raw_public/pdebench")
    parser.add_argument("--cache-root", default="data/public_benchmark_cache")
    parser.add_argument("--families", nargs="*", default=[spec.family for spec in SOURCES if spec.role == "base"])
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--holdout-samples", type=int, default=1024)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--spatial-stride-1d", type=int, default=2)
    parser.add_argument("--spatial-stride-2d", type=int, default=2)
    parser.add_argument("--write-resolution-transfer", action="store_true", default=True)
    parser.add_argument("--no-resolution-transfer", dest="write_resolution_transfer", action="store_false")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    cache_root = Path(args.cache_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw root does not exist: {raw_root}")
    _require_h5py()

    selected_families = set(args.families)
    specs = [spec for spec in SOURCES if spec.family in selected_families]
    if args.inspect_only:
        inspect_sources(raw_root, specs)
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"raw_root": str(raw_root), "cache_root": str(cache_root), "families": {}}
    for family in sorted(selected_families):
        family_specs = [spec for spec in specs if spec.family == family]
        family_manifest = prepare_family(raw_root, cache_root, family, family_specs, args)
        manifest["families"][family] = family_manifest

    manifest_path = cache_root / "pdebench_public_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


def inspect_sources(raw_root: Path, specs: Iterable[SourceSpec]) -> None:
    import h5py

    for spec in specs:
        path = raw_root / spec.filename
        print(f"\n== {spec.family} {spec.role}: {path} ==")
        if not path.exists():
            print("missing")
            continue
        with h5py.File(path, "r") as handle:
            group_names = sample_group_names(handle)
            if group_names:
                print(f"sample_groups {len(group_names)}; showing first 3 groups")
                for group_name in group_names[:3]:
                    print(f"  [{group_name}]")
                    for name, obj in iter_group_datasets(handle[group_name]):
                        print(f"    {name} {tuple(obj.shape)} {obj.dtype}")
                continue

            def visit(name: str, obj: Any) -> None:
                if hasattr(obj, "shape"):
                    print(name, tuple(obj.shape), obj.dtype)

            handle.visititems(visit)


def prepare_family(raw_root: Path, cache_root: Path, family: str, specs: list[SourceSpec], args: Any) -> dict[str, Any]:
    family_root = cache_root / family
    family_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    base = next((spec for spec in specs if spec.role == "base"), None)
    if base is None:
        raise ValueError(f"missing base source for {family}")

    base_path = raw_root / base.filename
    base_samples = read_source(
        base_path,
        family=family,
        sample_count=args.max_samples,
        seed=args.seed,
        spatial_stride=_base_stride(family, args),
    )
    train, iid = split_train_iid(base_samples, train_fraction=args.train_fraction)
    write_cache(family_root / "train.npz", train, overwrite=args.overwrite)
    write_cache(family_root / "iid.npz", iid, overwrite=args.overwrite)
    manifest["train"] = sample_manifest(base, train)
    manifest["iid"] = sample_manifest(base, iid)

    if args.write_resolution_transfer:
        resolution = read_source(
            base_path,
            family=family,
            sample_count=args.holdout_samples,
            seed=args.seed + 1009,
            spatial_stride=1,
        )
        write_cache(family_root / "resolution_transfer.npz", resolution, overwrite=args.overwrite)
        manifest["resolution_transfer"] = sample_manifest(base, resolution)

    for holdout in [spec for spec in specs if spec.role == "parameter_holdout"]:
        holdout_samples = read_source(
            raw_root / holdout.filename,
            family=family,
            sample_count=args.holdout_samples,
            seed=args.seed + 2017,
            spatial_stride=_base_stride(family, args),
        )
        write_cache(family_root / "parameter_holdout.npz", holdout_samples, overwrite=args.overwrite)
        manifest["parameter_holdout"] = sample_manifest(holdout, holdout_samples)

    return manifest


def read_source(path: Path, *, family: str, sample_count: int, seed: int, spatial_stride: int) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing PDEBench source file: {path}")
    import h5py

    with h5py.File(path, "r") as handle:
        total = infer_sample_count(handle)
        indices = choose_indices(total, sample_count, seed)
        raw = read_operator_pair(handle, family=family, indices=indices, spatial_stride=spatial_stride)
    return normalize_cache_arrays(raw)


def infer_sample_count(handle: Any) -> int:
    groups = sample_group_names(handle)
    if groups:
        return len(groups)
    candidates = []
    handle.visititems(lambda _name, obj: candidates.append(int(obj.shape[0])) if hasattr(obj, "shape") and obj.shape else None)
    if not candidates:
        raise ValueError("HDF5 file does not contain any array datasets")
    return max(set(candidates), key=candidates.count)


def choose_indices(total: int, requested: int, seed: int) -> np.ndarray:
    count = min(total, requested)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False))


def read_operator_pair(handle: Any, *, family: str, indices: np.ndarray, spatial_stride: int) -> dict[str, np.ndarray]:
    if sample_group_names(handle):
        return read_grouped_time_dependent_pair(handle, indices=indices, spatial_stride=spatial_stride)
    if family == "pdebench_darcy_2d":
        input_key = first_existing(handle, ("nu", "coeff", "coefficient", "input", "input_field", "a"))
        output_key = first_existing(handle, ("tensor", "solution", "output", "output_field", "u"))
        if input_key is not None and output_key is not None and input_key != output_key:
            input_field = apply_spatial_stride(canonicalize_field_layout(read_dataset(handle[input_key], indices)), spatial_stride)
            output_field = apply_spatial_stride(canonicalize_field_layout(read_dataset(handle[output_key], indices)), spatial_stride)
            coordinates = coordinates_for_shape(handle, input_field.shape[1:-1] if input_field.ndim >= 4 else input_field.shape[1:], spatial_stride)
            return {"input_field": input_field, "output_field": output_field, "coordinates": coordinates}
    return read_time_dependent_pair(handle, indices=indices, spatial_stride=spatial_stride)


def read_grouped_time_dependent_pair(handle: Any, *, indices: np.ndarray, spatial_stride: int) -> dict[str, np.ndarray]:
    groups = sample_group_names(handle)
    if not groups:
        raise ValueError("grouped reader requires top-level sample groups")
    selected_groups = [handle[groups[int(index)]] for index in indices]
    dataset_path = infer_time_dataset_path(selected_groups[0])
    input_rows = []
    output_rows = []
    for group in selected_groups:
        data = np.asarray(group[dataset_path][()], dtype=np.float32)
        input_rows.append(slice_sample_time(data, time_index=0))
        output_rows.append(slice_sample_time(data, time_index=-1))
    input_field = canonicalize_field_layout(np.stack(input_rows, axis=0))
    output_field = canonicalize_field_layout(np.stack(output_rows, axis=0))
    input_field = apply_spatial_stride(input_field, spatial_stride)
    output_field = apply_spatial_stride(output_field, spatial_stride)
    spatial_shape = input_field.shape[1:-1] if input_field.ndim >= 4 else input_field.shape[1:]
    coordinates = coordinates_for_shape(handle, spatial_shape, spatial_stride)
    return {"input_field": input_field, "output_field": output_field, "coordinates": coordinates}


def read_time_dependent_pair(handle: Any, *, indices: np.ndarray, spatial_stride: int) -> dict[str, np.ndarray]:
    tensor_key = first_existing(handle, ("tensor", "data", "u", "solution"))
    if tensor_key is None:
        raise KeyError(f"cannot find time-dependent tensor key; available top-level keys: {sorted(handle.keys())}")
    tensor = handle[tensor_key]
    if tensor.ndim < 3:
        raise ValueError(f"time-dependent tensor must be at least [sample,time,space], got {tuple(tensor.shape)}")
    input_field = read_time_slice(tensor, indices, time_index=0)
    output_field = read_time_slice(tensor, indices, time_index=-1)
    input_field = canonicalize_field_layout(input_field)
    output_field = canonicalize_field_layout(output_field)
    input_field = apply_spatial_stride(input_field, spatial_stride)
    output_field = apply_spatial_stride(output_field, spatial_stride)
    coordinates = coordinates_for_shape(handle, input_field.shape[1:], spatial_stride)
    return {"input_field": input_field, "output_field": output_field, "coordinates": coordinates}


def sample_group_names(handle: Any) -> list[str]:
    import h5py

    names = [name for name, obj in handle.items() if isinstance(obj, h5py.Group)]
    numeric = [name for name in names if name.isdigit()]
    selected = numeric if numeric else names
    return sorted(selected)


def iter_group_datasets(group: Any) -> Iterable[tuple[str, Any]]:
    datasets = []

    def visit(name: str, obj: Any) -> None:
        if hasattr(obj, "shape"):
            datasets.append((name, obj))

    group.visititems(visit)
    return datasets


def infer_time_dataset_path(group: Any) -> str:
    candidates = []
    for name, obj in iter_group_datasets(group):
        if len(obj.shape) < 3:
            continue
        if any(part.lower() in {"x", "y", "grid", "coord", "coords", "coordinates"} for part in name.split("/")):
            continue
        size = int(np.prod(obj.shape))
        score = size
        lowered = name.lower()
        if any(token in lowered for token in ("tensor", "data", "solution", "u", "height", "water")):
            score += size
        candidates.append((score, name))
    if not candidates:
        available = [f"{name}{tuple(obj.shape)}" for name, obj in iter_group_datasets(group)]
        raise KeyError(f"cannot infer grouped time dataset; available datasets: {available}")
    return max(candidates)[1]


def read_dataset(dataset: Any, indices: np.ndarray) -> np.ndarray:
    return np.asarray(dataset[indices], dtype=np.float32)


def read_time_slice(dataset: Any, indices: np.ndarray, time_index: int) -> np.ndarray:
    if dataset.ndim == 3:
        return np.asarray(dataset[indices, time_index, :], dtype=np.float32)
    if dataset.ndim == 4:
        return np.asarray(dataset[indices, time_index, :, :], dtype=np.float32)
    if dataset.ndim == 5:
        return np.asarray(dataset[indices, time_index, :, :, :], dtype=np.float32)
    raise ValueError(f"unsupported time-dependent tensor shape: {tuple(dataset.shape)}")


def slice_sample_time(array: np.ndarray, time_index: int) -> np.ndarray:
    if array.ndim == 3:
        return array[time_index, :, :]
    if array.ndim == 4:
        if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            return array[:, time_index, :, :]
        return array[time_index, ...]
    if array.ndim == 5:
        return array[time_index, ...]
    raise ValueError(f"unsupported grouped sample tensor shape: {array.shape}")


def canonicalize_field_layout(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[1] <= 8 and array.shape[2] > array.shape[1]:
        return np.moveaxis(array, 1, -1)
    if array.ndim == 4 and array.shape[1] <= 8 and array.shape[2] > array.shape[1] and array.shape[3] > array.shape[1]:
        return np.moveaxis(array, 1, -1)
    return array


def apply_spatial_stride(array: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return array
    if array.ndim == 2:
        return array[:, ::stride]
    if array.ndim == 3:
        return array[:, ::stride, ::stride]
    if array.ndim == 4:
        return array[:, ::stride, ::stride, :]
    raise ValueError(f"unsupported array shape for spatial stride: {array.shape}")


def coordinates_for_shape(handle: Any, spatial_shape: tuple[int, ...], spatial_stride: int) -> np.ndarray:
    x_key = first_existing(handle, ("x-coordinate", "x_coordinate", "x", "grid_x"))
    y_key = first_existing(handle, ("y-coordinate", "y_coordinate", "y", "grid_y"))
    if len(spatial_shape) == 1:
        x = np.asarray(handle[x_key][()], dtype=np.float32) if x_key else np.linspace(0.0, 1.0, spatial_shape[0], dtype=np.float32)
        x = x[::spatial_stride] if x.shape[0] != spatial_shape[0] else x
        return x.reshape(-1, 1)
    if len(spatial_shape) >= 2:
        x = np.asarray(handle[x_key][()], dtype=np.float32) if x_key else np.linspace(0.0, 1.0, spatial_shape[0], dtype=np.float32)
        y = np.asarray(handle[y_key][()], dtype=np.float32) if y_key else np.linspace(0.0, 1.0, spatial_shape[1], dtype=np.float32)
        if x.shape[0] != spatial_shape[0]:
            x = x[::spatial_stride]
        if y.shape[0] != spatial_shape[1]:
            y = y[::spatial_stride]
        mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
        return np.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=-1).astype(np.float32)
    raise ValueError(f"cannot infer coordinates for spatial shape: {spatial_shape}")


def normalize_cache_arrays(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_field = field_to_points(raw["input_field"])
    output_field = field_to_points(raw["output_field"])
    coordinates = np.asarray(raw["coordinates"], dtype=np.float32)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(-1, 1)
    if input_field.shape[:2] != output_field.shape[:2]:
        raise ValueError(f"input/output shape mismatch: {input_field.shape} vs {output_field.shape}")
    if coordinates.shape[0] != input_field.shape[1]:
        raise ValueError(f"coordinates length {coordinates.shape[0]} != field points {input_field.shape[1]}")
    return {
        "input_field": input_field.astype(np.float32, copy=False),
        "output_field": output_field.astype(np.float32, copy=False),
        "coordinates": coordinates.astype(np.float32, copy=False),
    }


def field_to_points(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 2:
        return value[:, :, None]
    if value.ndim == 3:
        return value.reshape(value.shape[0], value.shape[1] * value.shape[2], 1)
    if value.ndim == 4:
        return value.reshape(value.shape[0], value.shape[1] * value.shape[2], value.shape[3])
    raise ValueError(f"unsupported field shape: {value.shape}")


def split_train_iid(samples: dict[str, np.ndarray], *, train_fraction: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    total = int(samples["input_field"].shape[0])
    train_count = max(1, min(total - 1, int(total * train_fraction)))
    train = {key: value[:train_count] if key != "coordinates" else value for key, value in samples.items()}
    iid = {key: value[train_count:] if key != "coordinates" else value for key, value in samples.items()}
    return train, iid


def write_cache(path: Path, samples: dict[str, np.ndarray], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"cache already exists, pass --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **samples)
    print(f"{path}")
    for key, value in samples.items():
        print(f"  {key:12s} {value.shape}")


def sample_manifest(source: SourceSpec, samples: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "source_file": source.filename,
        "source_role": source.role,
        "input_shape": list(samples["input_field"].shape),
        "output_shape": list(samples["output_field"].shape),
        "coordinates_shape": list(samples["coordinates"].shape),
    }


def first_existing(handle: Any, names: Iterable[str]) -> str | None:
    for name in names:
        if name in handle:
            return name
    return None


def _base_stride(family: str, args: Any) -> int:
    return args.spatial_stride_1d if family.endswith("_1d") else args.spatial_stride_2d


def _require_h5py() -> None:
    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("h5py is required; install it with: python -m pip install h5py") from exc


if __name__ == "__main__":
    main()
