from __future__ import annotations

from pathlib import Path
from typing import Any


class MissingBenchmarkDataError(FileNotFoundError):
    pass


class PublicBenchmarkLoader:
    dataset_label = "public benchmark"

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def split_path(self, split: str) -> Path:
        for suffix in (".npz", ".npy", ".h5", ".hdf5"):
            path = self.root / f"{split}{suffix}"
            if path.exists():
                return path
        raise MissingBenchmarkDataError(f"{self.dataset_label} subset not found at {self.root} for split {split!r}")

    def load_split(self, split: str) -> dict[str, Any]:
        path = self.split_path(split)
        if path.suffix == ".npz":
            return self._load_npz(path)
        if path.suffix == ".npy":
            return self._load_npy(path)
        if path.suffix in {".h5", ".hdf5"}:
            return self._load_hdf5(path)
        raise ValueError(f"unsupported benchmark file format: {path}")

    def _load_npz(self, path: Path) -> dict[str, Any]:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to load NPZ benchmark caches") from exc
        data = np.load(path)
        return {key: data[key] for key in data.files}

    def _load_npy(self, path: Path) -> dict[str, Any]:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to load NPY benchmark caches") from exc
        return {"array": np.load(path)}

    def _load_hdf5(self, path: Path) -> dict[str, Any]:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required to load HDF5 benchmark caches") from exc
        with h5py.File(path, "r") as handle:
            return {key: handle[key][()] for key in handle.keys()}
