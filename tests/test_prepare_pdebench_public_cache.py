from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "prepare_pdebench_public_cache.py"
    spec = importlib.util.spec_from_file_location("prepare_pdebench_public_cache", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PreparePDEBenchPublicCacheTests(unittest.TestCase):
    def test_field_to_points_handles_1d_2d_and_channels(self):
        module = _load_module()

        one_d = np.zeros((3, 5), dtype=np.float32)
        two_d = np.zeros((3, 4, 5), dtype=np.float32)
        channeled = np.zeros((3, 4, 5, 2), dtype=np.float32)

        self.assertEqual(module.field_to_points(one_d).shape, (3, 5, 1))
        self.assertEqual(module.field_to_points(two_d).shape, (3, 20, 1))
        self.assertEqual(module.field_to_points(channeled).shape, (3, 20, 2))

    def test_split_train_iid_preserves_coordinates(self):
        module = _load_module()
        samples = {
            "input_field": np.zeros((10, 8, 1), dtype=np.float32),
            "output_field": np.zeros((10, 8, 1), dtype=np.float32),
            "coordinates": np.zeros((8, 1), dtype=np.float32),
        }

        train, iid = module.split_train_iid(samples, train_fraction=0.8)

        self.assertEqual(train["input_field"].shape[0], 8)
        self.assertEqual(iid["input_field"].shape[0], 2)
        self.assertIs(train["coordinates"], samples["coordinates"])
        self.assertIs(iid["coordinates"], samples["coordinates"])

    def test_canonicalize_field_layout_moves_channel_first_to_last(self):
        module = _load_module()
        one_d = np.zeros((3, 2, 8), dtype=np.float32)
        two_d = np.zeros((3, 2, 4, 5), dtype=np.float32)

        self.assertEqual(module.canonicalize_field_layout(one_d).shape, (3, 8, 2))
        self.assertEqual(module.canonicalize_field_layout(two_d).shape, (3, 4, 5, 2))

    def test_end_to_end_converts_small_burgers_cache_when_h5py_is_available(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is not installed")
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            cache_root = Path(tmp) / "cache"
            raw_root.mkdir()
            with h5py.File(raw_root / "1D_Burgers_Sols_Nu0.01.hdf5", "w") as handle:
                handle.create_dataset("tensor", data=np.arange(12 * 4 * 8, dtype=np.float32).reshape(12, 4, 8))
                handle.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, 8, dtype=np.float32))

            args = type(
                "Args",
                (),
                {
                    "max_samples": 10,
                    "holdout_samples": 4,
                    "train_fraction": 0.8,
                    "seed": 81,
                    "spatial_stride_1d": 2,
                    "spatial_stride_2d": 2,
                    "write_resolution_transfer": True,
                    "overwrite": False,
                },
            )()
            spec = module.SourceSpec("pdebench_burgers_1d", "base", "1D_Burgers_Sols_Nu0.01.hdf5", "base")

            module.prepare_family(raw_root, cache_root, "pdebench_burgers_1d", [spec], args)

            train = np.load(cache_root / "pdebench_burgers_1d" / "train.npz")
            iid = np.load(cache_root / "pdebench_burgers_1d" / "iid.npz")
            resolution = np.load(cache_root / "pdebench_burgers_1d" / "resolution_transfer.npz")
            self.assertEqual(train["input_field"].shape, (8, 4, 1))
            self.assertEqual(iid["input_field"].shape, (2, 4, 1))
            self.assertEqual(resolution["input_field"].shape, (4, 8, 1))


if __name__ == "__main__":
    unittest.main()
