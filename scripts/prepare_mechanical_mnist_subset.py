#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local Mechanical MNIST subset cache manifest for Phase 1.6.")
    parser.add_argument("--source", required=True, help="Directory containing Mechanical MNIST or mechanics-material files.")
    parser.add_argument("--output", default="data/phase1_6/mechanical_mnist_subset")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Mechanical MNIST source directory not found: {source}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(source),
        "expected_splits": ["train", "val", "test", "parameter_holdout"],
        "expected_datasets": ["mechanical_mnist_small"],
        "note": "Convert mechanics/material fields into metadata-free field-pair caches before GPU public pilot.",
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
