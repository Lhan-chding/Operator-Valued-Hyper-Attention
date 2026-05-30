#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.data.multimodal.validation import validate_multimodal_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a multimodal OVHA cache contract.")
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("dataset_name")
    parser.add_argument("version")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    report = validate_multimodal_cache(args.cache_root, args.dataset_name, args.version, tuple(args.splits))
    print(json.dumps({"ok": report.ok, "errors": report.errors, "warnings": report.warnings}, indent=2, sort_keys=True))
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
