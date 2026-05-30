#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize controlled multimodal oracle/gate JSONL rows.")
    parser.add_argument("controlled_jsonl", type=Path)
    args = parser.parse_args()
    rows = []
    with args.controlled_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    print(json.dumps(build_controlled_report(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
