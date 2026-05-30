#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize multimodal robustness JSONL rows.")
    parser.add_argument("robustness_jsonl", type=Path)
    parser.add_argument("--full-model", default="ovha_full")
    parser.add_argument("--baseline-model", required=True)
    args = parser.parse_args()
    rows = []
    with args.robustness_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    print(
        json.dumps(
            summarize_robustness_rows(rows, full_model=args.full_model, baseline_model=args.baseline_model),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
