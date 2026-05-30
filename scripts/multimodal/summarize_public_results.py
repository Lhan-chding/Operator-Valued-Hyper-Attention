#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize multimodal public multi-seed metrics for top-conference tables.")
    parser.add_argument("metrics_jsonl", type=Path)
    parser.add_argument("--full-model", required=True)
    parser.add_argument("--baseline-model", required=True)
    args = parser.parse_args()
    rows = []
    with args.metrics_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    summary = summarize_public_results(rows, full_model=args.full_model, baseline_model=args.baseline_model)
    validation = validate_public_summary(summary)
    payload = {
        **summary,
        "validation": {"ok": validation.ok, "errors": validation.errors, "warnings": validation.warnings},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if validation.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
