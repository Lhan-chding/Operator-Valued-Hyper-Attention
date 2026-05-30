#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.eval.multimodal_diagnostics import summarize_diagnostic_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize multimodal OVHA diagnostics JSONL.")
    parser.add_argument("diagnostics_jsonl", type=Path)
    args = parser.parse_args()
    rows = []
    with args.diagnostics_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    print(json.dumps(summarize_diagnostic_rows(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
