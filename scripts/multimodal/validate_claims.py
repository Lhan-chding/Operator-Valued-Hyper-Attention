#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.eval.multimodal_claims import validate_claim_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OVHA paper claim boundaries against available evidence scope.")
    parser.add_argument("claim_file", type=Path)
    parser.add_argument("--scope", required=True, choices=("pdebench", "multimodal_main"))
    parser.add_argument("--evidence-json", type=Path)
    args = parser.parse_args()
    evidence = {}
    if args.evidence_json is not None:
        evidence = json.loads(args.evidence_json.read_text())
    report = validate_claim_text(args.claim_file.read_text(), evidence_scope=args.scope, evidence=evidence)
    print(json.dumps({"ok": report.ok, "errors": report.errors, "warnings": report.warnings}, indent=2, sort_keys=True))
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
