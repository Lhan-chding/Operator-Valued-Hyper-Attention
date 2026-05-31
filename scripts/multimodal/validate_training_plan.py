#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.train.multimodal_protocol import validate_training_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a multimodal OVHA training/loss protocol plan.")
    parser.add_argument("plan_json", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan_json.read_text())
    report = validate_training_protocol(_protocol_plan(plan))
    print(json.dumps({"ok": report.ok, "errors": report.errors, "warnings": report.warnings}, indent=2, sort_keys=True))
    return 0 if report.ok else 2


def _protocol_plan(plan: dict[str, object]) -> dict[str, object]:
    normalized = dict(plan)
    corruptions = normalized.get("robustness_corruptions")
    if isinstance(corruptions, list) and corruptions:
        normalized["task_type"] = "robustness_eval"
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
