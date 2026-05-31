#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate, evaluate_sentiment_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multimodal public Go/No-Go gates.")
    parser.add_argument("gate", choices=("region_text", "sentiment"))
    parser.add_argument("statistics_summary", type=Path)
    parser.add_argument("diagnostics_jsonl", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--no-cato-score", type=float)
    parser.add_argument("--no-lrio-score", type=float)
    parser.add_argument("--no-spo-score", type=float)
    parser.add_argument("--no-rceo-score", type=float)
    parser.add_argument("--robustness-summary", type=Path)
    args = parser.parse_args()

    statistics = json.loads(args.statistics_summary.read_text())
    diagnostics = _read_jsonl(args.diagnostics_jsonl)
    if args.gate == "region_text":
        robustness = json.loads(args.robustness_summary.read_text()) if args.robustness_summary else {}
        report = evaluate_region_text_gate(
            statistics_summary=statistics,
            diagnostics_rows=diagnostics,
            no_cato_score=args.no_cato_score,
            robustness_summary=robustness,
            task=args.task,
            split=args.split,
        )
    else:
        robustness = json.loads(args.robustness_summary.read_text()) if args.robustness_summary else {}
        report = evaluate_sentiment_gate(
            statistics_summary=statistics,
            diagnostics_rows=diagnostics,
            ablation_scores={
                "ovha_no_lrio": args.no_lrio_score,
                "ovha_no_spo": args.no_spo_score,
                "ovha_no_rceo": args.no_rceo_score,
            },
            robustness_summary=robustness,
            task=args.task,
            split=args.split,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
