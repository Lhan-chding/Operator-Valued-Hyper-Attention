#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply CMU-MOSEI validation-only residual operator admission for TANSOBase-noRCEO."
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--candidate-residual", action="append", required=True)
    parser.add_argument("--base-composite", type=float, required=True)
    parser.add_argument("--candidate-composite", type=float, required=True)
    parser.add_argument("--base-mae", type=float, required=True)
    parser.add_argument("--candidate-mae", type=float, required=True)
    parser.add_argument("--base-acc7", type=float, required=True)
    parser.add_argument("--candidate-acc7", type=float, required=True)
    parser.add_argument("--composite-epsilon", type=float, default=0.001)
    parser.add_argument("--mae-tolerance", type=float, default=0.001)
    parser.add_argument("--acc7-tolerance", type=float, default=0.002)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.base_config.read_text())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_residuals = tuple(str(candidate) for candidate in args.candidate_residual)
    admitted = _admitted_residuals(
        candidate_residuals,
        base_composite=float(args.base_composite),
        candidate_composite=float(args.candidate_composite),
        base_mae=float(args.base_mae),
        candidate_mae=float(args.candidate_mae),
        base_acc7=float(args.base_acc7),
        candidate_acc7=float(args.candidate_acc7),
        composite_epsilon=float(args.composite_epsilon),
        mae_tolerance=float(args.mae_tolerance),
        acc7_tolerance=float(args.acc7_tolerance),
    )
    bank = {
        "artifact_type": "mosei_operator_admission_bank",
        "base_config": str(args.base_config),
        "config_name": str(config.get("name", "")),
        "base_model": "TANSOBase-noRCEO",
        "candidate_residuals": list(candidate_residuals),
        "admitted_residuals": list(admitted),
        "admission_criteria": {
            "candidate_composite_lt_base_minus_epsilon": float(args.composite_epsilon),
            "candidate_mae_lte_base_plus_tolerance": float(args.mae_tolerance),
            "candidate_acc7_gte_base_minus_tolerance": float(args.acc7_tolerance),
            "selection_split": "validation",
            "test_policy": "final_once_after_validation_admission",
        },
    }
    table_rows = [
        {
            "base_model": "TANSOBase-noRCEO",
            "candidate_residuals": "+".join(candidate_residuals),
            "base_composite": float(args.base_composite),
            "candidate_composite": float(args.candidate_composite),
            "base_mae": float(args.base_mae),
            "candidate_mae": float(args.candidate_mae),
            "base_acc7": float(args.base_acc7),
            "candidate_acc7": float(args.candidate_acc7),
            "admitted": bool(admitted),
        }
    ]

    (output_dir / "admitted_bank.json").write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n")
    _write_csv(output_dir / "base_vs_candidate_val_table.csv", table_rows)
    print(json.dumps({"ok": True, "artifacts": {"admitted_bank": str(output_dir / "admitted_bank.json"), "val_table": str(output_dir / "base_vs_candidate_val_table.csv")}}, sort_keys=True))
    return 0


def _admitted_residuals(
    candidate_residuals: tuple[str, ...],
    *,
    base_composite: float,
    candidate_composite: float,
    base_mae: float,
    candidate_mae: float,
    base_acc7: float,
    candidate_acc7: float,
    composite_epsilon: float,
    mae_tolerance: float,
    acc7_tolerance: float,
) -> tuple[str, ...]:
    admitted = (
        candidate_composite < base_composite - composite_epsilon
        and candidate_mae <= base_mae + mae_tolerance
        and candidate_acc7 >= base_acc7 - acc7_tolerance
    )
    return candidate_residuals if admitted else ()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
