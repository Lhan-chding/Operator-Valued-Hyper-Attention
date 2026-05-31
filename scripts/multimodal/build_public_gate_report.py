#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.data.multimodal.cache_schema import file_sha256
from moat_ovha_torch.eval.multimodal_public_gates import evaluate_region_text_gate, evaluate_sentiment_gate
from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows
from moat_ovha_torch.eval.multimodal_statistics import summarize_public_results, validate_public_summary
from scripts.multimodal.evaluate_public_gates import (
    _evidence_artifacts,
    _robustness_rows_consistency_errors,
    _topconf_evidence_errors,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a top-conference public gate evidence bundle from raw metrics, "
            "diagnostics, and robustness rows."
        )
    )
    parser.add_argument("gate", choices=("region_text", "sentiment"))
    parser.add_argument("--raw-metrics", type=Path, action="append", required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--robustness-rows", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-model", default="ovha_full")
    parser.add_argument("--baseline-model", default="cross_attention_transformer")
    parser.add_argument("--no-cato-score", type=float)
    parser.add_argument("--no-lrio-score", type=float)
    parser.add_argument("--no-spo-score", type=float)
    parser.add_argument("--no-rceo-score", type=float)
    args = parser.parse_args()

    try:
        payload, exit_code = build_public_gate_report(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "ok": False,
            "mode": "public_gate_evidence_bundle",
            "policy": "fail-fast: public gate bundle inputs must be readable and internally consistent",
            "errors": [str(exc)],
            "warnings": [],
        }
        exit_code = 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def build_public_gate_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _read_raw_metric_rows(args.raw_metrics)
    diagnostics_rows = _read_jsonl(args.diagnostics)
    robustness_rows = _read_jsonl(args.robustness_rows)
    statistics = summarize_public_results(raw_rows, full_model=args.full_model, baseline_model=args.baseline_model)
    statistics_validation = validate_public_summary(statistics)
    statistics_path = output_dir / f"{args.gate}_statistics_summary.json"
    robustness_summary_path = output_dir / f"{args.gate}_robustness_summary.json"
    gate_report_path = output_dir / f"{args.gate}_gate_report.json"

    statistics_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n")
    robustness_summary = summarize_robustness_rows(
        robustness_rows,
        full_model=args.full_model,
        baseline_model=args.baseline_model,
    )
    robustness_summary = {
        **robustness_summary,
        "task": args.task,
        "source_rows_path": str(args.robustness_rows),
    }
    robustness_summary_path.write_text(json.dumps(robustness_summary, indent=2, sort_keys=True) + "\n")

    if args.gate == "region_text":
        gate_report = evaluate_region_text_gate(
            statistics_summary=statistics,
            diagnostics_rows=diagnostics_rows,
            no_cato_score=_score_or_main_table(args.no_cato_score, statistics, args.task, args.split, "ovha_no_cato"),
            robustness_summary=robustness_summary,
            task=args.task,
            split=args.split,
            full_model=args.full_model,
            baseline_model=args.baseline_model,
        )
    else:
        gate_report = evaluate_sentiment_gate(
            statistics_summary=statistics,
            diagnostics_rows=diagnostics_rows,
            ablation_scores={
                "ovha_no_lrio": _score_or_main_table(args.no_lrio_score, statistics, args.task, args.split, "ovha_no_lrio"),
                "ovha_no_spo": _score_or_main_table(args.no_spo_score, statistics, args.task, args.split, "ovha_no_spo"),
                "ovha_no_rceo": _score_or_main_table(args.no_rceo_score, statistics, args.task, args.split, "ovha_no_rceo"),
            },
            robustness_summary=robustness_summary,
            task=args.task,
            split=args.split,
            full_model=args.full_model,
            baseline_model=args.baseline_model,
        )

    evidence = _evidence_artifacts(
        statistics,
        statistics_path=statistics_path,
        diagnostics_path=args.diagnostics,
        robustness_path=robustness_summary_path,
        robustness_rows_path=args.robustness_rows,
        task=args.task,
        split=args.split,
    )
    gate_report["evidence_artifacts"] = evidence
    evidence_errors = _topconf_evidence_errors(evidence)
    evidence_errors.extend(_robustness_rows_consistency_errors(robustness_summary_path, args.robustness_rows))
    if gate_report["passed"] and (not statistics_validation.ok or evidence_errors):
        gate_report["passed"] = False
        gate_report["reasons"] = [
            *gate_report.get("reasons", []),
            *statistics_validation.errors,
            *evidence_errors,
        ]
    gate_report_path.write_text(json.dumps(gate_report, indent=2, sort_keys=True) + "\n")

    ok = bool(gate_report["passed"] and statistics_validation.ok and not evidence_errors)
    payload = {
        "ok": ok,
        "mode": "public_gate_evidence_bundle",
        "policy": "raw metrics, diagnostics, robustness, statistics, and gate report are bundled for top-conference entry validation",
        "gate": args.gate,
        "task": args.task,
        "split": args.split,
        "artifacts": {
            "statistics_summary": _artifact_descriptor(statistics_path),
            "diagnostics": _artifact_descriptor(args.diagnostics),
            "robustness_summary": _artifact_descriptor(robustness_summary_path),
            "robustness_rows": _artifact_descriptor(args.robustness_rows),
            "gate_report": _artifact_descriptor(gate_report_path),
            "raw_metrics": [_artifact_descriptor(path) for path in args.raw_metrics],
        },
        "validation": {
            "statistics": {
                "ok": statistics_validation.ok,
                "errors": statistics_validation.errors,
                "warnings": statistics_validation.warnings,
            },
            "gate_passed": bool(gate_report["passed"]),
            "gate_reasons": gate_report.get("reasons", []),
            "evidence_errors": evidence_errors,
        },
    }
    return payload, 0 if ok else 2


def _read_raw_metric_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        source_path = path.resolve()
        for row in _read_jsonl(path):
            raw_metric_path = str(row.get("raw_metric_path", "")).strip()
            resolved_raw_metric_path = Path(raw_metric_path).resolve() if raw_metric_path else source_path
            row = {**row, "raw_metric_path": str(resolved_raw_metric_path)}
            rows.append(row)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} must contain JSONL object rows")
                rows.append(payload)
    if not rows:
        raise ValueError(f"{path} must contain at least one JSONL row")
    return rows


def _score_or_main_table(
    supplied: float | None,
    statistics: dict[str, Any],
    task: str,
    split: str,
    model: str,
) -> float | None:
    if supplied is not None:
        return supplied
    main_table = statistics.get("main_table")
    task_table = main_table.get(task) if isinstance(main_table, dict) else None
    split_table = task_table.get(split) if isinstance(task_table, dict) else None
    model_row = split_table.get(model) if isinstance(split_table, dict) else None
    if isinstance(model_row, dict) and model_row.get("mean") is not None:
        return float(model_row["mean"])
    return None


def _artifact_descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


if __name__ == "__main__":
    raise SystemExit(main())
