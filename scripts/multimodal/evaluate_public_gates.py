#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


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
    report["evidence_artifacts"] = _evidence_artifacts(
        statistics,
        statistics_path=args.statistics_summary,
        diagnostics_path=args.diagnostics_jsonl,
        robustness_path=args.robustness_summary,
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


def _evidence_artifacts(
    statistics: dict[str, Any],
    *,
    statistics_path: Path,
    diagnostics_path: Path,
    robustness_path: Path | None,
    task: str,
    split: str,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "task": task,
        "split": split,
        "generated_by": str(Path(__file__).relative_to(ROOT)),
        "statistics_summary": _artifact_descriptor(statistics_path),
        "diagnostics": _artifact_descriptor(diagnostics_path),
        "raw_metrics": [
            _artifact_descriptor(path)
            for path in _raw_metric_paths(statistics, base_dir=statistics_path.parent)
        ],
    }
    if robustness_path is not None:
        evidence["robustness_summary"] = _artifact_descriptor(robustness_path)
    return evidence


def _artifact_descriptor(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.exists() else "",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_metric_paths(statistics: dict[str, Any], *, base_dir: Path) -> list[Path]:
    raw_paths: list[Path] = []
    metadata = statistics.get("metadata")
    if isinstance(metadata, Mapping):
        raw_paths.extend(_path_list(metadata.get("raw_metric_paths"), base_dir=base_dir))
    raw_paths.extend(_row_metric_paths(statistics.get("per_seed_appendix"), base_dir=base_dir))
    reporting_metadata = statistics.get("reporting_metadata")
    if isinstance(reporting_metadata, Mapping):
        raw_paths.extend(_row_metric_paths(reporting_metadata.get("per_seed_table"), base_dir=base_dir))
    return sorted(set(raw_paths), key=lambda path: str(path))


def _path_list(value: Any, *, base_dir: Path) -> list[Path]:
    if not isinstance(value, list):
        return []
    return [_resolve_metric_path(path, base_dir=base_dir) for path in value if _non_empty_text(path)]


def _row_metric_paths(rows: Any, *, base_dir: Path) -> list[Path]:
    if not isinstance(rows, list):
        return []
    paths: list[Path] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw_path = row.get("raw_metric_path")
        if _non_empty_text(raw_path):
            paths.append(_resolve_metric_path(str(raw_path), base_dir=base_dir))
    return paths


def _resolve_metric_path(path: str, *, base_dir: Path) -> Path:
    metric_path = Path(path)
    return metric_path if metric_path.is_absolute() else base_dir / metric_path


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


if __name__ == "__main__":
    raise SystemExit(main())
