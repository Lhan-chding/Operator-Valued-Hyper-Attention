#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def summarize(root: Path, output: Path | None = None) -> Path:
    eval_rows = _read_jsonl_paths(_find_eval_paths(root))
    diagnostic_rows = _read_jsonl_paths(_find_diagnostic_paths(root))
    output_path = output or root / "public_adapter_diagnostics.md"
    lines = _build_report(eval_rows, diagnostic_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(output_path)
    return output_path


def _build_report(eval_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]]) -> list[str]:
    return [
        "# Public Adapter Diagnostics",
        "",
        "## Adapter Effect By Dataset",
        "",
        "| dataset | split | ovha_full | ovha_no_hyper_adapter | adapter_delta | conclusion | best_model | best_relL2 |",
        "|---|---|---:|---:|---:|---|---|---:|",
        *_adapter_effect_rows(eval_rows),
        "",
        "## Full OVHA Diagnostic Signals",
        "",
        "| dataset | split | model | adapter_norm | router_query_residual_norm | router_context_entropy | primitive_entropy | q_variance |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        *_diagnostic_signal_rows(diagnostic_rows),
        "",
        "## Interpretation Guide",
        "",
        "- `adapter_delta = ovha_no_hyper_adapter - ovha_full`; positive means the hyper-adapter helped.",
        "- `adapter_hurts` means the primitive stack did better when the hyper-adapter was disabled.",
        "- High `router_query_residual_norm` or `q_variance` on a hurt case is evidence for over-conditioned routing/adapter behavior.",
    ]


def _adapter_effect_rows(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        dataset = row.get("dataset") or row.get("family")
        split = row.get("split")
        model = row.get("model_name") or row.get("model")
        if dataset and split and model and row.get("relative_l2") is not None:
            grouped[(str(dataset), str(split), str(model))].append(float(row["relative_l2"]))

    lines = []
    dataset_splits = sorted({(dataset, split) for dataset, split, _ in grouped})
    for dataset, split in dataset_splits:
        values = {
            model: mean(items)
            for (row_dataset, row_split, model), items in grouped.items()
            if row_dataset == dataset and row_split == split
        }
        full = values.get("ovha_full")
        no_adapter = values.get("ovha_no_hyper_adapter")
        delta = None if full is None or no_adapter is None else no_adapter - full
        best_model, best_value = min(values.items(), key=lambda item: item[1])
        lines.append(
            f"| {dataset} | {split} | {_fmt(full)} | {_fmt(no_adapter)} | {_fmt(delta)} | "
            f"{_adapter_conclusion(delta)} | {best_model} | {_fmt(best_value)} |"
        )
    return lines or ["| not_available | not_available |  |  |  | no eval rows found |  |  |"]


def _diagnostic_signal_rows(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset = row.get("dataset") or row.get("family")
        split = row.get("split")
        model = row.get("model_name") or row.get("model")
        if dataset and split and model:
            grouped[(str(dataset), str(split), str(model))].append(row)

    lines = []
    for (dataset, split, model), items in sorted(grouped.items()):
        if model != "ovha_full":
            continue
        lines.append(
            f"| {dataset} | {split} | {model} | {_fmt(_mean_adapter_norm(items))} | "
            f"{_fmt(_mean_field(items, 'router_query_residual_norm'))} | "
            f"{_fmt(_mean_field(items, 'router_context_prior_entropy'))} | "
            f"{_fmt(_mean_field(items, 'primitive_entropy'))} | {_fmt(_mean_q_variance(items))} |"
        )
    return lines or ["| not_available | not_available | not_available |  |  |  |  |  |"]


def _find_eval_paths(root: Path) -> list[Path]:
    paths = sorted((root / "eval_metrics").glob("*/*.jsonl"))
    paths.extend(sorted(root.glob("*/eval_metrics/*/*.jsonl")))
    if (root / "eval_metrics.jsonl").exists():
        paths.append(root / "eval_metrics.jsonl")
    return _unique_paths(paths)


def _find_diagnostic_paths(root: Path) -> list[Path]:
    paths = sorted((root / "diagnostics").glob("*/*.jsonl"))
    paths.extend(sorted(root.glob("*/diagnostics/*/*.jsonl")))
    if (root / "diagnostics.jsonl").exists():
        paths.append(root / "diagnostics.jsonl")
    return _unique_paths(paths)


def _read_jsonl_paths(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    unique = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _adapter_conclusion(delta: float | None) -> str:
    if delta is None:
        return "missing_comparison"
    if delta > 1e-12:
        return "adapter_helps"
    if delta < -1e-12:
        return "adapter_hurts"
    return "adapter_neutral"


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def _mean_adapter_norm(rows: list[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        adapter_norms = row.get("adapter_norms") or {}
        if adapter_norms:
            values.append(mean(float(value) for value in adapter_norms.values()))
    return mean(values) if values else None


def _mean_q_variance(rows: list[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        adapter_stats = row.get("adapter_stats") or {}
        for stats in adapter_stats.values():
            for key, value in stats.items():
                if key.startswith("q_variance") and value is not None:
                    values.append(float(value))
    return mean(values) if values else None


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize public benchmark adapter/router diagnostic signals.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summarize(args.root, args.output)


if __name__ == "__main__":
    main()
