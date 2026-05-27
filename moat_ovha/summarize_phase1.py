from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Union


def summarize_metrics(metrics_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
    rows = [json.loads(line) for line in Path(metrics_path).read_text().splitlines() if line.strip()]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["family"]))].append(row)

    lines = [
        "# Phase-1 OVHA Report",
        "",
        "## Aggregate Metrics",
        "",
        "| model | family | mean relative L2 | mean MSE | mean entropy |",
        "|---|---:|---:|---:|---:|",
    ]
    aggregate_by_model: dict[str, list[float]] = defaultdict(list)
    for (model, family), items in sorted(grouped.items()):
        rel = mean(float(item["relative_l2"]) for item in items)
        err = mean(float(item["mse"]) for item in items)
        ent = mean(float(item["expert_entropy"]) for item in items)
        aggregate_by_model[model].append(rel)
        lines.append(f"| {model} | {family} | {rel:.6f} | {err:.6f} | {ent:.6f} |")

    lines.extend(["", "## Go / No-Go", ""])
    ovha_score = mean(aggregate_by_model.get("ovha_full", [float("inf")]))
    simple_score = mean(aggregate_by_model.get("simple_stack", [float("inf")]))
    vector_score = mean(aggregate_by_model.get("vector_value", [float("inf")]))
    if ovha_score < simple_score and ovha_score < vector_score:
        conclusion = "Go: OVHA-full beats simple-stack and vector-value attention on the configured sweep."
    else:
        conclusion = "No-Go: current configured sweep does not yet establish a robust ablation gap."
    lines.append(conclusion)
    lines.extend(
        [
            "",
            "## Required Phase-2 Questions",
            "",
            "- Expand beyond deterministic toy operators with learned training loops once torch/numpy are available.",
            "- Stress-test primitive specialization under metadata-free contexts.",
            "- Add higher-resolution transfer and operator-family holdout curves.",
        ]
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize phase-1 JSONL metrics into a Markdown report.")
    parser.add_argument("--metrics", required=True, help="Path to eval_metrics.jsonl or train_metrics.jsonl.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    args = parser.parse_args()
    print(summarize_metrics(args.metrics, args.output))


if __name__ == "__main__":
    main()
