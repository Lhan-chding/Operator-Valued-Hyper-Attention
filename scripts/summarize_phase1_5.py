#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env_path = output_dir / "environment.json"
    metrics_path = output_dir / "eval_metrics.jsonl"
    report_path = output_dir / "phase1_5_report.md"
    env = json.loads(env_path.read_text()) if env_path.exists() else {"torch_available": False}
    lines = ["# Phase 1.5 Report", "", "## Environment", "", f"- torch_available: {env.get('torch_available')}", f"- device: {env.get('device')}"]

    if not env.get("torch_available"):
        lines.extend(["", "## Status", "", "Torch is not installed; torch implementation tests skipped."])
        report_path.write_text("\n".join(lines) + "\n")
        print(report_path)
        return report_path

    if not metrics_path.exists():
        lines.extend(
            [
                "",
                "## Status",
                "",
                f"Evaluation metrics were not found at `{metrics_path.name}`.",
                "The run is incomplete; inspect the preceding training/evaluation traceback before using this report.",
            ]
        )
        report_path.write_text("\n".join(lines) + "\n")
        print(report_path)
        return report_path

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["split"], row["family"])].append(row)
    lines.extend(["", "## Aggregate Metrics", "", "| model | split | family | mean relL2 | median relL2 | p90 relL2 | entropy |", "|---|---|---|---:|---:|---:|---:|"])
    by_model = defaultdict(list)
    for (model, split, family), items in sorted(grouped.items()):
        rel = mean(float(item["relative_l2"]) for item in items)
        med = mean(float(item["median_relative_l2"]) for item in items)
        p90 = mean(float(item["p90_relative_l2"]) for item in items)
        ent = mean(float(item["primitive_entropy"]) for item in items)
        by_model[model].append(rel)
        lines.append(f"| {model} | {split} | {family} | {rel:.6f} | {med:.6f} | {p90:.6f} | {ent:.6f} |")
    ovha = mean(by_model.get("ovha_full", [float("inf")]))
    simple = mean(by_model.get("simple_stack", [float("inf")]))
    vector = mean(by_model.get("transformer_only", [float("inf")]))
    lines.extend(["", "## Go / No-Go", ""])
    if ovha < simple and ovha < vector:
        lines.append("Provisional Go for local smoke: OVHA-full beats simple_stack and vector attention in this run.")
    else:
        lines.append("No-Go for local smoke: OVHA-full did not beat required comparators in this run.")
    lines.extend(
        [
            "",
            "## Metadata-Free Checklist",
            "",
            "- Model inputs use only context_u/context_q/context_y/target_u/target_q/support_grid/masks.",
            "- Hidden family, latent params, mixture weights and oracle hints are offline diagnostics only.",
            "- oracle_metadata_upper_bound is labeled separately and excluded from the main claim.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    print(report_path)
    return report_path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 scripts/summarize_phase1_5.py <output_dir>")
    summarize(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
