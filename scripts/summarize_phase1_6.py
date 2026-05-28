#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


CONTROLLED_STRESS_FAMILIES = {
    "single_primitive_representable",
    "query_piecewise_router",
    "context_identifiable_mixture",
    "hyper_parameter_family",
    "same_target_counterfactual",
    "modality_reliability_conflict",
    "history_session_preference",
    "query_piecewise_composition_family",
    "context_identifiable_mixture_family",
    "anti_single_primitive_family",
    "confounded_family_pair",
}
MEMORY_ABLATION_MODELS = (
    "ovha_zero_memory",
    "ovha_learned_global_memory",
    "ovha_no_memory",
    "target_only",
    "ovha_shuffled_context_memory",
)


def summarize(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    eval_paths = sorted((root / "eval_metrics").glob("*/*.jsonl"))
    if not eval_paths:
        eval_paths = sorted(root.glob("*/eval_metrics/*/*.jsonl"))
    eval_rows = _read_jsonl_paths(eval_paths)
    if not eval_rows and (root / "eval_metrics.jsonl").exists():
        eval_rows = _read_jsonl_paths([root / "eval_metrics.jsonl"])
    train_paths = sorted((root / "train_metrics").glob("*/*.jsonl"))
    if not train_paths:
        train_paths = sorted(root.glob("*/train_metrics/*/*.jsonl"))
    train_rows = _read_jsonl_paths(train_paths)
    diagnostic_paths = sorted((root / "diagnostics").glob("*/*.jsonl"))
    if not diagnostic_paths:
        diagnostic_paths = sorted(root.glob("*/diagnostics/*/*.jsonl"))
    diagnostic_rows = _read_jsonl_paths(diagnostic_paths)
    if not diagnostic_rows and (root / "diagnostics.jsonl").exists():
        diagnostic_rows = _read_jsonl_paths([root / "diagnostics.jsonl"])
    env_path = root / "environment.json"
    if not env_path.exists():
        env_path = next(iter(sorted(root.glob("*/environment.json"))), env_path)
    env = _read_json(env_path, default={})
    summary = _build_summary(eval_rows, train_rows, diagnostic_rows)

    report_path = root / "phase1_6_report.md"
    lines = [
        "# Phase 1.6 Report",
        "",
        "## Environment",
        "",
        f"- torch_available: {env.get('torch_available')}",
        f"- device: {env.get('device')}",
        f"- config_hash: {env.get('config_hash')}",
        "",
        "## Checkpoint Integrity",
        "",
        "| model | seed | checkpoint_loaded | train_steps | final_train_relL2 | eval_relL2 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    if summary["checkpoint_integrity"]:
        for row in summary["checkpoint_integrity"]:
            lines.append(
                f"| {row['model']} | {row['seed']} | {row['checkpoint_loaded']} | "
                f"{row['train_steps']} | {_fmt(row['final_train_relL2'])} | {_fmt(row['eval_relL2'])} |"
            )
    else:
        lines.append("| not_available |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Controlled Stress Tasks",
            "",
            "| family | ovha_full | best_single | vector_big | no_memory | no_router | no_adapter | conclusion |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if summary["controlled_stress"]:
        for row in summary["controlled_stress"]:
            lines.append(
                f"| {row['family']} | {_fmt(row.get('ovha_full'))} | {_fmt(row.get('best_single'))} | "
                f"{_fmt(row.get('vector_big'))} | {_fmt(row.get('no_memory'))} | {_fmt(row.get('no_router'))} | "
                f"{_fmt(row.get('no_adapter'))} | {row['conclusion']} |"
            )
    else:
        lines.append("| not_available |  |  |  |  |  |  | no controlled stress metrics found |")

    lines.extend(
        [
            "",
            "## Diagnostic Signals",
            "",
            "| model | family | entropy | memory_norm | adapter_norm | primitive_load |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    if summary["diagnostic_signals"]:
        for row in summary["diagnostic_signals"]:
            lines.append(
                f"| {row['model']} | {row['family']} | {_fmt(row.get('mean_primitive_entropy'))} | "
                f"{_fmt(row.get('mean_memory_norm'))} | {_fmt(row.get('mean_adapter_norm'))} | "
                f"{_fmt_mapping(row.get('primitive_load'))} |"
            )
    else:
        lines.append("| not_available | not_available |  |  |  | no diagnostics rows found |")

    lines.extend(
        [
            "",
            "## Public Benchmark Pilot",
            "",
            "| dataset | split | ovha_full | best_baseline | delta | win? |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    if summary["public_benchmark"]:
        for row in summary["public_benchmark"]:
            lines.append(
                f"| {row['dataset']} | {row['split']} | {_fmt(row.get('ovha_full'))} | "
                f"{_fmt(row.get('best_baseline'))} | {_fmt(row.get('delta'))} | {row.get('win')} |"
            )
    else:
        lines.append("| not_available | not_available |  |  |  | public benchmark loaders prepared; full data run pending |")

    lines.extend(
        [
            "",
            "## Oracle / Untrained Diagnostics",
            "",
            "| model | rows | checkpoint_loaded_rows | note |",
            "|---|---:|---:|---|",
        ]
    )
    for row in summary["oracle_untrained"]:
        lines.append(f"| {row['model']} | {row['rows']} | {row['checkpoint_loaded_rows']} | {row['note']} |")

    lines.extend(
        [
            "",
            "## Dataset Card Appendix",
            "",
            "- PDEBench subset: requires external download/cache; planned splits are iid, parameter_holdout, resolution_transfer, context sweeps, noisy/confusable context where supported.",
            "- FNO classic subset: requires external canonical Burgers/Darcy/Navier-Stokes cache; aligned to neural-operator baselines.",
            "- Mechanical MNIST small: requires external mechanics/material cache; used for parameter/material holdout evidence.",
            "",
            "## Go / No-Go",
            "",
            summary["go_no_go"],
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    summary_path = root / "phase1_6_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(report_path)
    return report_path


def _build_summary(
    eval_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    final_train = {}
    for row in train_rows:
        final_train[(row.get("model"), row.get("seed"))] = row

    grouped_eval = defaultdict(list)
    for row in eval_rows:
        grouped_eval[(row.get("model_name", row.get("model")), row.get("seed"))].append(row)

    checkpoint_rows = []
    for (model, seed), rows in sorted(grouped_eval.items()):
        rel = [float(row["relative_l2"]) for row in rows if "relative_l2" in row]
        train = final_train.get((model, seed), {})
        checkpoint_rows.append(
            {
                "model": model,
                "seed": seed,
                "checkpoint_loaded": all(bool(row.get("checkpoint_loaded")) for row in rows),
                "train_steps": rows[0].get("checkpoint_train_steps"),
                "final_train_relL2": train.get("relative_l2"),
                "eval_relL2": mean(rel) if rel else None,
            }
        )

    stress_rows = []
    by_family_model = defaultdict(list)
    for row in eval_rows:
        family = row.get("family")
        if family in CONTROLLED_STRESS_FAMILIES:
            by_family_model[(family, row.get("model_name", row.get("model")))].append(float(row["relative_l2"]))
    for family in sorted({key[0] for key in by_family_model}):
        values = {model: mean(items) for (fam, model), items in by_family_model.items() if fam == family}
        singles = [values[name] for name in ("local_only", "separable_only", "spectral_only") if name in values]
        best_single = min(singles) if singles else None
        full = values.get("ovha_full")
        conclusion = _controlled_stress_conclusion(
            full=full,
            best_single=best_single,
            vector_big=values.get("ovha_vector_value_big"),
            no_memory=_best_available(values, MEMORY_ABLATION_MODELS),
            no_router=values.get("ovha_no_query_router"),
            no_adapter=values.get("ovha_no_hyper_adapter"),
        )
        stress_rows.append(
            {
                "family": family,
                "ovha_full": full,
                "best_single": best_single,
                "vector_big": values.get("ovha_vector_value_big"),
                "no_memory": _best_available(values, MEMORY_ABLATION_MODELS),
                "no_router": values.get("ovha_no_query_router"),
                "no_adapter": values.get("ovha_no_hyper_adapter"),
                "conclusion": conclusion,
            }
        )

    public_rows = []
    by_dataset = defaultdict(list)
    for row in eval_rows:
        if row.get("dataset"):
            by_dataset[(row["dataset"], row.get("split"), row.get("model_name", row.get("model")))].append(float(row["relative_l2"]))
    for dataset, split in sorted({(key[0], key[1]) for key in by_dataset}):
        values = {model: mean(items) for (ds, sp, model), items in by_dataset.items() if ds == dataset and sp == split}
        full = values.get("ovha_full")
        baselines = [value for model, value in values.items() if model != "ovha_full"]
        best_baseline = min(baselines) if baselines else None
        delta = None if full is None or best_baseline is None else best_baseline - full
        public_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "ovha_full": full,
                "best_baseline": best_baseline,
                "delta": delta,
                "win": delta is not None and delta > 0,
            }
        )

    model_rows = defaultdict(list)
    for row in eval_rows:
        model_rows[row.get("model_name", row.get("model"))].append(row)
    oracle_untrained = []
    for model, rows in sorted(model_rows.items()):
        checkpoint_loaded_rows = sum(1 for row in rows if row.get("checkpoint_loaded"))
        if model and ("oracle" in model or checkpoint_loaded_rows < len(rows)):
            note = "oracle diagnostic" if "oracle" in model else "untrained diagnostic"
            oracle_untrained.append({"model": model, "rows": len(rows), "checkpoint_loaded_rows": checkpoint_loaded_rows, "note": note})
    if not oracle_untrained:
        oracle_untrained.append({"model": "none", "rows": 0, "checkpoint_loaded_rows": 0, "note": "all eval rows were checkpoint-loaded main/ablation rows"})

    go = _go_no_go(checkpoint_rows, stress_rows)
    return {
        "checkpoint_integrity": checkpoint_rows,
        "controlled_stress": stress_rows,
        "public_benchmark": public_rows,
        "oracle_untrained": oracle_untrained,
        "diagnostic_signals": _diagnostic_signals(diagnostic_rows or []),
        "aggregate": _aggregate(eval_rows),
        "go_no_go": go,
    }


def _controlled_stress_conclusion(
    *,
    full: float | None,
    best_single: float | None,
    vector_big: float | None,
    no_memory: float | None,
    no_router: float | None,
    no_adapter: float | None,
) -> str:
    if full is None:
        return "incomplete; ovha_full missing"
    baseline_values = [value for value in (best_single, vector_big) if value is not None]
    if not baseline_values:
        return "incomplete; baseline missing"
    if full >= min(baseline_values):
        return "negative component signal"
    ablation_values = [value for value in (no_memory, no_router, no_adapter) if value is not None]
    if any(value <= full for value in ablation_values):
        return "mixed; ablation signal missing"
    return "provisional component signal"


def _best_available(values: dict[str, float], names: tuple[str, ...]) -> float | None:
    candidates = [values[name] for name in names if name in values]
    return min(candidates) if candidates else None


def _go_no_go(checkpoint_rows: list[dict[str, Any]], stress_rows: list[dict[str, Any]]) -> str:
    checkpoint_ok = bool(checkpoint_rows) and all(row["checkpoint_loaded"] for row in checkpoint_rows)
    if not checkpoint_ok:
        return "No-Go: checkpoint-loaded evaluation is incomplete; fix protocol integrity before interpreting Phase 1.6 results."

    conclusions = [row["conclusion"] for row in stress_rows]
    if any(conclusion == "negative component signal" for conclusion in conclusions):
        return (
            "Scientific No-Go: checkpoint path is wired, but OVHA-full does not beat controlled-stress "
            "baselines/ablations; diagnose model/config before running the larger main experiment."
        )
    if conclusions and all(conclusion == "provisional component signal" for conclusion in conclusions):
        return (
            "Protocol Go with provisional controlled-stress signal: checkpoint path is wired; "
            "public benchmark evidence is still required before Phase 2 scientific Go."
        )
    if conclusions:
        return (
            "Protocol Go only: checkpoint path is wired, but controlled-stress evidence is incomplete "
            "or mixed; diagnose before treating this as a scientific win."
        )
    return (
        "Protocol Go only: checkpoint path is wired; controlled-stress and public benchmark evidence "
        "are still required before Phase 2 scientific Go."
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rel = [float(row["relative_l2"]) for row in rows if "relative_l2" in row]
    if not rel:
        return {"count": 0}
    return {
        "count": len(rel),
        "mean_relative_l2": mean(rel),
        "median_relative_l2": median(rel),
        "std_relative_l2": stdev(rel) if len(rel) > 1 else 0.0,
        "p90_relative_l2": sorted(rel)[min(len(rel) - 1, int(0.9 * (len(rel) - 1)))],
    }


def _diagnostic_signals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        model = row.get("model_name", row.get("model"))
        family = row.get("family")
        if model and family:
            grouped[(model, family)].append(row)

    signals = []
    for (model, family), items in sorted(grouped.items()):
        adapter_means = []
        for row in items:
            adapter_norms = row.get("adapter_norms") or {}
            if adapter_norms:
                adapter_means.append(mean(float(value) for value in adapter_norms.values()))
        signals.append(
            {
                "model": model,
                "family": family,
                "mean_primitive_entropy": _mean_field(items, "primitive_entropy"),
                "mean_memory_norm": _mean_field(items, "memory_norm"),
                "mean_adapter_norm": mean(adapter_means) if adapter_means else None,
                "adapter_norms": _mean_mapping(items, "adapter_norms"),
                "primitive_load": _mean_mapping(items, "primitive_load"),
            }
        )
    return signals


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def _mean_mapping(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values_by_key = defaultdict(list)
    for row in rows:
        mapping = row.get(field) or {}
        for key, value in mapping.items():
            values_by_key[key].append(float(value))
    return {key: mean(values) for key, values in sorted(values_by_key.items())}


def _read_jsonl_paths(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _fmt_mapping(value: Any) -> str:
    if not value:
        return ""
    return ", ".join(f"{key}={_fmt(item)}" for key, item in value.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 1.6 checkpoint-loaded benchmark outputs.")
    parser.add_argument("--root", default="outputs/phase1_6")
    args = parser.parse_args()
    summarize(Path(args.root))


if __name__ == "__main__":
    main()
