#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import stat
from typing import Any


DEFAULT_CONTROLLED_REPORT = Path("outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json")
DEFAULT_REGION_CONFIG = Path("configs/multimodal_refcoco_public_main.json")
DEFAULT_SENTIMENT_CONFIG = Path("configs/multimodal_cmu_mosei_public_main.json")
DEFAULT_REGION_RAW_METRICS = Path("outputs/multimodal/refcoco_main/raw_metrics.jsonl")
DEFAULT_REGION_DIAGNOSTICS = Path("outputs/multimodal/refcoco_main/diagnostics.jsonl")
DEFAULT_REGION_ROBUSTNESS_ROWS = Path("outputs/multimodal/refcoco_main/robustness_rows.jsonl")
DEFAULT_REGION_GATE_DIR = Path("outputs/multimodal/refcoco_main/gate_bundle")
DEFAULT_SENTIMENT_RAW_METRICS = Path("outputs/multimodal/cmu_mosei_main/raw_metrics.jsonl")
DEFAULT_SENTIMENT_DIAGNOSTICS = Path("outputs/multimodal/cmu_mosei_main/diagnostics.jsonl")
DEFAULT_SENTIMENT_ROBUSTNESS_ROWS = Path("outputs/multimodal/cmu_mosei_main/robustness_rows.jsonl")
DEFAULT_SENTIMENT_GATE_DIR = Path("outputs/multimodal/cmu_mosei_main/gate_bundle")
DEFAULT_TOPCONF_MANIFEST = Path("outputs/multimodal/topconf_main_entry/topconf_entry_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write a reproducible runbook for public multimodal main evidence bundles. "
            "The runbook consumes real main raw metrics/diagnostics/robustness rows; it does not promote smoke artifacts."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument("--cache-version", default="v0.1")
    parser.add_argument("--controlled-report", type=Path, default=DEFAULT_CONTROLLED_REPORT)
    parser.add_argument("--region-config", type=Path, default=DEFAULT_REGION_CONFIG)
    parser.add_argument("--region-task", default="phrase_region_grounding")
    parser.add_argument("--region-split", default="test")
    parser.add_argument("--region-raw-metrics", type=Path, default=DEFAULT_REGION_RAW_METRICS)
    parser.add_argument("--region-diagnostics", type=Path, default=DEFAULT_REGION_DIAGNOSTICS)
    parser.add_argument("--region-robustness-rows", type=Path, default=DEFAULT_REGION_ROBUSTNESS_ROWS)
    parser.add_argument("--region-gate-output-dir", type=Path, default=DEFAULT_REGION_GATE_DIR)
    parser.add_argument("--sentiment-config", type=Path, default=DEFAULT_SENTIMENT_CONFIG)
    parser.add_argument("--sentiment-task", default="sentiment_emotion")
    parser.add_argument("--sentiment-split", default="test")
    parser.add_argument("--sentiment-raw-metrics", type=Path, default=DEFAULT_SENTIMENT_RAW_METRICS)
    parser.add_argument("--sentiment-diagnostics", type=Path, default=DEFAULT_SENTIMENT_DIAGNOSTICS)
    parser.add_argument("--sentiment-robustness-rows", type=Path, default=DEFAULT_SENTIMENT_ROBUSTNESS_ROWS)
    parser.add_argument("--sentiment-gate-output-dir", type=Path, default=DEFAULT_SENTIMENT_GATE_DIR)
    parser.add_argument("--topconf-manifest", type=Path, default=DEFAULT_TOPCONF_MANIFEST)
    parser.add_argument("--full-model", default="ovha_full")
    parser.add_argument("--baseline-model", default="cross_attention_transformer")
    args = parser.parse_args()

    payload = build_public_main_runbook(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_public_main_runbook(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runbook_path = args.output_dir / "public_main_runbook.json"
    commands_path = args.output_dir / "public_main_commands.sh"
    commands = _commands(args)
    runbook = {
        "mode": "public_main_runbook",
        "policy": (
            "real public main evidence only; smoke artifacts must not be used as "
            "top-conference main-table inputs"
        ),
        "requires_real_main_metrics": True,
        "datasets": ["refcoco", "cmu_mosei"],
        "controlled_report": str(args.controlled_report),
        "training_configs": {
            "region_text": _training_config(args.region_config),
            "sentiment": _training_config(args.sentiment_config),
        },
        "cache_targets": [
            _cache_target("refcoco", args.cache_root, args.cache_version),
            _cache_target("cmu_mosei", args.cache_root, args.cache_version),
        ],
        "required_real_inputs": {
            "region_text": {
                "task": args.region_task,
                "split": args.region_split,
                "raw_metrics": str(args.region_raw_metrics),
                "diagnostics": str(args.region_diagnostics),
                "robustness_rows": str(args.region_robustness_rows),
                "gate_output_dir": str(args.region_gate_output_dir),
            },
            "sentiment": {
                "task": args.sentiment_task,
                "split": args.sentiment_split,
                "raw_metrics": str(args.sentiment_raw_metrics),
                "diagnostics": str(args.sentiment_diagnostics),
                "robustness_rows": str(args.sentiment_robustness_rows),
                "gate_output_dir": str(args.sentiment_gate_output_dir),
            },
        },
        "input_status": _input_status(
            [
                args.controlled_report,
                args.region_config,
                args.region_raw_metrics,
                args.region_diagnostics,
                args.region_robustness_rows,
                args.sentiment_config,
                args.sentiment_raw_metrics,
                args.sentiment_diagnostics,
                args.sentiment_robustness_rows,
            ]
        ),
        "commands": commands,
        "generated_files": {
            "runbook": str(runbook_path),
            "commands": str(commands_path),
        },
        "next": [
            f"bash -n {commands_path}",
            f"bash {commands_path}",
        ],
    }
    runbook_path.write_text(json.dumps(runbook, indent=2, sort_keys=True) + "\n")
    commands_path.write_text(_shell_script(commands) + "\n")
    commands_path.chmod(commands_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return {
        "ok": True,
        "mode": "public_main_runbook",
        "policy": runbook["policy"],
        "runbook": str(runbook_path),
        "commands": str(commands_path),
        "next": runbook["next"],
    }


def _commands(args: argparse.Namespace) -> list[str]:
    return [
        _validate_training_plan_command(args.region_config),
        _validate_training_plan_command(args.sentiment_config),
        _validate_cache_command("refcoco", args.cache_root, args.cache_version),
        _validate_cache_command("cmu_mosei", args.cache_root, args.cache_version),
        _build_gate_command(
            "region_text",
            raw_metrics=args.region_raw_metrics,
            diagnostics=args.region_diagnostics,
            robustness_rows=args.region_robustness_rows,
            task=args.region_task,
            split=args.region_split,
            output_dir=args.region_gate_output_dir,
            full_model=args.full_model,
            baseline_model=args.baseline_model,
        ),
        _build_gate_command(
            "sentiment",
            raw_metrics=args.sentiment_raw_metrics,
            diagnostics=args.sentiment_diagnostics,
            robustness_rows=args.sentiment_robustness_rows,
            task=args.sentiment_task,
            split=args.sentiment_split,
            output_dir=args.sentiment_gate_output_dir,
            full_model=args.full_model,
            baseline_model=args.baseline_model,
        ),
        _topconf_manifest_command(args),
    ]


def _validate_training_plan_command(config: Path) -> str:
    return f"python scripts/multimodal/validate_training_plan.py {_q(config)}"


def _validate_cache_command(dataset: str, cache_root: Path, version: str) -> str:
    return (
        "python scripts/multimodal/validate_cache.py "
        f"{_q(cache_root)} {_q(dataset)} {_q(version)} --splits train val test"
    )


def _build_gate_command(
    gate: str,
    *,
    raw_metrics: Path,
    diagnostics: Path,
    robustness_rows: Path,
    task: str,
    split: str,
    output_dir: Path,
    full_model: str,
    baseline_model: str,
) -> str:
    return (
        f"python scripts/multimodal/build_public_gate_report.py {_q(gate)} "
        f"--raw-metrics {_q(raw_metrics)} "
        f"--diagnostics {_q(diagnostics)} "
        f"--robustness-rows {_q(robustness_rows)} "
        f"--task {_q(task)} "
        f"--split {_q(split)} "
        f"--output-dir {_q(output_dir)} "
        f"--full-model {_q(full_model)} "
        f"--baseline-model {_q(baseline_model)}"
    )


def _topconf_manifest_command(args: argparse.Namespace) -> str:
    return (
        "python scripts/multimodal/build_topconf_entry_manifest.py "
        f"--output {_q(args.topconf_manifest)} "
        f"--controlled-report {_q(args.controlled_report)} "
        f"--region-gate-bundle {_q(args.region_gate_output_dir)} "
        f"--sentiment-gate-bundle {_q(args.sentiment_gate_output_dir)} "
        f"--cache-target refcoco {_q(args.cache_root)} refcoco {_q(args.cache_version)} train,val,test "
        f"--cache-target cmu_mosei {_q(args.cache_root)} cmu_mosei {_q(args.cache_version)} train,val,test "
        "--validate"
    )


def _cache_target(name: str, cache_root: Path, version: str) -> dict[str, Any]:
    return {
        "name": name,
        "cache_root": str(cache_root),
        "dataset": name,
        "version": version,
        "splits": ["train", "val", "test"],
    }


def _training_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    return {
        "config": str(path),
        "name": str(payload.get("name", "")),
        "dataset_name": str(payload.get("dataset_name", "")),
        "task_type": str(payload.get("task_type", "")),
        "output_dir": str(payload.get("output_dir", "")),
        "seed_count": len(seeds),
        "seeds": seeds,
        "main_table_seed_policy": str(payload.get("main_table_seed_policy", "")),
        "training_stages": list(payload.get("training_stages", [])),
    }


def _input_status(paths: list[Path]) -> dict[str, bool]:
    return {str(path): path.exists() for path in paths}


def _shell_script(commands: list[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated public main runbook.",
        "# This script expects real main raw metrics, diagnostics, and robustness rows.",
        "# Do not substitute public_smoke_* artifacts for top-conference main-table inputs.",
        "",
    ]
    lines.extend(commands)
    return "\n".join(lines)


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
