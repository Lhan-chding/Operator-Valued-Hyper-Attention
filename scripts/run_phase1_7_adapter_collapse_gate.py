#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any


FAMILY_TO_CONFIG = {
    "single_primitive_spectral": "configs/phase1_7_g1_single_iid_spectral.json",
    "single_primitive_local": "configs/phase1_7_g1_single_iid_local.json",
    "single_primitive_separable": "configs/phase1_7_g1_single_iid_separable.json",
}
FAMILY_TO_SPECIALIST = {
    "single_primitive_spectral": "spectral_only",
    "single_primitive_local": "local_only",
    "single_primitive_separable": "separable_only",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1.7 oracle-routed adapter specialist-collapse gate.")
    parser.add_argument("--family", choices=tuple(FAMILY_TO_CONFIG) + ("all",), default="all")
    parser.add_argument("--output-root", default="outputs/phase1_7/adapter_collapse_gate")
    parser.add_argument("--device", default=os.environ.get("OVHA_DEVICE", "cuda"))
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--threshold-ratio", type=float, default=1.05)
    parser.add_argument("--threshold-abs", type=float, default=0.01)
    parser.add_argument("--skip-run", action="store_true", help="Only evaluate already-written metrics under output root.")
    args = parser.parse_args()

    families = tuple(FAMILY_TO_CONFIG) if args.family == "all" else (args.family,)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config_paths = []
    for family in families:
        config_path = _write_stage_a_config(family, output_root)
        config_paths.append((family, config_path, Path(json.loads(config_path.read_text())["output_dir"])))

    if not args.skip_run:
        for family, config_path, output_dir in config_paths:
            _run_command([args.python, "-u", "train_torch_meta_operator.py", "--config", str(config_path), "--device", args.device, "--output-dir", str(output_dir)])
            _run_command([args.python, "-u", "eval_torch_meta_operator.py", "--config", str(config_path), "--device", args.device, "--output-dir", str(output_dir)])
            _run_command([args.python, "-u", "scripts/summarize_phase1_6.py", "--root", str(output_dir)])

    report = {
        family: _gate_result(
            family,
            output_dir,
            threshold_ratio=args.threshold_ratio,
            threshold_abs=args.threshold_abs,
        )
        for family, _, output_dir in config_paths
    }
    report_path = output_root / "adapter_collapse_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report_path)
    failed = {family: row for family, row in report.items() if not row["passed"]}
    if failed:
        raise SystemExit(1)


def _write_stage_a_config(family: str, output_root: Path) -> Path:
    base = json.loads(Path(FAMILY_TO_CONFIG[family]).read_text())
    specialist = FAMILY_TO_SPECIALIST[family]
    config = dict(base)
    config["output_dir"] = str(output_root / family)
    config["families"] = [family]
    config["train_models"] = ["ovha_full", specialist]
    config["eval_models"] = ["ovha_full", specialist]
    config["oracle_route_warmup_steps"] = int(config["steps"])
    config["oracle_route_probability"] = 1.0
    config["train_active_primitive_only"] = True
    config["freeze_router"] = True
    config["adapter_auxiliary_loss_weight"] = 1.0
    config["primitive_output_loss_weight"] = 1.0
    config["oracle_routed_prediction_loss_weight"] = 1.0
    config["param_scope_loss_weight"] = 0.1
    config["router_auxiliary_loss_weight"] = 0.0
    config_dir = output_root / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{family}.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config_path


def _gate_result(family: str, output_dir: Path, threshold_ratio: float, threshold_abs: float) -> dict[str, Any]:
    rows = _read_jsonl(output_dir / "eval_metrics.jsonl")
    full_rows = [row for row in rows if row.get("model_name") == "ovha_full" and row.get("family") == family]
    specialist_rows = [row for row in rows if row.get("model_name") == FAMILY_TO_SPECIALIST[family] and row.get("family") == family]
    full_rel = _mean_metric(full_rows, "true_router_learned_adapter_relative_l2") or _mean_metric(full_rows, "relative_l2")
    specialist_rel = _mean_metric(specialist_rows, "relative_l2")
    threshold = None if specialist_rel is None else threshold_ratio * specialist_rel + threshold_abs
    return {
        "family": family,
        "full_true_router_learned_adapter_relL2": full_rel,
        "specialist_relL2": specialist_rel,
        "threshold": threshold,
        "passed": bool(full_rel is not None and threshold is not None and full_rel <= threshold),
    }


def _mean_metric(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return mean(values) if values else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
