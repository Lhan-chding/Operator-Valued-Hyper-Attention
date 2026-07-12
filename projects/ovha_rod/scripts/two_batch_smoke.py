#!/usr/bin/env python3
"""Run exactly two MMDetection optimizer iterations and audit artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-batch OVHA-ROD train/backward/checkpoint smoke test")
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default={})
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trusted_root = Path(__file__).resolve().parents[1] / "configs"
    config_path = args.config.resolve()
    if not config_path.is_relative_to(trusted_root.resolve()):
        raise ValueError("smoke config must come from the project configs directory")
    config = Config.fromfile(str(config_path))
    config.merge_from_dict(args.cfg_options)
    config.work_dir = str(args.work_dir.resolve())
    config.train_cfg = dict(
        type="IterBasedTrainLoop", max_iters=2, val_interval=3)
    config.param_scheduler = []
    config.default_hooks.logger.interval = 1
    config.default_hooks.checkpoint.interval = 2
    config.default_hooks.checkpoint.by_epoch = False
    for hook in config.get("custom_hooks", []):
        if hook.get("type") == "SeedLossWarmupHook":
            hook["warmup_iters"] = 2
        if hook.get("type") == "OperatorDiagnosticsHook":
            hook["interval"] = 1

    runner = Runner.from_cfg(config)
    runner.train()
    summary = _audit_outputs(Path(config.work_dir))
    summary_path = Path(config.work_dir) / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 2


def _audit_outputs(work_dir: Path) -> dict:
    diagnostics_path = work_dir / "operator_diagnostics.jsonl"
    rows = []
    if diagnostics_path.is_file():
        rows = [json.loads(line) for line in diagnostics_path.read_text().splitlines()
                if line.strip()]
    nonfinite = []
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                nonfinite.append({"row": row_index, "key": key, "value": value})
    checkpoints = sorted(str(path) for path in work_dir.glob("iter_2.pth"))
    active_rows = [row for row in rows if row.get("seed_operator") in {"rqgo", "generic"}]
    gradients_ok = not active_rows or any(
        float(row.get("seed_gradient_norm", 0.0)) > 0.0 for row in active_rows)
    bias_ok = all(float(row.get("seed_bias_abs_max", 0.0)) <= 2.0 + 1e-6
                  and float(row.get("seed_invalid_bias_abs_max", 0.0)) == 0.0
                  for row in active_rows)
    ok = (len(rows) >= 2 and not nonfinite and bool(checkpoints)
          and gradients_ok and bias_ok)
    return {
        "ok": ok,
        "contract": "ovha_rod_phase1_two_batch_smoke_v1",
        "diagnostic_rows": len(rows),
        "nonfinite": nonfinite,
        "active_seed_gradient_nonzero": gradients_ok,
        "seed_bias_contract": bias_ok,
        "checkpoints": checkpoints,
        "diagnostics": str(diagnostics_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
