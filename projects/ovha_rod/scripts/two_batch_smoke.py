#!/usr/bin/env python3
"""Run exactly two MMDetection optimizer iterations and audit artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from ovha_rod.runtime_contracts import (
    audit_smoke_outputs,
    find_remote_weight_values,
    prepare_fresh_private_work_dir,
    require_selected_gpus_idle,
    require_visible_device_ids,
    validate_local_bert,
    validate_locked_checkpoint,
)
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


ALLOWED_CFG_OPTIONS = frozenset({
    "model.seed_operator",
    "load_from",
    "model.language_model.name",
    "train_dataloader.batch_size",
    "train_dataloader.num_workers",
    "train_dataloader.dataset.data_root",
    "train_dataloader.dataset.pipeline.5.tokenizer_name",
    "val_dataloader.dataset.data_root",
    "val_evaluator.ann_file",
    "optim_wrapper.type",
    "optim_wrapper.loss_scale",
    "randomness.seed",
    "randomness.deterministic",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-batch OVHA-ROD train/backward/checkpoint smoke test")
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default={})
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    unknown_options = sorted(set(args.cfg_options) - ALLOWED_CFG_OPTIONS)
    if unknown_options:
        raise ValueError(f"smoke cfg-options are not allowed: {unknown_options}")
    device_ids = require_visible_device_ids(
        os.environ.get("CUDA_VISIBLE_DEVICES"), expected_count=1)
    require_selected_gpus_idle(device_ids)
    trusted_root = Path(__file__).resolve().parents[1] / "configs"
    config_path = args.config.resolve()
    if not config_path.is_relative_to(trusted_root.resolve()):
        raise ValueError("smoke config must come from the project configs directory")
    work_dir = prepare_fresh_private_work_dir(args.work_dir)
    config = Config.fromfile(str(config_path))
    config.merge_from_dict(args.cfg_options)
    remote_weights = find_remote_weight_values({
        "model": config.model,
        "load_from": config.get("load_from"),
    })
    if remote_weights:
        raise ValueError(f"remote model weights are forbidden: {remote_weights}")
    validate_locked_checkpoint(Path(str(config.get("load_from", ""))))
    validate_local_bert(Path(str(config.model.language_model.name)))
    expected_operator = str(config.model.get("seed_operator", "none"))
    if expected_operator not in {"none", "generic", "rqgo"}:
        raise ValueError(f"unsupported smoke seed operator: {expected_operator}")
    config.work_dir = str(work_dir)
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
    summary = audit_smoke_outputs(
        Path(config.work_dir), expected_operator=expected_operator)
    summary_path = Path(config.work_dir) / "smoke_summary.json"
    if summary_path.is_symlink():
        raise ValueError("smoke summary must not be a symlink")
    rendered = json.dumps(
        summary, indent=2, sort_keys=True, allow_nan=False)
    summary_path.write_text(rendered + "\n")
    print(rendered)
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
