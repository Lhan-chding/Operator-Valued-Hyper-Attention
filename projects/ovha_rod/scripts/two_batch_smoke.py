#!/usr/bin/env python3
"""Run exactly two MMDetection optimizer iterations and audit artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ovha_rod.runtime_contracts import (
    audit_smoke_outputs,
    find_remote_weight_values,
    prepare_fresh_private_work_dir,
    require_selected_gpus_idle,
    require_visible_device_ids,
    validate_local_bert,
    validate_locked_checkpoint,
)
from ovha_rod.models.positional_encoding import (
    DeterministicSinePositionalEncoding,
)
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


ALLOWED_CFG_OPTIONS = frozenset({
    "model.seed_operator",
    "model.bbox_head.loss_seed_weight",
    "model.bbox_head.loss_ref_weight",
    "model.bbox_head.loss_role_div_weight",
    "load_from",
    "model.language_model.name",
    "train_dataloader.num_workers",
    "train_dataloader.persistent_workers",
    "train_dataloader.dataset.data_root",
    "train_dataloader.dataset.pipeline.5.tokenizer_name",
    "val_dataloader.dataset.data_root",
    "val_evaluator.ann_file",
    "optim_wrapper.type",
    "optim_wrapper.loss_scale",
    "optim_wrapper.accumulative_counts",
    "param_scheduler.0.end",
    "custom_hooks.0.warmup_iters",
    "randomness.seed",
    "randomness.deterministic",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-batch OVHA-ROD train/backward/checkpoint smoke test")
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--batch-size",
        type=int, required=True,
        choices=(1, 2, 4, 8, 16, 32),
        help="Explicit single-GPU capacity-smoke batch size.")
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
    if not config_path.name.startswith("ovha_rod_"):
        raise ValueError("Phase 1 smoke requires an OVHA config")
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
    bert_root = validate_local_bert(
        Path(str(config.model.language_model.name)))
    tokenizer_root = Path(str(
        config.train_dataloader.dataset.pipeline[5].tokenizer_name
    )).expanduser().resolve()
    if tokenizer_root != bert_root:
        raise ValueError(
            "tokenizer root must equal the locked local BERT root")
    expected_operator = str(config.model.get("seed_operator", "none"))
    if expected_operator not in {"none", "generic", "rqgo"}:
        raise ValueError(f"unsupported smoke seed operator: {expected_operator}")
    expected_loss_weights = {
        "none": (0.0, 0.5, 0.0),
        "generic": (0.5, 0.5, 0.0),
        "rqgo": (0.5, 0.5, 0.005),
    }[expected_operator]
    observed_loss_weights = (
        float(config.model.bbox_head.loss_seed_weight),
        float(config.model.bbox_head.loss_ref_weight),
        float(config.model.bbox_head.loss_role_div_weight),
    )
    if observed_loss_weights != expected_loss_weights:
        raise ValueError(
            "smoke auxiliary loss weights do not match the locked variant: "
            f"{observed_loss_weights} != {expected_loss_weights}")
    config.work_dir = str(work_dir)
    config.train_dataloader.batch_size = args.batch_size
    config.train_dataloader.num_workers = 0
    config.train_dataloader.persistent_workers = False
    config.optim_wrapper.type = "AmpOptimWrapper"
    config.optim_wrapper.dtype = "bfloat16"
    config.optim_wrapper.loss_scale = 1.0
    config.optim_wrapper.clip_grad.error_if_nonfinite = True
    config.optim_wrapper.accumulative_counts = 1
    config.train_cfg = dict(
        type="IterBasedTrainLoop", max_iters=2, val_interval=3)
    config.val_cfg = None
    config.val_dataloader = None
    config.val_evaluator = None
    config.param_scheduler = []
    config.default_hooks.logger.interval = 1
    config.default_hooks.checkpoint.interval = 2
    config.default_hooks.checkpoint.by_epoch = False
    config.custom_hooks = [
        hook for hook in config.get("custom_hooks", [])
        if hook.get("type") != "CheckpointProvenanceHook"
    ]
    for hook in config.get("custom_hooks", []):
        if hook.get("type") == "SeedLossWarmupHook":
            hook["warmup_iters"] = 2
        if hook.get("type") == "OperatorDiagnosticsHook":
            hook["interval"] = 1

    require_selected_gpus_idle(device_ids)
    runner = Runner.from_cfg(config)
    observed_batch_size = runner.train_dataloader.batch_size
    if observed_batch_size is None:
        observed_batch_size = runner.train_dataloader.batch_sampler.batch_size
    observed_batch_size = int(observed_batch_size)
    if observed_batch_size != args.batch_size:
        raise RuntimeError(
            "runner dataloader batch size does not match the requested smoke "
            f"capacity: {observed_batch_size} != {args.batch_size}")
    if not isinstance(
            runner.model.positional_encoding,
            DeterministicSinePositionalEncoding):
        raise TypeError(
            "smoke requires DeterministicSinePositionalEncoding")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "smoke requires torch deterministic algorithms")
    positional_mask = torch.tensor(
        [[[False, False, True],
          [False, True, True]]],
        dtype=torch.bool,
        device="cuda",
    )
    with torch.no_grad():
        positional_probe = runner.model.positional_encoding(positional_mask)
    expected_shape = (1, int(runner.model.embed_dims), 2, 3)
    if (tuple(positional_probe.shape) != expected_shape
            or not bool(torch.isfinite(positional_probe).all())):
        raise RuntimeError(
            "deterministic positional CUDA probe failed: "
            f"shape={tuple(positional_probe.shape)}")
    torch.cuda.synchronize()
    print("deterministic positional CUDA probe: ok")
    torch.cuda.reset_peak_memory_stats()
    runner.train()
    torch.cuda.synchronize()
    summary = audit_smoke_outputs(
        Path(config.work_dir), expected_operator=expected_operator)
    summary.update({
        "requested_batch_size": args.batch_size,
        "observed_dataloader_batch_size": observed_batch_size,
        "amp_dtype": str(config.optim_wrapper.dtype),
        "amp_loss_scale": float(config.optim_wrapper.loss_scale),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    })
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
