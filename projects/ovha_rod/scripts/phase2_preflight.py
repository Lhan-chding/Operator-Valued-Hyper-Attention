#!/usr/bin/env python3
"""Fail closed unless every Phase-2 executable input is reviewed."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from migrate_epoch_resume import validate_runtime_checkout
from ovha_rod.runtime_contracts import validate_local_bert
from server_preflight import (
    CheckResult,
    _cuda_checks,
    _data_checks,
    _mmdet_checks,
    _package_checks,
    _python_checks,
    _storage_checks,
)


PHASE2_CONFIG = Path("configs/phase2/ovha_rod_swin_t_3e_refcoco_full.py")


def _require_checks(checks) -> tuple[dict[str, object], ...]:
    rows = tuple({
        "name": check.name,
        "ok": bool(check.ok),
        "detail": check.detail,
        "observed": check.observed,
    } for check in checks)
    failed = tuple(row["name"] for row in rows if not row["ok"])
    if failed:
        raise RuntimeError(
            "Phase-2 preflight failed: " + ", ".join(str(name) for name in failed))
    return rows


def _training_surface_checks(model, optim_wrapper) -> tuple[CheckResult, ...]:
    """Require the optimizer to contain every and only trainable bank tensor."""
    named_parameters = tuple(model.named_parameters())
    trainable = tuple(
        (name, parameter)
        for name, parameter in named_parameters
        if bool(parameter.requires_grad)
    )
    operator = getattr(model, "decoder_operator", None)
    operator_parameters = (
        tuple(operator.parameters()) if operator is not None else ())
    trainable_ids = {id(parameter) for _, parameter in trainable}
    operator_ids = {id(parameter) for parameter in operator_parameters}
    trainable_names = tuple(name for name, _ in trainable)
    operator_only = (
        bool(trainable)
        and trainable_ids == operator_ids
        and all(name.startswith("decoder_operator.") for name in trainable_names)
    )

    optimizer = getattr(optim_wrapper, "optimizer", None)
    groups = tuple(getattr(optimizer, "param_groups", ()))
    optimizer_parameters = tuple(
        parameter
        for group in groups
        for parameter in group.get("params", ())
    )
    optimizer_ids = tuple(id(parameter) for parameter in optimizer_parameters)
    optimizer_exact = (
        bool(optimizer_parameters)
        and len(optimizer_ids) == len(set(optimizer_ids))
        and set(optimizer_ids) == trainable_ids
        and all(bool(parameter.requires_grad) for parameter in optimizer_parameters)
    )
    return (
        CheckResult(
            "phase2_operator_trainable_only",
            operator_only,
            "only decoder_operator parameters may remain trainable",
            trainable_names,
        ),
        CheckResult(
            "phase2_optimizer_exact_trainable_set",
            optimizer_exact,
            "optimizer parameters must equal the trainable operator set",
            {
                "groups": len(groups),
                "parameters": len(optimizer_parameters),
                "unique_parameters": len(set(optimizer_ids)),
                "trainable_parameters": len(trainable_ids),
            },
        ),
    )


def _phase2_build_checks(
    runtime_project_dir: Path,
    *,
    data_root: Path,
    bert_root: Path,
    work_root: Path,
    per_device_batch: int,
    accumulative_counts: int,
    warmup_iters: int,
    seed: int,
) -> tuple[CheckResult, ...]:
    """Resolve and build the exact Phase-2 model, metric, and optimizer."""
    config_path = runtime_project_dir / PHASE2_CONFIG
    try:
        from mmengine.config import Config
        from mmengine.optim import build_optim_wrapper
        from mmengine.registry import init_default_scope
        from mmdet.registry import METRICS, MODELS
        from ovha_rod.runtime_contracts import find_remote_weight_values

        import ovha_rod  # noqa: F401  # trigger project registry imports

        config = Config.fromfile(str(config_path))
        overrides = {
            "load_from": None,
            "model.train_decoder_operator_only": True,
            "model.language_model.name": str(bert_root),
            "train_dataloader.batch_size": per_device_batch,
            "train_dataloader.dataset.data_root": str(data_root),
            "train_dataloader.dataset.pipeline.5.tokenizer_name": str(bert_root),
            "val_dataloader.dataset.data_root": str(data_root),
            "val_evaluator.ann_file": str(
                data_root
                / "mdetr_annotations"
                / "finetune_refcoco_val.json"),
            "optim_wrapper.accumulative_counts": accumulative_counts,
            "optim_wrapper.clip_grad.error_if_nonfinite": True,
            "param_scheduler.0.end": warmup_iters,
            "randomness.seed": seed,
            "randomness.deterministic": False,
            "default_hooks.checkpoint.by_epoch": True,
            "default_hooks.checkpoint.interval": 1,
            "default_hooks.checkpoint.max_keep_ckpts": 3,
            "default_hooks.checkpoint.save_last": True,
            "custom_hooks.1.identity_path": str(
                work_root / ".phase2-preflight-identity.json"),
        }
        config.merge_from_dict(overrides)
        remote_weights = find_remote_weight_values({
            "model": config.model,
            "load_from": config.get("load_from"),
        })
        if remote_weights:
            raise ValueError(
                f"resolved Phase-2 config contains remote weights: {remote_weights}")
        init_default_scope("mmdet")
        model = MODELS.build(config.model)
        metric = METRICS.build(config.val_evaluator)
        optim_wrapper = build_optim_wrapper(model, config.optim_wrapper)
        surface_checks = _training_surface_checks(model, optim_wrapper)
        bank = config.model.decoder_operator_cfg
        config_exact = (
            int(config.train_cfg.max_epochs) == 3
            and int(config.train_dataloader.batch_size) == per_device_batch
            and int(config.optim_wrapper.accumulative_counts)
            == accumulative_counts
            and int(config.param_scheduler[0].end) == warmup_iters
            and int(config.randomness.seed) == seed
            and config.randomness.deterministic is False
            and config.get("resume", False) is False
            and config.get("load_from") is None
            and config.model.train_decoder_operator_only is True
            and tuple(bank.enabled_operators)
            == ("qsro", "tq_cato", "ms_tleo")
            and int(bank.qsro_query_chunk_size) == 128
            and all(bool(bank[key]) for key in (
                "enabled",
                "use_router",
                "use_memory",
                "use_hyper_adapter",
                "use_rceo",
            ))
            and config.optim_wrapper.constructor
            == "TrainableOnlyOptimWrapperConstructor"
        )
        checks = (
            CheckResult(
                "phase2_resolved_config",
                config_exact,
                "resolved config must match the final three-epoch launch",
                {
                    "config": str(config_path),
                    "epochs": int(config.train_cfg.max_epochs),
                    "batch": int(config.train_dataloader.batch_size),
                    "accumulative_counts": int(
                        config.optim_wrapper.accumulative_counts),
                    "warmup_iters": int(config.param_scheduler[0].end),
                    "seed": int(config.randomness.seed),
                    "operators": tuple(bank.enabled_operators),
                    "optimizer_constructor": str(
                        config.optim_wrapper.constructor),
                },
            ),
            CheckResult(
                "phase2_model_metric_optimizer_build",
                True,
                "final detector, evaluator, and optimizer build successfully",
                {
                    "model": type(model).__name__,
                    "metric": type(metric).__name__,
                    "optim_wrapper": type(optim_wrapper).__name__,
                },
            ),
            CheckResult(
                "phase2_no_remote_model_weights",
                True,
                "resolved model initialization is local-only",
            ),
            *surface_checks,
        )
        del metric, optim_wrapper, model, config
        return checks
    except Exception as exc:  # pragma: no cover - server runtime integration
        return (CheckResult(
            "phase2_model_metric_optimizer_build",
            False,
            f"{type(exc).__name__}: {exc}",
            str(config_path),
        ),)
    finally:
        gc.collect()


def validate_phase2_preflight(
    runtime_project_dir: Path,
    *,
    target_project_commit: str,
    mmdet_root: Path,
    data_root: Path,
    bert_root: Path,
    work_root: Path,
    min_free_gb: float = 8.0,
    per_device_batch: int = 8,
    accumulative_counts: int = 4,
    warmup_iters: int = 2000,
    seed: int = 2026,
) -> dict[str, object]:
    """Attest code, BERT, RefCOCO data, and destination storage."""
    if per_device_batch <= 0 or accumulative_counts <= 0 or warmup_iters <= 0:
        raise ValueError("batch, accumulation, and warmup must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    runtime = validate_runtime_checkout(
        runtime_project_dir, target_project_commit)
    bert = validate_local_bert(bert_root)
    data = data_root.expanduser().resolve()
    work = work_root.expanduser().resolve()
    static_checks = _require_checks((
        *_python_checks(),
        *_mmdet_checks(mmdet_root),
        *_package_checks(mmdet_root),
        *_data_checks(data, "refcoco"),
        *_storage_checks(work, min_free_gb),
        *_cuda_checks(),
    ))
    build_checks = _require_checks(_phase2_build_checks(
        runtime,
        data_root=data,
        bert_root=bert,
        work_root=work,
        per_device_batch=per_device_batch,
        accumulative_counts=accumulative_counts,
        warmup_iters=warmup_iters,
        seed=seed,
    ))
    checks = (*static_checks, *build_checks)
    return {
        "contract": "ovha_rod_phase2_preflight_v1",
        "target_project_commit": target_project_commit,
        "runtime_project_dir": str(runtime),
        "mmdet_root": str(mmdet_root.expanduser().resolve()),
        "data_root": str(data),
        "bert_root": str(bert),
        "work_root": str(work),
        "per_device_batch": per_device_batch,
        "accumulative_counts": accumulative_counts,
        "warmup_iters": warmup_iters,
        "seed": seed,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attest the complete frozen Phase-2 runtime")
    parser.add_argument("--runtime-project-dir", type=Path, required=True)
    parser.add_argument("--target-project-commit", required=True)
    parser.add_argument("--mmdet-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bert-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--per-device-batch", type=int, required=True)
    parser.add_argument("--accumulative-counts", type=int, required=True)
    parser.add_argument("--warmup-iters", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    payload = validate_phase2_preflight(
        args.runtime_project_dir,
        target_project_commit=args.target_project_commit,
        mmdet_root=args.mmdet_root,
        data_root=args.data_root,
        bert_root=args.bert_root,
        work_root=args.work_root,
        min_free_gb=args.min_free_gb,
        per_device_batch=args.per_device_batch,
        accumulative_counts=args.accumulative_counts,
        warmup_iters=args.warmup_iters,
        seed=args.seed,
    )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
