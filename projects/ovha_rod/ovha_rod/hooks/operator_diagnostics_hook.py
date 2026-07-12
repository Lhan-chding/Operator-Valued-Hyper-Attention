from __future__ import annotations

import json
from pathlib import Path

import torch
from mmengine.dist import is_main_process
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class OperatorDiagnosticsHook(Hook):
    """Write finite, machine-readable Phase-1 diagnostics as JSONL."""

    priority = "LOW"

    def __init__(self, interval: int = 50,
                 filename: str = "operator_diagnostics.jsonl") -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.interval = int(interval)
        if Path(filename).name != filename:
            raise ValueError("diagnostics filename must be a basename")
        self.filename = filename

    def before_train(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        model._seed_grad_squared = 0.0
        if getattr(model, "seed_operator", None) is not None:
            for parameter in model.seed_operator.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(
                        lambda gradient, target=model: _record_gradient(target, gradient))

    def before_train_iter(self, runner, batch_idx: int,
                          data_batch=None) -> None:
        del batch_idx, data_batch
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        model._seed_grad_squared = 0.0

    def after_train_iter(self, runner, batch_idx: int,
                         data_batch=None, outputs=None) -> None:
        del batch_idx, data_batch
        if not is_main_process() or (runner.iter + 1) % self.interval:
            return
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        row = {
            "iteration": int(runner.iter + 1),
            "seed_operator": getattr(model, "seed_operator_name", "unknown"),
            "loss_seed_weight": float(getattr(model.bbox_head, "loss_seed_weight", 0.0)),
            "seed_gradient_norm": float(
                getattr(model, "_seed_grad_squared", 0.0) ** 0.5),
        }
        row.update(_finite_scalars(getattr(model, "last_seed_diagnostics", {})))
        row.update(_finite_scalars(
            getattr(model.bbox_head, "last_seed_metrics", {})))
        row.update(_finite_scalars(outputs or {}, prefix="loss/"))
        path = Path(runner.work_dir) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _finite_scalars(values, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            number = float(value.detach().cpu())
        elif isinstance(value, (float, int)):
            number = float(value)
        else:
            continue
        if math_isfinite(number):
            result[f"{prefix}{key}"] = number
    return result


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _record_gradient(model, gradient: torch.Tensor) -> None:
    model._seed_grad_squared += float(gradient.detach().float().square().sum().cpu())
