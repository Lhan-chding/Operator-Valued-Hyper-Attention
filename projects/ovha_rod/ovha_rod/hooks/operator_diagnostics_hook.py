from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from mmengine.dist import is_main_process
from mmengine.hooks import Hook

from mmdet.registry import HOOKS

from ..gradient_accumulator import (
    accumulate_gradient_square,
    finalize_gradient_norm,
)
from ..runtime_contracts import collect_scalar_diagnostics


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
        model._seed_grad_squared = None
        if getattr(model, "seed_operator", None) is not None:
            for parameter in model.seed_operator.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(
                        lambda gradient, target=model: _record_gradient(target, gradient))

    def before_train_iter(self, runner, batch_idx: int,
                          data_batch=None) -> None:
        del batch_idx, data_batch
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        model._seed_grad_squared = None

    def after_train_iter(self, runner, batch_idx: int,
                         data_batch=None, outputs=None) -> None:
        del batch_idx, data_batch
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        gradient_norm = finalize_gradient_norm(
            getattr(model, "_seed_grad_squared", None))
        finite_scalars = {}
        nonfinite_keys = []
        for values, prefix in (
            (getattr(model, "last_seed_diagnostics", {}), ""),
            (getattr(model.bbox_head, "last_seed_metrics", {}), ""),
            (outputs or {}, "loss/"),
        ):
            finite, nonfinite = collect_scalar_diagnostics(values, prefix=prefix)
            finite_scalars.update(finite)
            nonfinite_keys.extend(nonfinite)
        if not math.isfinite(gradient_norm):
            nonfinite_keys.append("seed_gradient_norm")
        if nonfinite_keys:
            raise FloatingPointError(
                "non-finite Phase 1 diagnostics: "
                + ", ".join(sorted(set(nonfinite_keys))))
        if not is_main_process() or (runner.iter + 1) % self.interval:
            return
        row = {
            "iteration": int(runner.iter + 1),
            "seed_operator": getattr(model, "seed_operator_name", "unknown"),
            "loss_seed_weight": float(getattr(model.bbox_head, "loss_seed_weight", 0.0)),
            "seed_gradient_norm": gradient_norm,
        }
        row.update(finite_scalars)
        row["nonfinite_scalar_count"] = len(nonfinite_keys)
        row["nonfinite_scalar_keys"] = sorted(set(nonfinite_keys))
        path = Path(runner.work_dir) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(
                row, sort_keys=True, allow_nan=False) + "\n")


def _record_gradient(model, gradient: torch.Tensor) -> None:
    model._seed_grad_squared = accumulate_gradient_square(
        getattr(model, "_seed_grad_squared", None), gradient)
