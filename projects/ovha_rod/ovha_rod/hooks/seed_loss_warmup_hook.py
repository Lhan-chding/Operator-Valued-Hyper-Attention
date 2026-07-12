from __future__ import annotations

from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class SeedLossWarmupHook(Hook):
    """Linearly warm the auxiliary seed loss without freezing the parent."""

    priority = "NORMAL"

    def __init__(self, warmup_iters: int = 500) -> None:
        if warmup_iters <= 0:
            raise ValueError("warmup_iters must be positive")
        self.warmup_iters = int(warmup_iters)

    def before_train(self, runner) -> None:
        head = _bbox_head(runner.model)
        target = float(getattr(head, "loss_seed_target_weight", head.loss_seed_weight))
        head.loss_seed_target_weight = target
        head.loss_seed_weight = 0.0

    def before_train_iter(self, runner, batch_idx: int,
                          data_batch=None) -> None:
        del batch_idx, data_batch
        head = _bbox_head(runner.model)
        progress = min(float(runner.iter + 1) / self.warmup_iters, 1.0)
        head.loss_seed_weight = head.loss_seed_target_weight * progress


def _bbox_head(model):
    unwrapped = model.module if hasattr(model, "module") else model
    return unwrapped.bbox_head
