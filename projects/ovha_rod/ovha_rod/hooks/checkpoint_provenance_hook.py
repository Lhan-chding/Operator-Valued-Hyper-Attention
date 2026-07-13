from __future__ import annotations

from pathlib import Path

from mmengine.dist import is_main_process
from mmengine.hooks import Hook

from mmdet.registry import HOOKS

from ..runtime_contracts import write_checkpoint_provenance


@HOOKS.register_module()
class CheckpointProvenanceHook(Hook):
    """Publish a digest sidecar after MMEngine completes an epoch checkpoint."""

    priority = "LOWEST"

    def __init__(self, identity_path: str | None = None) -> None:
        self.identity_path = identity_path

    def after_train_epoch(self, runner) -> None:
        if not is_main_process():
            return
        if not self.identity_path:
            raise ValueError("checkpoint provenance requires run identity path")
        checkpoint = Path(runner.work_dir) / f"epoch_{runner.epoch + 1}.pth"
        write_checkpoint_provenance(checkpoint, Path(self.identity_path))
