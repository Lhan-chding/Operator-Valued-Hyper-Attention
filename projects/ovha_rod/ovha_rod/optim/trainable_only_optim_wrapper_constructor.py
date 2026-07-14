"""MMEngine optimizer construction for a strictly frozen parent model."""

from __future__ import annotations

from typing import Any, Optional

from mmengine.optim import (
    OPTIM_WRAPPER_CONSTRUCTORS,
    DefaultOptimWrapperConstructor,
)


def _trainable_param_groups(
    param_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return new parameter groups containing trainable parameters only."""
    filtered: list[dict[str, Any]] = []
    for group in param_groups:
        trainable = [
            parameter
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        if trainable:
            filtered.append({**group, "params": trainable})
    return filtered


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TrainableOnlyOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Preserve MMEngine paramwise rules while excluding frozen parameters."""

    def __init__(
        self,
        optim_wrapper_cfg: dict[str, Any],
        paramwise_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        effective_paramwise_cfg = dict(paramwise_cfg or {})
        if not effective_paramwise_cfg:
            # DefaultOptimWrapperConstructor bypasses ``add_params`` when this
            # mapping is empty. The harmless sentinel keeps filtering active.
            effective_paramwise_cfg["bypass_duplicate"] = False
        super().__init__(optim_wrapper_cfg, effective_paramwise_cfg)

    def add_params(
        self,
        params: list[dict[str, Any]],
        module: Any,
        prefix: str = "",
        is_dcn_module: Optional[float] = None,
    ) -> None:
        super().add_params(
            params,
            module,
            prefix=prefix,
            is_dcn_module=is_dcn_module,
        )
        if not prefix:
            params[:] = _trainable_param_groups(params)
