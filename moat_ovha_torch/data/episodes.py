from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional


FORBIDDEN_METADATA_KEYS = (
    "family",
    "operator_id",
    "gain",
    "hint",
    "hints",
    "latent",
    "latent_params",
    "mixture",
    "mixture_weights",
    "oracle",
    "oracle_hints",
)


@dataclass(frozen=True)
class MetaOperatorBatch:
    context_u: Any
    context_q: Any
    context_y: Any
    target_u: Any
    target_q: Any
    target_y: Any
    support_grid: Any
    context_mask: Optional[Any] = None
    target_mask: Optional[Any] = None

    def model_inputs(self) -> dict[str, Any]:
        values = {
            "context_u": self.context_u,
            "context_q": self.context_q,
            "context_y": self.context_y,
            "target_u": self.target_u,
            "target_q": self.target_q,
            "support_grid": self.support_grid,
            "context_mask": self.context_mask,
            "target_mask": self.target_mask,
        }
        assert_no_metadata_leakage(values)
        return values


@dataclass(frozen=True)
class EpisodeHiddenInfo:
    family: str
    latent_params: dict[str, Any]
    mixture_weights: Optional[dict[str, Any]] = None
    oracle_hints: Optional[dict[str, Any]] = None


def batch_public_tensor_names() -> tuple[str, ...]:
    return (
        "context_u",
        "context_q",
        "context_y",
        "target_u",
        "target_q",
        "target_y",
        "support_grid",
        "context_mask",
        "target_mask",
    )


def assert_no_metadata_leakage(inputs: dict[str, Any]) -> None:
    lowered = {key.lower() for key in inputs}
    for key in lowered:
        for forbidden in FORBIDDEN_METADATA_KEYS:
            if forbidden in key:
                raise ValueError(f"metadata leakage detected in model input key: {key}")


def hash_model_inputs(batch: MetaOperatorBatch) -> str:
    payload = {name: _stable_value(getattr(batch, name)) for name in batch_public_tensor_names()}
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "shape"):
        tensor = value.detach().cpu()
        return {"shape": list(tensor.shape), "values": tensor.reshape(-1)[:32].tolist()}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stable_value(val) for key, val in sorted(value.items())}
    return value
