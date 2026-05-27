from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union


@dataclass(frozen=True)
class Phase15Config:
    seed: int = 17
    output_dir: Path = Path("outputs/phase1_5_cpu_smoke")
    device: str = "cpu"
    steps: int = 20
    batch_size: int = 2
    support_points: int = 32
    query_points: int = 32
    num_demos: int = 2
    context_points: int = 16
    d_model: int = 64
    memory_tokens: int = 4
    lr: float = 1e-3
    mode: str = "operator_transfer"
    train_split: str = "iid"
    eval_splits: tuple[str, ...] = ("iid", "parameter_holdout", "resolution_transfer", "confusable_context")
    families: tuple[str, ...] = (
        "spectral_family",
        "local_green_family",
        "separable_lowrank_family",
        "nonlinear_family",
        "compositional_mixed_family",
    )
    models: tuple[str, ...] = (
        "ovha_full",
        "transformer_only",
        "perceiver_io_style",
        "icon_style",
        "spectral_only",
        "separable_only",
        "local_only",
        "simple_stack",
        "mlp_expert_moe",
        "ovha_no_memory",
        "ovha_no_hyper_adapter",
        "ovha_vector_value_only",
        "ovha_random_router",
        "ovha_no_query_router",
        "ovha_no_query_adapter",
        "oracle_metadata_upper_bound",
    )
    top_k: Optional[int] = None
    allow_metadata_inputs: bool = False

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Phase15Config":
        raw = json.loads(Path(path).read_text())
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Phase15Config":
        values = dict(raw)
        if "output_dir" in values:
            values["output_dir"] = Path(values["output_dir"])
        for key in ("eval_splits", "families", "models"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def with_overrides(self, output_dir: Optional[Path] = None, device: Optional[str] = None) -> "Phase15Config":
        values = asdict(self)
        if output_dir is not None:
            values["output_dir"] = output_dir
        if device is not None:
            values["device"] = device
        return Phase15Config.from_mapping(values)

    def to_jsonable(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values

    def config_hash(self) -> str:
        payload = json.dumps(self.to_jsonable(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
