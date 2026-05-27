from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union


@dataclass(frozen=True)
class Phase1Config:
    seed: int = 0
    output_dir: Path = Path("outputs/phase1")
    families: tuple[str, ...] = ("fourier", "green", "separable", "mixed")
    context_sizes: tuple[int, ...] = (2, 4, 8)
    resolution: int = 16
    baselines: tuple[str, ...] = (
        "transformer_only",
        "perceiver_io_style",
        "icon_style",
        "deeponet",
        "fno",
        "simple_stack",
    )
    ablations: tuple[str, ...] = (
        "vector_value",
        "no_hyper_adapter",
        "no_memory",
        "mlp_expert",
        "random_router",
    )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Phase1Config":
        raw = json.loads(Path(path).read_text())
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Phase1Config":
        values = dict(raw)
        if "output_dir" in values:
            values["output_dir"] = Path(values["output_dir"])
        for key in ("families", "context_sizes", "baselines", "ablations"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)
