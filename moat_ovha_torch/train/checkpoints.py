from __future__ import annotations

from pathlib import Path


def checkpoint_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / "checkpoints" / f"{model_name}.pt"
