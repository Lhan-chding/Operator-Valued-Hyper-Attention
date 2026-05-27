from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from moat_ovha_torch.config import Phase15Config
from moat_ovha_torch.runtime import require_torch


def checkpoint_path(output_dir: Path, model_name: str, seed: int | None = None) -> Path:
    if seed is None:
        return output_dir / "checkpoints" / f"{model_name}.pt"
    return output_dir / "checkpoints" / model_name / f"seed_{seed}" / "model.pt"


def train_metrics_path(output_dir: Path, model_name: str, seed: int) -> Path:
    return output_dir / "train_metrics" / model_name / f"seed_{seed}.jsonl"


def eval_metrics_path(output_dir: Path, model_name: str, seed: int) -> Path:
    return output_dir / "eval_metrics" / model_name / f"seed_{seed}.jsonl"


def diagnostics_path(output_dir: Path, model_name: str, seed: int) -> Path:
    return output_dir / "diagnostics" / model_name / f"seed_{seed}.jsonl"


def parameter_count(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def save_training_checkpoint(model: Any, config: Phase15Config, model_name: str) -> Path:
    torch = require_torch()
    path = checkpoint_path(config.output_dir, model_name, config.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "config": config.to_jsonable(),
            "seed": config.seed,
            "train_steps": config.steps,
            "config_hash": config.config_hash(),
            "parameter_count": parameter_count(model),
            "git_commit": current_git_commit(),
        },
        path,
    )
    return path


def load_checkpoint_for_eval(model: Any, checkpoint: Path, strict: bool = True) -> dict[str, Any]:
    torch = require_torch()
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"required evaluation checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload.get("model"))
    if state_dict is None:
        raise KeyError(f"checkpoint has no model_state_dict: {checkpoint}")
    model.load_state_dict(state_dict, strict=strict)
    return payload


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
