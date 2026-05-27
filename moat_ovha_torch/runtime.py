from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any, Optional


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def numpy_available() -> bool:
    return importlib.util.find_spec("numpy") is not None


def require_torch():
    if not torch_available():
        raise RuntimeError("Torch is not installed; install requirements_torch.txt to run torch workloads.")
    import torch

    return torch


def environment_payload(device: str, config_hash: Optional[str] = None) -> dict[str, Any]:
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "torch_available": torch_available(),
        "numpy_available": numpy_available(),
        "config_hash": config_hash,
    }
    if torch_available():
        torch = require_torch()
        payload["torch_version"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return payload


def write_environment(output_dir: Path, device: str, config_hash: Optional[str] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "environment.json"
    path.write_text(json.dumps(environment_payload(device, config_hash), indent=2, sort_keys=True) + "\n")
    return path


def write_torch_skip_report(output_dir: Path, command_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "phase1_5_report.md"
    path.write_text(
        "\n".join(
            [
                "# Phase 1.5 Report",
                "",
                "## Status",
                "",
                "Torch is not installed; torch implementation tests skipped.",
                "",
                f"Command `{command_name}` did not run training or evaluation on this machine.",
                "",
                "Install `requirements_torch.txt` in a local virtual environment to run CPU smoke tests.",
            ]
        )
        + "\n"
    )
    return path
