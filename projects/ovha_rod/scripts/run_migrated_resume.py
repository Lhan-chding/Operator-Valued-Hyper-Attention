#!/usr/bin/env python3
"""Launch a guarded resume only from an attested runtime checkout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from migrate_epoch_resume import validate_runtime_checkout


def _validate_locked_python(python_bin: Path) -> tuple[Path, Path]:
    python_lexical = Path(os.path.abspath(python_bin.expanduser()))
    lexical_parent = python_lexical.parent
    if lexical_parent.is_symlink():
        raise ValueError("locked Python bin directory must not be symlinked")
    python_parent = lexical_parent.resolve(strict=True)
    if lexical_parent != python_parent:
        raise ValueError("locked Python environment path contains a symlink")
    if os.pathsep in str(python_parent):
        raise ValueError("locked Python bin path contains a PATH separator")
    environment_root = python_parent.parent
    home = Path.home().resolve()
    if not environment_root.is_relative_to(home):
        raise ValueError("locked Python environment must stay under the user home")
    for component in (environment_root, *environment_root.parents):
        if component.is_symlink():
            raise ValueError("locked Python environment path is symlinked")
        info = component.stat()
        if info.st_uid not in {0, os.getuid()} or info.st_mode & 0o022:
            raise ValueError("locked Python environment path is unsafe")
        if component == home:
            break
    for directory, names, files in os.walk(environment_root, followlinks=False):
        directory_path = Path(directory)
        info = directory_path.stat()
        if (
            directory_path.is_symlink()
            or not directory_path.is_dir()
            or info.st_uid != os.getuid()
            or info.st_mode & 0o222
        ):
            raise ValueError(
                f"locked Python environment directory is unsafe: {directory_path}")
        for name in names + files:
            path = directory_path / name
            if path.is_symlink():
                target = path.resolve(strict=True)
                target_info = target.stat()
                internal = target.is_relative_to(environment_root)
                if target_info.st_uid == os.getuid():
                    unsafe_mode = bool(target_info.st_mode & 0o222)
                else:
                    unsafe_mode = bool(target_info.st_mode & 0o022)
                if target_info.st_uid not in {0, os.getuid()} or unsafe_mode:
                    raise ValueError(
                        f"locked Python symlink target is unsafe: {path}")
                if target.is_dir() and not internal:
                    raise ValueError(
                        f"locked Python directory symlink escapes the environment: {path}")
                if target.is_file() and target_info.st_nlink != 1:
                    raise ValueError(
                        f"locked Python symlink target is multiply linked: {path}")
                continue
            item_info = path.stat()
            if (
                item_info.st_uid != os.getuid()
                or item_info.st_mode & 0o222
                or (path.is_file() and item_info.st_nlink != 1)
                or not (path.is_file() or path.is_dir())
            ):
                raise ValueError(f"locked Python environment item is unsafe: {path}")
    python_resolved = python_lexical.resolve(strict=True)
    if not python_resolved.is_file() or not os.access(python_resolved, os.X_OK):
        raise ValueError("locked Python executable is invalid")
    return python_lexical, python_parent


def launch_migrated_resume(
    runtime_project_dir: Path,
    target_project_commit: str,
    python_bin: Path,
    runner_arguments: tuple[str, ...],
) -> int:
    if "--resume" not in runner_arguments:
        raise ValueError("migrated launch requires --resume")
    if "--dry-run" in runner_arguments:
        raise ValueError("migrated launch cannot use --dry-run")
    if "--python" in runner_arguments:
        raise ValueError("pass the locked interpreter with --python-bin")
    python_lexical, python_parent = _validate_locked_python(python_bin)
    runtime = validate_runtime_checkout(
        runtime_project_dir, target_project_commit)
    runner = runtime / "scripts" / "run_phase1_server.sh"
    bash = shutil.which("bash", path=os.defpath)
    if bash is None:
        raise RuntimeError("system Bash executable is unavailable")
    allowed = {
        "CUDA_VISIBLE_DEVICES", "HF_HOME", "HOME", "LANG", "LC_ALL",
        "LOGNAME", "TMPDIR",
        "TORCH_HOME", "USER", "XDG_CACHE_HOME",
    }
    environment = {
        key: value for key, value in os.environ.items()
        if key in allowed
    }
    with tempfile.TemporaryDirectory(prefix="ovha-python-path-") as directory:
        isolated_bin = Path(directory)
        (isolated_bin / "python").symlink_to(python_lexical)
        isolated_bin.chmod(0o500)
        environment["PATH"] = f"{isolated_bin}{os.pathsep}{os.defpath}"
        try:
            result = subprocess.run(
                (
                    bash,
                    str(runner),
                    *runner_arguments,
                    "--python", str(python_lexical),
                ),
                cwd=runtime,
                check=False,
                env=environment,
            )
        finally:
            isolated_bin.chmod(0o700)
    return int(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an epoch resume from an attested fixed checkout")
    parser.add_argument("--runtime-project-dir", type=Path, required=True)
    parser.add_argument("--target-project-commit", required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("runner_arguments", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner_arguments = tuple(args.runner_arguments)
    if runner_arguments and runner_arguments[0] == "--":
        runner_arguments = runner_arguments[1:]
    if not runner_arguments:
        raise ValueError("run_phase1_server.sh arguments are required")
    return launch_migrated_resume(
        args.runtime_project_dir,
        args.target_project_commit,
        args.python_bin,
        runner_arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
