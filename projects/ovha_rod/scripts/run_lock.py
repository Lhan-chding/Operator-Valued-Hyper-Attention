#!/usr/bin/env python3
"""Hold a non-blocking, no-follow lock for one private training workdir."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys

from ovha_rod.runtime_contracts import validate_existing_private_work_dir
from ovha_rod.runtime_contracts import (
    require_selected_gpus_idle,
    require_visible_device_ids,
)
from port_guard import require_port_available
from resume_guard import (
    _parse_identity,
    _validated_provenance,
    freeze_checkpoint,
    initialize_identity,
    validate_identity,
)
from ovha_rod.runtime_contracts import validate_private_epoch_checkpoint


def _remove_created_run_lock(lock_path: Path, descriptor: int) -> None:
    """Remove only the lock path still naming the file created by this process."""
    descriptor_info = os.fstat(descriptor)
    try:
        path_info = lock_path.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(path_info.st_mode)
        or path_info.st_dev != descriptor_info.st_dev
        or path_info.st_ino != descriptor_info.st_ino
        or path_info.st_nlink != 1
    ):
        return
    lock_path.unlink()


def _open_run_lock_with_state(work_dir: Path) -> tuple[int, Path | None]:
    root = validate_existing_private_work_dir(work_dir)
    lock_path = root / ".run.lock"
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    created = False
    try:
        descriptor = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        descriptor = os.open(lock_path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
        ):
            raise ValueError("run lock must be a private singly-linked regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, lock_path if created else None


def open_run_lock(work_dir: Path) -> int:
    descriptor, _ = _open_run_lock_with_state(work_dir)
    return descriptor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--guard-gpus", type=int)
    parser.add_argument("--guard-port", type=int)
    parser.add_argument(
        "--expected-identity", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _write_command_log(work_dir: Path, command: list[str], mode: str) -> None:
    path = work_dir / "command.txt"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("command log must be a regular non-symlink file")
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_TRUNC if mode == "fresh" else os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
        ):
            raise ValueError("command log must remain user-owned and private")
        lines = []
        if mode == "resume":
            lines.append("\n# guarded epoch-boundary resume\n")
        for key in (
            "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "MASTER_ADDR",
            "PORT", "TRANSFORMERS_OFFLINE",
        ):
            lines.append(f"{key}={shlex.quote(os.environ.get(key, ''))}\n")
        lines.append(f"COMMAND={shlex.join(command)}\n")
        os.write(descriptor, "".join(lines).encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("locked training command is required")
    expected = _parse_identity(args.expected_identity)
    descriptor, created_lock = _open_run_lock_with_state(args.work_dir)
    try:
        try:
            if args.guard_gpus is not None:
                device_ids = require_visible_device_ids(
                    os.environ.get("CUDA_VISIBLE_DEVICES"), args.guard_gpus)
                require_selected_gpus_idle(device_ids)
            if args.guard_port is not None:
                require_port_available(args.guard_port)
        except BaseException:
            if created_lock is not None:
                _remove_created_run_lock(created_lock, descriptor)
            raise
        if args.mode == "fresh":
            initialize_identity(args.work_dir, expected)
        else:
            identity_path = validate_identity(args.work_dir, expected)
            checkpoint = validate_private_epoch_checkpoint(args.work_dir)
            expected_digest, _ = _validated_provenance(
                checkpoint, identity_path)
            frozen = freeze_checkpoint(
                checkpoint, args.freeze_dir,
                expected_sha256=expected_digest)
            command.extend(("--resume", str(frozen)))
        _write_command_log(args.work_dir, command, args.mode)
        result = subprocess.run(command, cwd=args.cwd, check=False)
        return int(result.returncode)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
