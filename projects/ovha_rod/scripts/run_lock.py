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
from resume_guard import (
    OBSERVER_CONTINUATION_FILE,
    _parse_identity,
    _validated_provenance,
    freeze_checkpoint,
    initialize_identity,
    load_observer_compatibility_policy,
    prepare_observer_compatible_resume,
    validate_identity,
)
from ovha_rod.runtime_contracts import validate_private_epoch_checkpoint


def open_run_lock(work_dir: Path) -> int:
    root = validate_existing_private_work_dir(work_dir)
    lock_path = root / ".run.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
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
    return descriptor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("fresh", "resume", "observer-resume"),
        required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--source-work-dir", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--observer-policy", type=Path)
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
        elif mode == "observer-resume":
            lines.append("\n# guarded observer-compatible epoch continuation\n")
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
    descriptor = open_run_lock(args.work_dir)
    source_descriptor = None
    try:
        if args.mode == "fresh":
            initialize_identity(args.work_dir, expected)
        elif args.mode == "resume":
            identity_path = validate_identity(args.work_dir, expected)
            checkpoint = validate_private_epoch_checkpoint(args.work_dir)
            expected_digest, _ = _validated_provenance(
                checkpoint, identity_path)
            frozen = freeze_checkpoint(
                checkpoint, args.freeze_dir,
                expected_sha256=expected_digest)
            command.extend(("--resume", str(frozen)))
        else:
            if (
                args.source_work_dir is None
                or args.project_root is None
                or args.observer_policy is None
            ):
                raise ValueError(
                    "observer-resume requires source workdir, project root, "
                    "and observer policy")
            if args.source_work_dir.resolve() == args.work_dir.resolve():
                raise ValueError(
                    "observer continuation requires a separate target workdir")
            source_descriptor = open_run_lock(args.source_work_dir)
            policy = load_observer_compatibility_policy(args.observer_policy)
            frozen, manifest = prepare_observer_compatible_resume(
                source_work_dir=args.source_work_dir,
                target_work_dir=args.work_dir,
                target_identity=expected,
                repository=args.project_root,
                freeze_dir=args.freeze_dir,
                policy=policy,
            )
            if manifest.name != OBSERVER_CONTINUATION_FILE:
                raise RuntimeError("observer_continuation.json was not sealed")
            command.extend(("--resume", str(frozen)))
        _write_command_log(args.work_dir, command, args.mode)
        result = subprocess.run(command, cwd=args.cwd, check=False)
        return int(result.returncode)
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
