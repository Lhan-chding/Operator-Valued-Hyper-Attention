#!/usr/bin/env python3
"""Rebind one trusted epoch checkpoint to a reviewed project commit."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from ovha_rod.runtime_contracts import (
    validate_existing_private_work_dir,
    validate_private_epoch_checkpoint,
    write_checkpoint_provenance,
)
from resume_guard import (
    CONTRACT as IDENTITY_CONTRACT,
    IDENTITY_FILE,
    _parse_identity,
    _private_directory,
    _validated_provenance,
    initialize_identity,
)


MIGRATION_CONTRACT = "ovha_rod_resume_migration_v1"
MIGRATION_FILE = "resume_migration.json"
REQUIRED_IDENTITY_FIELDS = frozenset({
    "dataset",
    "variant",
    "seed",
    "gpus",
    "per_device_batch",
    "accumulative_counts",
    "global_batch",
    "amp",
    "amp_dtype",
    "physical_cuda_devices",
    "data_root",
    "bert_root",
    "initial_checkpoint_sha256",
    "project_commit",
    "mmdetection_commit",
    "environment_profile",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_commit(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return value


def _read_source_identity(work_dir: Path) -> tuple[Path, dict[str, str]]:
    root = validate_existing_private_work_dir(work_dir)
    manifest = root / IDENTITY_FILE
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("private source run identity is missing")
    info = manifest.stat()
    if (
        info.st_uid != os.getuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
        or info.st_size <= 0
        or info.st_size > 64 * 1024
    ):
        raise ValueError("source run identity must be user-owned and private")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"contract", "identity"}:
        raise ValueError("source run identity schema is invalid")
    identity = payload.get("identity")
    if payload.get("contract") != IDENTITY_CONTRACT or not isinstance(identity, dict):
        raise ValueError("source run identity contract is invalid")
    if set(identity) != REQUIRED_IDENTITY_FIELDS:
        raise ValueError("source run identity fields are incomplete or unexpected")
    if not all(isinstance(key, str) and isinstance(value, str)
               for key, value in identity.items()):
        raise ValueError("source run identity fields must be strings")
    validated = _parse_identity([
        f"{key}={value}" for key, value in identity.items()
    ])
    if validated != identity:
        raise ValueError("source run identity failed canonical validation")
    return manifest, validated


def _open_source_lock(work_dir: Path) -> int:
    lock_path = work_dir / ".run.lock"
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags)
    except FileNotFoundError as exc:
        raise ValueError("source run lock is missing") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
        ):
            raise ValueError("source run lock must be private and singly linked")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("source run is still active") from exc
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _copy_checkpoint(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> tuple[str, int]:
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_fd = os.open(source, source_flags)
    destination_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_mode & 0o077
            or before.st_nlink != 1
            or before.st_size <= 0
        ):
            raise ValueError("source checkpoint is not a private regular file")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        destination_fd = os.open(destination, flags, 0o600)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(source_fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("checkpoint copy made no progress")
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("source checkpoint changed during migration")
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise ValueError("source checkpoint digest does not match provenance")
        os.fchmod(destination_fd, 0o400)
        os.fsync(destination_fd)
        return observed, before.st_size
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)


def _write_private_json(path: Path, payload: dict[str, object], mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        rendered = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        remaining = memoryview(rendered.encode("utf-8"))
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("private JSON write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_incomplete_target(target: Path, created: list[Path]) -> None:
    for path in reversed(created):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    try:
        target.rmdir()
    except FileNotFoundError:
        pass


def migrate_epoch_resume(
    source_work_dir: Path,
    target_work_dir: Path,
    *,
    runtime_project_dir: Path,
    expected_source_project_commit: str,
    target_project_commit: str,
) -> Path:
    """Copy a guarded checkpoint while changing only its project commit."""
    runtime_project = validate_runtime_checkout(
        runtime_project_dir, target_project_commit)
    source_root = validate_existing_private_work_dir(source_work_dir)
    source_lock = _open_source_lock(source_root)
    try:
        return _migrate_epoch_resume_locked(
            source_root,
            target_work_dir,
            runtime_project_dir=runtime_project,
            expected_source_project_commit=expected_source_project_commit,
            target_project_commit=target_project_commit,
        )
    finally:
        os.close(source_lock)


def _migrate_epoch_resume_locked(
    source_work_dir: Path,
    target_work_dir: Path,
    *,
    runtime_project_dir: Path,
    expected_source_project_commit: str,
    target_project_commit: str,
) -> Path:
    expected_source = _require_commit(
        expected_source_project_commit, "expected source project commit")
    target_commit = _require_commit(
        target_project_commit, "target project commit")
    source_identity_path, source_identity = _read_source_identity(source_work_dir)
    source_commit = _require_commit(
        source_identity["project_commit"], "source project commit")
    if source_commit != expected_source:
        raise ValueError("source project commit does not match the explicit expectation")
    if source_commit == target_commit:
        raise ValueError("source and target project commits must differ")

    source_root = source_identity_path.parent
    source_checkpoint = validate_private_epoch_checkpoint(source_root)
    source_digest, source_size = _validated_provenance(
        source_checkpoint, source_identity_path)
    source_provenance = source_checkpoint.with_name(
        source_checkpoint.name + ".provenance.json")
    source_identity_digest = _sha256(source_identity_path)
    source_provenance_digest = _sha256(source_provenance)

    target_lexical = Path(os.path.abspath(target_work_dir.expanduser()))
    parent = _private_directory(target_lexical.parent)
    target = parent / target_lexical.name
    if (
        target == source_root
        or target.is_relative_to(source_root)
        or target.exists()
        or target.is_symlink()
    ):
        raise ValueError("target work directory must not exist")

    target.mkdir(mode=0o700)
    created: list[Path] = []
    try:
        target_identity = dict(source_identity)
        target_identity["project_commit"] = target_commit
        identity_path = initialize_identity(target, target_identity)
        created.append(identity_path)

        checkpoint = target / source_checkpoint.name
        created.append(checkpoint)
        observed_digest, observed_size = _copy_checkpoint(
            source_checkpoint, checkpoint, expected_sha256=source_digest)
        if observed_size != source_size:
            raise ValueError("source checkpoint size does not match provenance")

        provenance = write_checkpoint_provenance(checkpoint, identity_path)
        created.append(provenance)
        receipt = target / MIGRATION_FILE
        _write_private_json(receipt, {
            "contract": MIGRATION_CONTRACT,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": observed_digest,
            "checkpoint_size": observed_size,
            "source_identity_sha256": source_identity_digest,
            "source_project_commit": source_commit,
            "source_provenance_sha256": source_provenance_digest,
            "runtime_project_dir": str(runtime_project_dir),
            "target_identity_sha256": _sha256(identity_path),
            "target_project_commit": target_commit,
        }, 0o400)
        created.append(receipt)

        pointer = target / "last_checkpoint"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(pointer, flags, 0o600)
        try:
            os.write(descriptor, f"{checkpoint.name}\n".encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        created.append(pointer)
        directory_fd = os.open(target, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

        guarded = validate_private_epoch_checkpoint(target)
        final_digest, final_size = _validated_provenance(
            guarded, identity_path)
        if (final_digest, final_size) != (observed_digest, observed_size):
            raise ValueError("target checkpoint failed final provenance validation")
        if (
            _sha256(source_identity_path) != source_identity_digest
            or _sha256(source_provenance) != source_provenance_digest
        ):
            raise ValueError("source identity or provenance changed during migration")
        validate_runtime_checkout(runtime_project_dir, target_commit)
        return target.resolve(strict=True)
    except BaseException:
        _cleanup_incomplete_target(target, created)
        raise


def _git_output(project_dir: Path, *arguments: str) -> str:
    git = shutil.which("git", path=os.defpath)
    if git is None:
        raise RuntimeError("system Git executable is unavailable")
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": str(Path.home()),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }
    result = subprocess.run(
        (
            git,
            "--no-replace-objects",
            "-c", "core.fsmonitor=false",
            "-c", "core.hooksPath=/dev/null",
            "-c", "core.attributesFile=/dev/null",
            "-c", "diff.external=",
            "-C", str(project_dir),
            *arguments,
        ),
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _git_blob_digest(path: Path) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"tracked path is not a regular file: {path}")
        digest = hashlib.sha1()
        digest.update(f"blob {before.st_size}\0".encode("ascii"))
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"tracked file changed during validation: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _require_read_only_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"runtime directory is unsafe: {path}")
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o222:
        raise ValueError(f"runtime directory must be user-owned and read-only: {path}")


def _attest_runtime_tree(root: Path, top_level: Path, commit: str) -> None:
    for directory in (top_level, top_level / "projects", root):
        _require_read_only_directory(directory)
    prefix = root.relative_to(top_level).as_posix() + "/"
    tree = _git_output(
        PROJECT_DIR, "ls-tree", "-rz", "--full-tree", commit, "--", prefix)
    expected: dict[Path, tuple[str, str]] = {}
    for record in tree.split("\0"):
        if not record:
            continue
        metadata, separator, name = record.partition("\t")
        parts = metadata.split()
        if not separator or len(parts) != 3 or parts[1] != "blob":
            raise ValueError("runtime commit tree contains an unsupported entry")
        mode, _, digest = parts
        candidate = PurePosixPath(name)
        if candidate.is_absolute() or ".." in candidate.parts or not name.startswith(prefix):
            raise ValueError("runtime commit tree contains an unsafe path")
        relative = Path(*PurePosixPath(name[len(prefix):]).parts)
        if not relative.parts or relative in expected:
            raise ValueError("runtime commit tree contains a duplicate path")
        expected[relative] = (mode, digest)
    if not expected:
        raise ValueError("runtime project tree is empty")

    observed: set[Path] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        _require_read_only_directory(directory_path)
        for name in tuple(names):
            path = directory_path / name
            if path.is_symlink():
                observed.add(path.relative_to(root))
                names.remove(name)
        observed.update(
            (directory_path / name).relative_to(root) for name in files)
    if observed != set(expected):
        raise ValueError("runtime checkout has missing or extra project files")

    for relative, (mode, expected_digest) in expected.items():
        path = root / relative
        info = path.lstat()
        if mode in {"100644", "100755"}:
            if (
                path.is_symlink()
                or not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o222
                or info.st_mode & 0o022
                or info.st_nlink != 1
            ):
                raise ValueError(f"tracked runtime file is unsafe: {path}")
            executable = bool(info.st_mode & 0o111)
            if executable != (mode == "100755"):
                raise ValueError(f"tracked executable mode mismatch: {path}")
            digest = _git_blob_digest(path)
        else:
            raise ValueError(f"tracked runtime symlinks are not allowed: {path}")
        if digest != expected_digest:
            raise ValueError(f"tracked runtime file does not match commit: {path}")


def _validate_detached_runtime_metadata(top_level: Path, commit: str) -> None:
    git_dir = top_level / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise ValueError("runtime must be a standalone Git clone")
    if (git_dir / "objects" / "info" / "alternates").exists():
        raise ValueError("runtime Git object alternates are not allowed")
    for directory, names, files in os.walk(git_dir, followlinks=False):
        directory_path = Path(directory)
        _require_read_only_directory(directory_path)
        for name in names:
            if (directory_path / name).is_symlink():
                raise ValueError("runtime Git metadata must not contain symlinks")
        for name in files:
            path = directory_path / name
            info = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o222
                or info.st_nlink != 1
            ):
                raise ValueError(f"runtime Git metadata file is unsafe: {path}")
    head = git_dir / "HEAD"
    expected = f"{commit}\n".encode("ascii")
    if head.read_bytes() != expected:
        raise ValueError("runtime checkout is not detached at the target commit")


def validate_runtime_checkout(project_dir: Path, target_commit: str) -> Path:
    expected = _require_commit(target_commit, "target project commit")
    root = project_dir.expanduser().resolve(strict=True)
    if root.name != "ovha_rod" or root.parent.name != "projects":
        raise ValueError("runtime project directory is not the reviewed project path")
    top_level = root.parents[1]
    if root != top_level / "projects" / "ovha_rod":
        raise ValueError("runtime project directory is not the reviewed project path")
    _validate_detached_runtime_metadata(top_level, expected)
    _attest_runtime_tree(root, top_level, expected)
    try:
        _validate_detached_runtime_metadata(top_level, expected)
    except ValueError as exc:
        raise ValueError("runtime checkout changed during validation") from exc
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate one guarded epoch checkpoint to a reviewed commit")
    parser.add_argument("--source-work-dir", type=Path, required=True)
    parser.add_argument("--target-work-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--expected-source-project-commit", required=True)
    parser.add_argument("--target-project-commit", required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    print(migrate_epoch_resume(
        args.source_work_dir,
        args.target_work_dir,
        runtime_project_dir=args.project_dir,
        expected_source_project_commit=args.expected_source_project_commit,
        target_project_commit=args.target_project_commit,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
