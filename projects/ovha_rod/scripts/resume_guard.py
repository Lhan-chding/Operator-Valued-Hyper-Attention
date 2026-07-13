#!/usr/bin/env python3
"""Initialize or verify the private, epoch-boundary resume trust contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile

from ovha_rod.runtime_contracts import validate_private_epoch_checkpoint


CONTRACT = "ovha_rod_epoch_resume_v1"
IDENTITY_FILE = "run_identity.json"


def _parse_identity(values: list[str]) -> dict[str, str]:
    identity: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"identity must use KEY=VALUE: {value!r}")
        key, item = value.split("=", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"invalid identity key: {key!r}")
        if not item or len(item) > 512 or any(ord(char) < 0x20 for char in item):
            raise ValueError(f"invalid identity value for {key}")
        if key in identity:
            raise ValueError(f"duplicate identity key: {key}")
        identity[key] = item
    if not identity:
        raise ValueError("at least one expected identity field is required")
    return identity


def _private_directory(path: Path, *, create: bool = False) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    home = Path.home().resolve()
    if not lexical.is_relative_to(home):
        raise ValueError(f"private directory must stay under {home}")
    if create and not lexical.exists():
        parent = _private_directory(lexical.parent)
        lexical = parent / lexical.name
        lexical.mkdir(mode=0o700)
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError("private directory must be a real directory")
    for component in (lexical, *lexical.parents):
        if component.is_symlink():
            raise ValueError(f"private directory component is symlinked: {component}")
        info = component.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError(f"private directory component is not private: {component}")
        if component == home:
            break
    lexical.chmod(0o700)
    return lexical.resolve(strict=True)


def initialize_identity(work_dir: Path, expected: dict[str, str]) -> Path:
    root = _private_directory(work_dir)
    manifest = root / IDENTITY_FILE
    if manifest.exists() or manifest.is_symlink():
        raise ValueError("run identity already exists; refusing to overwrite it")
    payload = {"contract": CONTRACT, "identity": expected}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(manifest, flags, 0o600)
    try:
        rendered = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        os.write(descriptor, rendered.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest


def validate_identity(work_dir: Path, expected: dict[str, str]) -> Path:
    root = _private_directory(work_dir)
    manifest = root / IDENTITY_FILE
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("private run identity is missing")
    info = manifest.stat()
    if (
        info.st_uid != os.getuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
        or info.st_size > 64 * 1024
    ):
        raise ValueError("run identity must be user-owned and private")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload != {"contract": CONTRACT, "identity": expected}:
        raise ValueError("resume identity does not exactly match the original run")
    return manifest


def _validated_provenance(
    checkpoint: Path, identity_path: Path,
) -> tuple[str, int]:
    sidecar = checkpoint.with_name(checkpoint.name + ".provenance.json")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError("checkpoint provenance sidecar is missing")
    info = sidecar.stat()
    if (
        info.st_uid != os.getuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
        or info.st_size > 64 * 1024
    ):
        raise ValueError("checkpoint provenance sidecar is not private")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    expected_keys = {
        "contract", "checkpoint", "checkpoint_sha256",
        "checkpoint_size", "run_identity_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("checkpoint provenance schema is invalid")
    identity_digest = hashlib.sha256(identity_path.read_bytes()).hexdigest()
    digest = payload["checkpoint_sha256"]
    size = payload["checkpoint_size"]
    if (
        payload["contract"] != "ovha_rod_checkpoint_provenance_v1"
        or payload["checkpoint"] != checkpoint.name
        or payload["run_identity_sha256"] != identity_digest
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or not isinstance(size, int)
        or size <= 0
        or size != checkpoint.stat().st_size
    ):
        raise ValueError("checkpoint provenance does not match this run")
    return digest, size


def freeze_checkpoint(
    checkpoint: Path, freeze_dir: Path, *, expected_sha256: str,
) -> Path:
    """Copy an already-guarded checkpoint to an immutable private snapshot."""
    destination_root = _private_directory(freeze_dir, create=True)
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_fd = os.open(checkpoint, source_flags)
    temporary_path: Path | None = None
    try:
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_mode & 0o077
            or before.st_nlink != 1
            or before.st_size <= 0
        ):
            raise ValueError("resume source changed before freezing")
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
                prefix=".resume-", suffix=".pth", dir=destination_root,
                delete=False) as output:
            temporary_path = Path(output.name)
            while True:
                chunk = os.read(source_fd, 8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(source_fd)
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("resume source changed while it was being frozen")
        observed_digest = digest.hexdigest()
        if observed_digest != expected_sha256:
            raise ValueError("resume checkpoint digest does not match provenance")
        temporary_path.chmod(0o400)
        frozen = destination_root / (
            f"{checkpoint.stem}-{observed_digest}-{secrets.token_hex(8)}.pth")
        os.link(temporary_path, frozen, follow_symlinks=False)
        temporary_path.unlink()
        temporary_path = None
        frozen_info = frozen.stat()
        if (
            frozen.is_symlink()
            or not frozen.is_file()
            or frozen_info.st_uid != os.getuid()
            or frozen_info.st_mode & 0o777 != 0o400
            or frozen_info.st_nlink != 1
            or frozen_info.st_size != before.st_size
        ):
            raise ValueError("frozen resume checkpoint failed final checks")
        manifest = frozen.with_name(frozen.name + ".sha256.json")
        manifest_payload = {
            "contract": "ovha_rod_frozen_resume_v1",
            "sha256": observed_digest,
            "size": frozen_info.st_size,
            "source": str(checkpoint),
        }
        descriptor = os.open(
            manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400)
        try:
            os.write(descriptor, (json.dumps(
                manifest_payload, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_fd = os.open(destination_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return frozen
    finally:
        os.close(source_fd)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-identity", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--freeze-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    expected = _parse_identity(args.expected_identity)
    if args.initialize:
        if args.freeze_dir is not None:
            raise ValueError("--freeze-dir is invalid with --initialize")
        print(initialize_identity(args.work_dir, expected))
        return 0
    if args.freeze_dir is None:
        raise ValueError("--freeze-dir is required when validating resume")
    identity_path = validate_identity(args.work_dir, expected)
    checkpoint = validate_private_epoch_checkpoint(args.work_dir)
    expected_digest, _ = _validated_provenance(checkpoint, identity_path)
    print(freeze_checkpoint(
        checkpoint, args.freeze_dir, expected_sha256=expected_digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
