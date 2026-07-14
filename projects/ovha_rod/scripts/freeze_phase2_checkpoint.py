#!/usr/bin/env python3
"""Validate and freeze an inactive Phase-1 epoch-2 checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from migrate_epoch_resume import (
    _read_source_identity,
    _require_commit,
    _sha256,
    _write_private_json,
)
from ovha_rod.runtime_contracts import (
    LOCKED_CHECKPOINT_SHA256,
    validate_existing_private_work_dir,
)
from resume_guard import _private_directory, _validated_provenance, freeze_checkpoint
from server_preflight import EXPECTED_MMDET_COMMIT


SOURCE_ENVIRONMENT_PROFILE = "cu121-wheel"
SOURCE_GLOBAL_BATCH = "32"
SOURCE_GPUS = "1"
SOURCE_PER_DEVICE_BATCH = "8"
SOURCE_ACCUMULATIVE_COUNTS = "4"
RECEIPT_CONTRACT = "ovha_rod_phase2_frozen_source_v1"


def _open_inactive_run_lock(work_dir: Path) -> int:
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
            raise RuntimeError(
                "source Phase-1 run is still active; stop it after epoch 2"
            ) from exc
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_source_identity(
    identity: dict[str, str],
    *,
    expected_source_project_commit: str,
    expected_seed: int,
    expected_data_root: str,
    expected_bert_root: str,
) -> str:
    expected_commit = _require_commit(
        expected_source_project_commit, "expected source project commit")
    if not isinstance(expected_seed, int) or isinstance(expected_seed, bool) \
            or expected_seed < 0:
        raise ValueError("expected seed must be a non-negative integer")
    expected = {
        "dataset": "refcoco",
        "variant": "rqgo",
        "seed": str(expected_seed),
        "gpus": SOURCE_GPUS,
        "per_device_batch": SOURCE_PER_DEVICE_BATCH,
        "accumulative_counts": SOURCE_ACCUMULATIVE_COUNTS,
        "global_batch": SOURCE_GLOBAL_BATCH,
        "amp": "false",
        "amp_dtype": "none",
        "data_root": expected_data_root,
        "bert_root": expected_bert_root,
        "initial_checkpoint_sha256": LOCKED_CHECKPOINT_SHA256,
        "project_commit": expected_commit,
        "mmdetection_commit": EXPECTED_MMDET_COMMIT,
        "environment_profile": SOURCE_ENVIRONMENT_PROFILE,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            label = "source project commit" if key == "project_commit" else key
            raise ValueError(f"{label} does not match the explicit expectation")
    physical_devices = identity.get("physical_cuda_devices", "")
    if not re.fullmatch(r"[0-9]+", physical_devices):
        raise ValueError("source physical CUDA device identity is invalid")
    return expected_commit


def _validate_private_json(path: Path, expected: dict[str, object]) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"frozen checkpoint metadata is missing: {path.name}")
    info = path.stat()
    if (
        info.st_uid != os.getuid()
        or info.st_mode & 0o777 != 0o400
        or info.st_nlink != 1
        or info.st_size <= 0
        or info.st_size > 64 * 1024
    ):
        raise ValueError(f"frozen checkpoint metadata is unsafe: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != expected:
        raise ValueError(f"frozen checkpoint metadata changed: {path.name}")


def _validate_frozen_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("content-addressed frozen checkpoint is missing")
    info = path.stat()
    if (
        info.st_uid != os.getuid()
        or info.st_mode & 0o777 != 0o400
        or info.st_nlink != 1
        or info.st_size != expected_size
        or _sha256(path) != expected_sha256
    ):
        raise ValueError("content-addressed frozen checkpoint is unsafe")


def _content_addressed_freeze(
    checkpoint: Path,
    freeze_dir: Path,
    *,
    checkpoint_sha256: str,
    checkpoint_size: int,
    source_identity_sha256: str,
    receipt: dict[str, object],
) -> Path:
    root = _private_directory(freeze_dir, create=True)
    name = (
        f"phase1-epoch2-{checkpoint_sha256}-"
        f"{source_identity_sha256}.pth")
    frozen = root / name
    manifest = frozen.with_name(frozen.name + ".sha256.json")
    receipt_path = frozen.with_name(frozen.name + ".phase2-source.json")
    manifest_payload = {
        "contract": "ovha_rod_frozen_resume_v1",
        "sha256": checkpoint_sha256,
        "size": checkpoint_size,
        "source": str(checkpoint),
    }
    if frozen.exists() or frozen.is_symlink():
        _validate_frozen_checkpoint(
            frozen,
            expected_sha256=checkpoint_sha256,
            expected_size=checkpoint_size,
        )
        _validate_private_json(manifest, manifest_payload)
        _validate_private_json(receipt_path, receipt)
        return frozen

    required_free = checkpoint_size + 512 * 1024 * 1024
    if shutil.disk_usage(root).free < required_free:
        raise RuntimeError("insufficient free space to freeze the epoch-2 checkpoint")

    random_frozen: Path | None = None
    created: list[Path] = []
    try:
        random_frozen = freeze_checkpoint(
            checkpoint, root, expected_sha256=checkpoint_sha256)
        random_manifest = random_frozen.with_name(
            random_frozen.name + ".sha256.json")
        os.link(random_frozen, frozen, follow_symlinks=False)
        created.append(frozen)
        random_frozen.unlink()
        random_frozen = None
        random_manifest.unlink()
        _write_private_json(manifest, manifest_payload, 0o400)
        created.append(manifest)
        _write_private_json(receipt_path, receipt, 0o400)
        created.append(receipt_path)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _validate_frozen_checkpoint(
            frozen,
            expected_sha256=checkpoint_sha256,
            expected_size=checkpoint_size,
        )
        return frozen
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if random_frozen is not None:
            random_manifest = random_frozen.with_name(
                random_frozen.name + ".sha256.json")
            try:
                random_frozen.unlink()
            except FileNotFoundError:
                pass
            try:
                random_manifest.unlink()
            except FileNotFoundError:
                pass


def freeze_phase2_checkpoint(
    checkpoint: Path,
    freeze_dir: Path,
    *,
    expected_source_project_commit: str,
    expected_seed: int,
    expected_data_root: str,
    expected_bert_root: str,
) -> tuple[Path, str, str, str]:
    lexical = Path(os.path.abspath(checkpoint.expanduser()))
    if lexical.name != "epoch_2.pth":
        raise ValueError("Phase 2 requires the exact epoch_2.pth checkpoint")
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError("epoch_2.pth must be a regular non-symlink file")
    work_dir = validate_existing_private_work_dir(lexical.parent)
    resolved = lexical.resolve(strict=True)
    if resolved.parent != work_dir:
        raise ValueError("epoch_2.pth must stay directly inside its work directory")
    info = resolved.stat()
    if (
        info.st_uid != os.getuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
        or info.st_size <= 0
    ):
        raise ValueError("epoch_2.pth must be private, non-empty, and singly linked")

    descriptor = _open_inactive_run_lock(work_dir)
    try:
        identity_path, identity = _read_source_identity(work_dir)
        source_commit = _validate_source_identity(
            identity,
            expected_source_project_commit=expected_source_project_commit,
            expected_seed=expected_seed,
            expected_data_root=expected_data_root,
            expected_bert_root=expected_bert_root,
        )
        checkpoint_sha256, checkpoint_size = _validated_provenance(
            resolved, identity_path)
        provenance_path = resolved.with_name(
            resolved.name + ".provenance.json")
        identity_sha256 = _sha256(identity_path)
        provenance_sha256 = _sha256(provenance_path)
        receipt = {
            "contract": RECEIPT_CONTRACT,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size": checkpoint_size,
            "source_checkpoint": str(resolved),
            "source_identity_sha256": identity_sha256,
            "source_project_commit": source_commit,
            "source_provenance_sha256": provenance_sha256,
        }
        frozen = _content_addressed_freeze(
            resolved,
            freeze_dir,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_size=checkpoint_size,
            source_identity_sha256=identity_sha256,
            receipt=receipt,
        )
        if (
            _sha256(identity_path) != identity_sha256
            or _sha256(provenance_path) != provenance_sha256
        ):
            raise ValueError("source identity or provenance changed while freezing")
    finally:
        os.close(descriptor)
    return frozen, checkpoint_sha256, identity_sha256, provenance_sha256


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a provenance-bound epoch-2 checkpoint for Phase 2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--expected-source-project-commit", required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-data-root", required=True)
    parser.add_argument("--expected-bert-root", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    frozen, digest, identity_digest, provenance_digest = (
        freeze_phase2_checkpoint(
            args.checkpoint,
            args.freeze_dir,
            expected_source_project_commit=(
                args.expected_source_project_commit),
            expected_seed=args.expected_seed,
            expected_data_root=args.expected_data_root,
            expected_bert_root=args.expected_bert_root,
        ))
    if any(ord(character) < 0x20 for character in str(frozen)):
        raise ValueError("frozen checkpoint path contains control characters")
    print(frozen)
    print(digest)
    print(identity_digest)
    print(provenance_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
