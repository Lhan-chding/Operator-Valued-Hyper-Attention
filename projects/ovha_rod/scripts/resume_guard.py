#!/usr/bin/env python3
"""Initialize or verify the private, epoch-boundary resume trust contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import tempfile
from types import MappingProxyType
from typing import Mapping

from ovha_rod.runtime_contracts import validate_private_epoch_checkpoint


CONTRACT = "ovha_rod_epoch_resume_v1"
IDENTITY_FILE = "run_identity.json"
OBSERVER_CONTINUATION_FILE = "observer_continuation.json"
OBSERVER_CONTINUATION_CONTRACT = "ovha_rod_observer_continuation_v1"


@dataclass(frozen=True)
class ObserverCompatibilityPolicy:
    """Immutable exact-blob policy for one reviewed observer transition."""

    source_commit: str
    target_blobs: Mapping[str, str]
    policy_path: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_commit):
            raise ValueError("observer policy source_commit must be 40 lowercase hex")
        blobs = dict(self.target_blobs)
        if not blobs:
            raise ValueError("observer policy must name exact target blobs")
        for path, digest in blobs.items():
            candidate = Path(path)
            if (
                not path
                or candidate.is_absolute()
                or ".." in candidate.parts
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError("observer policy contains an unsafe path or digest")
        if self.policy_path is not None:
            policy_candidate = Path(self.policy_path)
            if (
                policy_candidate.is_absolute()
                or ".." in policy_candidate.parts
                or not self.policy_path
            ):
                raise ValueError("observer policy_path is unsafe")
        object.__setattr__(self, "target_blobs", MappingProxyType(blobs))


def validate_observer_identity_transition(
    source: Mapping[str, str], target: Mapping[str, str],
) -> None:
    """Allow exactly one identity delta: a reviewed project commit change."""
    source_values = dict(source)
    target_values = dict(target)
    if set(source_values) != set(target_values):
        missing = sorted(set(source_values) ^ set(target_values))
        raise ValueError(f"observer identity fields differ: {', '.join(missing)}")
    for key in sorted(source_values):
        if key != "project_commit" and source_values[key] != target_values[key]:
            raise ValueError(f"observer resume cannot change {key}")
    source_commit = source_values.get("project_commit", "")
    target_commit = target_values.get("project_commit", "")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", source_commit)
        or not re.fullmatch(r"[0-9a-f]{40}", target_commit)
        or source_commit == target_commit
    ):
        raise ValueError("observer resume requires two distinct full project_commit values")


def _git_output(repo: Path, *arguments: str, text: bool = True):
    result = subprocess.run(
        ("git", *arguments), cwd=repo, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=text,
    )
    if result.returncode:
        stderr = result.stderr if text else result.stderr.decode("utf-8", "replace")
        raise ValueError(f"observer repository verification failed: {stderr.strip()}")
    return result.stdout


def validate_observer_repository_transition(
    repo: Path,
    source_commit: str,
    target_commit: str,
    policy: ObserverCompatibilityPolicy,
) -> dict[str, object]:
    """Verify a clean checkout whose exact changed blobs match the policy."""
    repository = Path(os.path.abspath(repo.expanduser()))
    if repository.is_symlink() or not repository.is_dir():
        raise ValueError("observer repository must be a real directory")
    if source_commit != policy.source_commit:
        raise ValueError("observer source commit is not approved")
    for name, value in (("source", source_commit), ("target", target_commit)):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"observer {name} commit must be full lowercase hex")
    head = _git_output(repository, "rev-parse", "HEAD").strip()
    if head != target_commit:
        raise ValueError("observer target commit is not the checked-out HEAD")
    if _git_output(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("observer target checkout must be clean")

    changed: dict[str, str] = {}
    output = _git_output(
        repository, "diff", "--name-status", "--no-renames",
        source_commit, target_commit,
    )
    for line in output.splitlines():
        status_value, separator, path = line.partition("\t")
        if not separator or status_value not in {"A", "M"}:
            raise ValueError("observer transition contains an unauthorized change type")
        changed[path] = status_value
    approved_paths = set(policy.target_blobs)
    observed_paths = set(changed)
    if policy.policy_path is not None:
        if policy.policy_path not in observed_paths:
            raise ValueError("observer transition policy file is missing from the diff")
        observed_paths.remove(policy.policy_path)
    if observed_paths != approved_paths:
        unauthorized = sorted(observed_paths ^ approved_paths)
        raise ValueError(
            "observer transition contains unauthorized paths: "
            + ", ".join(unauthorized))

    target_digests: dict[str, str] = {}
    audited_paths = sorted(approved_paths)
    for path in audited_paths:
        tree_row = _git_output(
            repository, "ls-tree", target_commit, "--", path).strip()
        match = re.fullmatch(r"(100644|100755) blob [0-9a-f]{40}\t(.+)", tree_row)
        if match is None or match.group(2) != path:
            raise ValueError(f"observer target is not a regular git blob: {path}")
        content = _git_output(
            repository, "show", f"{target_commit}:{path}", text=False)
        digest = hashlib.sha256(content).hexdigest()
        if digest != policy.target_blobs[path]:
            raise ValueError(f"observer target blob digest mismatch: {path}")
        target_digests[path] = digest

    canonical = json.dumps(
        {
            "source_commit": source_commit,
            "target_commit": target_commit,
            "changed": [
                {"path": path, "status": changed[path], "sha256": target_digests[path]}
                for path in audited_paths
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "source_commit": source_commit,
        "target_commit": target_commit,
        "changed_paths": tuple(audited_paths),
        "diff_sha256": hashlib.sha256(canonical).hexdigest(),
        "target_blobs": MappingProxyType(target_digests),
    }


def _validate_private_manifest_input(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    info = lexical.stat()
    if (
        info.st_uid != os.getuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
        or info.st_size <= 0
        or info.st_size > 64 * 1024
    ):
        raise ValueError(f"{label} must be a small private singly-linked file")
    return lexical.resolve(strict=True)


def _canonical_observer_transition(
    transition: Mapping[str, object], *, source_commit: str, target_commit: str,
) -> dict[str, object]:
    """Validate and materialize the immutable repository-audit result."""
    values = dict(transition)
    expected_keys = {
        "source_commit", "target_commit", "changed_paths",
        "diff_sha256", "target_blobs",
    }
    if set(values) != expected_keys:
        raise ValueError("observer transition schema is invalid")
    if (
        values["source_commit"] != source_commit
        or values["target_commit"] != target_commit
    ):
        raise ValueError("observer transition commits do not match lineage")
    diff_digest = values["diff_sha256"]
    if not isinstance(diff_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", diff_digest
    ):
        raise ValueError("observer transition diff digest is invalid")
    changed_paths_value = values["changed_paths"]
    if not isinstance(changed_paths_value, (tuple, list)):
        raise ValueError("observer transition changed_paths is invalid")
    changed_paths = tuple(changed_paths_value)
    if (
        not changed_paths
        or not all(isinstance(path, str) and path for path in changed_paths)
        or len(set(changed_paths)) != len(changed_paths)
    ):
        raise ValueError("observer transition changed_paths is invalid")
    blobs_value = values["target_blobs"]
    if not isinstance(blobs_value, Mapping):
        raise ValueError("observer transition target_blobs is invalid")
    blobs = dict(blobs_value)
    if set(blobs) != set(changed_paths) or not all(
        isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in blobs.values()
    ):
        raise ValueError("observer transition target_blobs is invalid")
    return {
        "source_commit": source_commit,
        "target_commit": target_commit,
        "changed_paths": list(changed_paths),
        "diff_sha256": diff_digest,
        "target_blobs": {path: blobs[path] for path in sorted(blobs)},
    }


def write_observer_continuation_manifest(
    target_work_dir: Path,
    *,
    source_identity_path: Path,
    source_provenance_path: Path,
    source_checkpoint_sha256: str,
    source_commit: str,
    target_commit: str,
    transition: Mapping[str, object],
) -> Path:
    """Create an append-only lineage record without rewriting source evidence."""
    root = _private_directory(target_work_dir)
    identity = _validate_private_manifest_input(
        source_identity_path, "source identity")
    provenance = _validate_private_manifest_input(
        source_provenance_path, "source provenance")
    if not re.fullmatch(r"[0-9a-f]{64}", source_checkpoint_sha256):
        raise ValueError("source checkpoint SHA-256 is invalid")
    for label, value in (("source", source_commit), ("target", target_commit)):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"{label} commit is invalid")
    destination = root / OBSERVER_CONTINUATION_FILE
    if destination.exists() or destination.is_symlink():
        raise ValueError("observer continuation manifest already exists")
    transition_payload = _canonical_observer_transition(
        transition, source_commit=source_commit, target_commit=target_commit)
    payload = {
        "contract": OBSERVER_CONTINUATION_CONTRACT,
        "source_identity_sha256": hashlib.sha256(identity.read_bytes()).hexdigest(),
        "source_provenance_sha256": hashlib.sha256(
            provenance.read_bytes()).hexdigest(),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_commit": source_commit,
        "target_commit": target_commit,
        "transition": transition_payload,
    }
    rendered = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        os.write(descriptor, rendered.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def read_identity(work_dir: Path) -> tuple[Path, dict[str, str]]:
    """Read a validated identity without accepting caller-selected omissions."""
    root = _private_directory(work_dir)
    manifest = _validate_private_manifest_input(
        root / IDENTITY_FILE, "source identity")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if set(payload) != {"contract", "identity"} or payload["contract"] != CONTRACT:
        raise ValueError("source run identity contract is invalid")
    identity = payload["identity"]
    if (
        not isinstance(identity, dict)
        or not identity
        or not all(isinstance(key, str) and isinstance(value, str)
                   for key, value in identity.items())
    ):
        raise ValueError("source run identity payload is invalid")
    return manifest, dict(identity)


def load_observer_compatibility_policy(path: Path) -> ObserverCompatibilityPolicy:
    """Load the repository-owned exact transition policy as inert JSON."""
    lexical = Path(os.path.abspath(path.expanduser()))
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError("observer compatibility policy must be a regular file")
    info = lexical.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o022 or info.st_size > 64 * 1024:
        raise ValueError("observer compatibility policy is not trusted")
    payload = json.loads(lexical.read_text(encoding="utf-8"))
    if set(payload) != {"contract", "source_commit", "target_blobs", "policy_path"}:
        raise ValueError("observer compatibility policy schema is invalid")
    if payload["contract"] != "ovha_rod_observer_compatibility_policy_v1":
        raise ValueError("observer compatibility policy contract is invalid")
    return ObserverCompatibilityPolicy(
        source_commit=payload["source_commit"],
        target_blobs=payload["target_blobs"],
        policy_path=payload["policy_path"],
    )


def prepare_observer_compatible_resume(
    *,
    source_work_dir: Path,
    target_work_dir: Path,
    target_identity: Mapping[str, str],
    repository: Path,
    freeze_dir: Path,
    policy: ObserverCompatibilityPolicy,
) -> tuple[Path, Path]:
    """Validate, freeze, and seal one observer-only epoch continuation."""
    source_identity_path, source_identity = read_identity(source_work_dir)
    target_values = dict(target_identity)
    validate_observer_identity_transition(source_identity, target_values)
    source_commit = source_identity["project_commit"]
    target_commit = target_values["project_commit"]
    transition = validate_observer_repository_transition(
        repository, source_commit, target_commit, policy)
    checkpoint = validate_private_epoch_checkpoint(source_work_dir)
    expected_digest, _ = _validated_provenance(
        checkpoint, source_identity_path)
    frozen = freeze_checkpoint(
        checkpoint, freeze_dir, expected_sha256=expected_digest)

    target_root = _private_directory(target_work_dir)
    target_identity_path = initialize_identity(target_root, target_values)
    source_provenance = checkpoint.with_name(
        checkpoint.name + ".provenance.json")
    manifest = write_observer_continuation_manifest(
        target_root,
        source_identity_path=source_identity_path,
        source_provenance_path=source_provenance,
        source_checkpoint_sha256=expected_digest,
        source_commit=source_commit,
        target_commit=target_commit,
        transition=transition,
    )
    if not target_identity_path.is_file() or not manifest.is_file():
        raise RuntimeError("observer continuation sealing failed")
    return frozen, manifest


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
