from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

import torch


LOCKED_CHECKPOINT_SHA256 = (
    "b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a")
LOCKED_CHECKPOINT_SIZE = 1093815743
LOCKED_BERT_SHA256 = (
    "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3")
LOCKED_BERT_SIZE = 440449768


REQUIRED_SMOKE_FIELDS = frozenset({
    "iteration",
    "seed_operator",
    "loss_seed_weight",
    "seed_gradient_norm",
    "seed_bias_abs_max",
    "seed_invalid_bias_abs_max",
    "nonfinite_scalar_count",
})


def require_visible_device_ids(
        value: str | None, expected_count: int) -> tuple[int, ...]:
    if expected_count <= 0:
        raise ValueError("expected GPU count must be positive")
    if value is None or not value.strip():
        raise ValueError("CUDA_VISIBLE_DEVICES must be set explicitly")
    tokens = tuple(token.strip() for token in value.split(","))
    if any(not token.isdigit() for token in tokens):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain numeric GPU indices")
    device_ids = tuple(int(token) for token in tokens)
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("CUDA_VISIBLE_DEVICES must not contain duplicates")
    if len(device_ids) != expected_count:
        raise ValueError(
            "visible GPU count does not match requested process count: "
            f"{len(device_ids)} != {expected_count}")
    return device_ids


def require_selected_gpus_idle(
    device_ids: Sequence[int],
    run_command: Callable[..., Any] = subprocess.run,
) -> None:
    gpu_result = run_command(
        ("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"),
        check=False, capture_output=True, text=True)
    if gpu_result.returncode != 0:
        raise RuntimeError(f"cannot query GPU inventory: {gpu_result.stderr}")
    index_to_uuid = {}
    for line in gpu_result.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) == 2 and fields[0].isdigit():
            index_to_uuid[int(fields[0])] = fields[1]
    missing = tuple(index for index in device_ids if index not in index_to_uuid)
    if missing:
        raise RuntimeError(f"selected GPU indices do not exist: {missing}")

    app_result = run_command(
        ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader"),
        check=False, capture_output=True, text=True)
    if app_result.returncode != 0:
        raise RuntimeError(f"cannot query GPU processes: {app_result.stderr}")
    selected = {index_to_uuid[index]: index for index in device_ids}
    busy: list[dict[str, str | int]] = []
    for line in app_result.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(",", 3))
        if len(fields) == 4 and fields[0] in selected:
            busy.append({
                "gpu": selected[fields[0]],
                "pid": fields[1],
                "process": fields[2],
                "memory": fields[3],
            })
    if busy:
        detail = "; ".join(
            f"GPU {row['gpu']} pid={row['pid']} {row['process']} {row['memory']}"
            for row in busy)
        raise RuntimeError(f"selected GPU already has compute processes: {detail}")


def prepare_fresh_private_work_dir(path: Path) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    trusted_home = Path.home().resolve()
    if not lexical.is_relative_to(trusted_home):
        raise ValueError(f"work directory must stay under {trusted_home}")
    if lexical.is_symlink():
        raise ValueError("work directory must not be a symlink")
    for parent in lexical.parents:
        if not parent.exists():
            continue
        if parent.is_symlink():
            raise ValueError(f"work directory parent must not be a symlink: {parent}")
        parent_stat = parent.stat()
        if parent_stat.st_uid != os.getuid() or parent_stat.st_mode & 0o022:
            raise ValueError(f"work directory parent is not private: {parent}")
        if parent == trusted_home:
            break
    if lexical.exists():
        if not lexical.is_dir():
            raise ValueError("work directory path is not a directory")
        if any(lexical.iterdir()):
            raise ValueError("work directory must be empty for a new run")
    else:
        lexical.mkdir(parents=True, mode=0o700)
    lexical.chmod(0o700)
    return lexical.resolve(strict=True)


def validate_existing_private_work_dir(path: Path) -> Path:
    root = _resolve_private_input(path, "work directory")
    info = root.stat()
    if (
        not root.is_dir()
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("work directory must be user-owned and mode 700")
    return root


def validate_private_epoch_checkpoint(work_dir: Path) -> Path:
    """Resolve a private MMEngine epoch checkpoint without deserializing it."""
    root = _resolve_private_input(work_dir, "resume work directory")
    if not root.is_dir():
        raise ValueError("resume work directory is not a directory")
    root_stat = root.stat()
    if root_stat.st_uid != os.getuid() or root_stat.st_mode & 0o077:
        raise ValueError("resume work directory must be user-owned and private")

    pointer = root / "last_checkpoint"
    if pointer.is_symlink() or not pointer.is_file():
        raise ValueError("last_checkpoint must be a regular non-symlink file")
    pointer_stat = pointer.stat()
    if pointer_stat.st_uid != os.getuid() or pointer_stat.st_mode & 0o077:
        raise ValueError("last_checkpoint must be user-owned and private")
    raw_pointer = pointer.read_bytes()
    checkpoint_bytes = (
        raw_pointer[:-1] if raw_pointer.endswith(b"\n") else raw_pointer)
    if (
        not checkpoint_bytes
        or b"\n" in checkpoint_bytes
        or b"\x00" in checkpoint_bytes
        or any(byte < 0x20 for byte in checkpoint_bytes)
    ):
        raise ValueError("last_checkpoint must contain exactly one safe line")
    try:
        checkpoint_text = checkpoint_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("last_checkpoint is not valid UTF-8") from exc
    if not checkpoint_text:
        raise ValueError("last_checkpoint is empty")

    candidate = Path(checkpoint_text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    lexical = Path(os.path.abspath(candidate))
    if not lexical.is_relative_to(root):
        raise ValueError("resume checkpoint must stay inside its work directory")
    if lexical.parent != root:
        raise ValueError("resume checkpoint must be directly inside its work directory")
    if not re.fullmatch(r"epoch_[1-9][0-9]*\.pth", lexical.name):
        raise ValueError("formal resume requires an epoch-boundary checkpoint")
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError("resume checkpoint must be a regular non-symlink file")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("resume checkpoint must resolve inside its work directory")
    for parent in lexical.parents:
        if parent.is_symlink():
            raise ValueError(f"resume checkpoint parent is symlinked: {parent}")
        if parent == root:
            break
    checkpoint_stat = resolved.stat()
    if (
        checkpoint_stat.st_uid != os.getuid()
        or checkpoint_stat.st_mode & 0o077
        or checkpoint_stat.st_nlink != 1
        or checkpoint_stat.st_size <= 0
    ):
        raise ValueError(
            "resume checkpoint must be private, non-empty, and singly linked")
    if pointer_stat.st_mtime_ns < checkpoint_stat.st_mtime_ns:
        raise ValueError("last_checkpoint predates the checkpoint target")
    provenance = resolved.with_name(resolved.name + ".provenance.json")
    if provenance.is_symlink() or not provenance.is_file():
        raise ValueError("resume checkpoint provenance is missing")
    provenance_stat = provenance.stat()
    if (
        provenance_stat.st_uid != os.getuid()
        or provenance_stat.st_mode & 0o077
        or provenance_stat.st_nlink != 1
    ):
        raise ValueError("resume checkpoint provenance must be private")
    return resolved


def write_checkpoint_provenance(
    checkpoint: Path, identity_path: Path,
) -> Path:
    """Atomically bind a completed epoch checkpoint to its run identity."""
    if checkpoint.is_symlink() or identity_path.is_symlink():
        raise ValueError("checkpoint provenance inputs must not be symlinks")
    checkpoint = checkpoint.resolve(strict=True)
    identity_path = identity_path.resolve(strict=True)
    if checkpoint.parent != identity_path.parent:
        raise ValueError("checkpoint and run identity must share a work directory")
    for path, label in ((checkpoint, "checkpoint"), (identity_path, "identity")):
        info = path.stat()
        if (
            path.is_symlink()
            or not path.is_file()
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
            or info.st_size <= 0
        ):
            raise ValueError(f"{label} must be a private regular file")
    payload = {
        "contract": "ovha_rod_checkpoint_provenance_v1",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "run_identity_sha256": _sha256(identity_path),
    }
    destination = checkpoint.with_name(checkpoint.name + ".provenance.json")
    if destination.exists() or destination.is_symlink():
        raise ValueError("checkpoint provenance already exists")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        rendered = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        os.write(descriptor, rendered.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination


def collect_scalar_diagnostics(
        values: Mapping[str, Any], prefix: str = "") -> tuple[dict[str, float], tuple[str, ...]]:
    finite: dict[str, float] = {}
    nonfinite: list[str] = []
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            number = float(value.detach().cpu())
        elif isinstance(value, (float, int)):
            number = float(value)
        else:
            continue
        name = f"{prefix}{key}"
        if math.isfinite(number):
            finite[name] = number
        else:
            nonfinite.append(name)
    return finite, tuple(sorted(nonfinite))


def audit_smoke_outputs(work_dir: Path, expected_operator: str) -> dict[str, Any]:
    diagnostics_path = work_dir / "operator_diagnostics.jsonl"
    rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    if diagnostics_path.is_file():
        for line_number, line in enumerate(
                diagnostics_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (TypeError, ValueError) as exc:
                parse_errors.append(f"line {line_number}: {exc}")
            else:
                rows.append(row)
    missing_fields = sorted({
        field for row in rows for field in REQUIRED_SMOKE_FIELDS - row.keys()
    })
    row_count_ok = len(rows) == 2
    iteration_ok = [row.get("iteration") for row in rows] == [1, 2]
    operator_ok = bool(rows) and all(
        row.get("seed_operator") == expected_operator for row in rows)
    nonfinite = [
        {"row": index, "keys": row.get("nonfinite_scalar_keys", [])}
        for index, row in enumerate(rows)
        if int(row.get("nonfinite_scalar_count", 0)) != 0
    ]
    numeric_nonfinite = []
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                numeric_nonfinite.append(
                    {"row": row_index, "key": key, "value": value})
    checkpoints = sorted(
        str(path) for path in work_dir.glob("iter_2.pth")
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0)
    active = expected_operator in {"rqgo", "generic"}
    observed_weights = [
        float(row.get("loss_seed_weight", float("nan"))) for row in rows
    ]
    expected_weights = [0.25, 0.5] if active else [0.0, 0.0]
    warmup_ok = (
        len(observed_weights) == 2
        and all(math.isclose(observed, expected, abs_tol=1e-8)
                for observed, expected in zip(
                    observed_weights, expected_weights))
    )
    gradients_ok = (not active) or (
        operator_ok and any(
            float(row.get("seed_gradient_norm", 0.0)) > 0.0 for row in rows))
    bias_ok = bool(rows) and not missing_fields and all(
        float(row["seed_bias_abs_max"]) <= 2.0 + 1e-6
        and float(row["seed_invalid_bias_abs_max"]) == 0.0
        for row in rows)
    ok = all((
        row_count_ok,
        iteration_ok,
        operator_ok,
        not parse_errors,
        not missing_fields,
        not nonfinite,
        not numeric_nonfinite,
        bool(checkpoints),
        gradients_ok,
        bias_ok,
        warmup_ok,
    ))
    return {
        "ok": ok,
        "contract": "ovha_rod_phase1_two_batch_smoke_v2",
        "expected_seed_operator": expected_operator,
        "diagnostic_rows": len(rows),
        "iterations_exact": iteration_ok,
        "operator_rows_exact": operator_ok,
        "missing_fields": missing_fields,
        "parse_errors": parse_errors,
        "nonfinite": [*nonfinite, *numeric_nonfinite],
        "active_seed_gradient_nonzero": gradients_ok,
        "seed_bias_contract": bias_ok,
        "warmup_schedule_exact": warmup_ok,
        "checkpoints": checkpoints,
        "diagnostics": str(diagnostics_path),
    }


def find_remote_weight_values(value: Any) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, str):
        if value.lower().startswith(("http://", "https://")):
            found.append(value)
    elif isinstance(value, Mapping):
        for nested in value.values():
            found.extend(find_remote_weight_values(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            found.extend(find_remote_weight_values(nested))
    return tuple(sorted(set(found)))


def validate_locked_checkpoint(path: Path) -> Path:
    checkpoint = _resolve_private_input(path, "checkpoint")
    file_stat = checkpoint.stat()
    if (
        not checkpoint.is_file()
        or file_stat.st_size != LOCKED_CHECKPOINT_SIZE
        or _sha256(checkpoint) != LOCKED_CHECKPOINT_SHA256
    ):
        raise ValueError("checkpoint does not match locked size and SHA-256")
    if file_stat.st_uid != os.getuid() or file_stat.st_mode & 0o077:
        raise ValueError("checkpoint must be user-owned and private")
    return checkpoint


def validate_local_bert(path: Path) -> Path:
    root = _resolve_private_input(path, "BERT directory")
    if not root.is_dir():
        raise ValueError("BERT directory is missing")
    if (root / "pytorch_model.bin").exists():
        raise ValueError("BERT pytorch_model.bin is forbidden; use safetensors")
    for required in (root / "config.json", root / "vocab.txt"):
        if not required.is_file() or required.is_symlink():
            raise ValueError(f"required BERT file is missing or linked: {required}")
    weights = root / "model.safetensors"
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("BERT model.safetensors is missing or linked")
    weights_stat = weights.stat()
    if (
        weights_stat.st_uid != os.getuid()
        or weights_stat.st_mode & 0o077
        or weights_stat.st_size != LOCKED_BERT_SIZE
        or _sha256(weights) != LOCKED_BERT_SHA256
    ):
        raise ValueError("BERT safetensors does not match the locked artifact")
    return root


def _resolve_private_input(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    trusted_home = Path.home().resolve()
    if not lexical.is_relative_to(trusted_home):
        raise ValueError(f"{label} must stay under {trusted_home}")
    if lexical.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if lexical.exists():
        lexical_stat = lexical.stat()
        if lexical_stat.st_uid != os.getuid() or lexical_stat.st_mode & 0o022:
            raise ValueError(f"{label} is not user-owned/private")
    for parent in lexical.parents:
        if parent.is_symlink():
            raise ValueError(f"{label} parent must not be a symlink: {parent}")
        parent_stat = parent.stat()
        if parent_stat.st_uid != os.getuid() or parent_stat.st_mode & 0o022:
            raise ValueError(f"{label} parent is not private: {parent}")
        if parent == trusted_home:
            break
    return lexical.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
