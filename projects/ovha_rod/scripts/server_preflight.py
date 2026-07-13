#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


EXPECTED_MMDET_COMMIT = "cfd5d3a985b0249de009b67d04f37263e11cdf3d"
EXPECTED_MMDET_TREE = "e389bc213f4772c481a0b5f81f0d1786c0b79e66"
EXPECTED_CHECKPOINT_SHA256 = (
    "b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a")
EXPECTED_CHECKPOINT_SIZE = 1093815743
ENVIRONMENT_PROFILE = "cu121-wheel"
EXPECTED_TORCH_CUDA = "12.1"
EXPECTED_BERT_SAFETENSORS_SHA256 = (
    "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3")
EXPECTED_BERT_SAFETENSORS_SIZE = 440449768
EXPECTED_VERSIONS = {
    "torch": "2.1.0",
    "torchvision": "0.16.0",
    "mmcv": "2.1.0",
    "mmengine": "0.10.3",
    "transformers": "4.36.2",
}
DATASET_FILES = {
    "refcoco": (
        "finetune_refcoco_train_vg.json",
        "finetune_refcoco_val.json",
    ),
    "refcoco_plus": (
        "finetune_refcoco+_train_vg.json",
        "finetune_refcoco+_val.json",
    ),
    "refcocog": (
        "finetune_refcocog_train_vg.json",
        "finetune_refcocog_val.json",
    ),
}
CONFIG_FILES = {
    "refcoco": "ovha_rod_swin_t_5e_refcoco.py",
    "refcoco_plus": "ovha_rod_swin_t_5e_refcoco_plus.py",
    "refcocog": "ovha_rod_swin_t_5e_refcocog.py",
}
FORBIDDEN_CONFIG_MARKERS = (
    "testa",
    "testb",
    "region_features.npy",
    "proposal_cache",
    "clip crop",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    observed: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-fast audit for the pinned OVHA-ROD Phase 1 server, "
            "training data, checkpoint, local BERT, and val-only config."
        ))
    parser.add_argument("--dataset", choices=tuple(DATASET_FILES), required=True)
    parser.add_argument("--mmdet-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bert-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="Required trusted lowercase SHA-256 for the pretrained checkpoint.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--work-root", type=Path, default=Path("work_dirs"))
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = run_checks(args)
    payload = {
        "ok": all(check.ok for check in checks),
        "contract": "ovha_rod_phase1_server_preflight_v1",
        "environment_profile": ENVIRONMENT_PROFILE,
        "dataset": args.dataset,
        "expected_mmdetection_commit": EXPECTED_MMDET_COMMIT,
        "checks": [asdict(check) for check in checks],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        work_root = args.work_root.expanduser().resolve()
        output_lexical = Path(os.path.abspath(args.output.expanduser()))
        if output_lexical.is_symlink():
            raise ValueError("preflight output must not be a symlink")
        output = output_lexical.resolve()
        if not output.is_relative_to(work_root):
            raise ValueError("preflight output must stay inside work root")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    return 0 if payload["ok"] else 2


def run_checks(args: argparse.Namespace) -> tuple[CheckResult, ...]:
    project_root = Path(__file__).resolve().parents[1]
    mmdet_root = args.mmdet_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    checkpoint_input = args.checkpoint.expanduser()
    try:
        checkpoint = _resolve_trusted_checkpoint(checkpoint_input)
    except ValueError:
        checkpoint = Path(os.path.abspath(checkpoint_input))
    bert_root = args.bert_root.expanduser()

    checks: list[CheckResult] = []
    checks.extend(_python_checks())
    trust_checks = (
        *_mmdet_checks(mmdet_root),
        *_project_source_checks(project_root),
    )
    checks.extend(trust_checks)
    if not all(check.ok for check in trust_checks):
        return tuple(checks)
    checkpoint_checks = _checkpoint_checks(
        checkpoint_input, args.checkpoint_sha256)
    checkpoint_trusted = all(check.ok for check in checkpoint_checks)
    checks.extend(_package_checks(mmdet_root))
    checks.extend(checkpoint_checks)
    bert_checks = _bert_checks(bert_root)
    checks.extend(bert_checks)
    checks.extend(_data_checks(data_root, args.dataset))
    checks.extend(_storage_checks(args.work_root, args.min_free_gb))
    if not checkpoint_trusted or not all(check.ok for check in bert_checks):
        return tuple(checks)
    checks.extend(_project_checks(
        project_root, args.dataset, data_root, bert_root,
        checkpoint, checkpoint_trusted))
    checks.extend(_cuda_checks())
    return tuple(checks)


def _storage_checks(work_root: Path, min_free_gb: float) -> tuple[CheckResult, ...]:
    root = work_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".ovha-rod-", delete=True) as probe:
            probe.write(b"ok\n")
            probe.flush()
        writable = True
        detail = "work root is writable"
    except OSError as exc:
        writable = False
        detail = str(exc)
    free_gb = shutil.disk_usage(root).free / (1024 ** 3)
    return (
        CheckResult("work_root_writable", writable, detail, str(root)),
        CheckResult(
            "work_root_free_space",
            free_gb >= min_free_gb,
            f"at least {min_free_gb:.1f} GiB free is required",
            round(free_gb, 2),
        ),
    )


def _python_checks() -> tuple[CheckResult, ...]:
    version = sys.version_info
    supported = version[:2] == (3, 10)
    return (
        CheckResult(
            "python_version",
            supported,
            "Python 3.10 is required by the cp310 cu121-wheel profile.",
            ".".join(str(part) for part in version[:3]),
        ),
    )


def _package_checks(mmdet_root: Path) -> tuple[CheckResult, ...]:
    checks: list[CheckResult] = []
    for package in ("torch", "torchvision", "mmcv", "mmengine", "mmdet", "transformers"):
        try:
            module = importlib.import_module(package)
        except Exception as exc:  # pragma: no cover - server-specific imports
            checks.append(CheckResult(f"import_{package}", False, str(exc)))
            continue
        version = str(getattr(module, "__version__", "unknown"))
        expected = EXPECTED_VERSIONS.get(package)
        checks.append(
            CheckResult(
                f"import_{package}",
                expected is None or version.split("+", 1)[0] == expected,
                "package imports and matches the lock" if expected else "package imports",
                version,
            ))
        if package == "mmdet":
            imported_path = Path(module.__file__).resolve()
            checks.append(CheckResult(
                "mmdetection_import_path",
                imported_path.is_relative_to((mmdet_root / "mmdet").resolve()),
                "imported mmdet must come from the pinned checkout",
                str(imported_path),
            ))
    try:
        from mmcv.ops import (MultiScaleDeformableAttention,  # noqa: F401
                              get_compiling_cuda_version)
    except Exception as exc:  # pragma: no cover - server-specific extension
        checks.append(CheckResult("mmcv_deformable_ops", False, str(exc)))
    else:
        checks.append(CheckResult("mmcv_deformable_ops", True, "compiled MMCV op imports"))
        compiled_cuda = str(get_compiling_cuda_version())
        checks.append(CheckResult(
            "mmcv_compiled_cuda",
            ".".join(compiled_cuda.split(".")[:2]) == EXPECTED_TORCH_CUDA,
            "MMCV must be the official CUDA 12.1 binary wheel",
            compiled_cuda,
        ))
    try:
        import torch
        torch_cuda = str(torch.version.cuda)
    except Exception as exc:  # pragma: no cover - server-specific import
        checks.append(CheckResult("torch_cuda_runtime", False, str(exc)))
    else:
        checks.append(CheckResult(
            "torch_cuda_runtime",
            torch_cuda == EXPECTED_TORCH_CUDA,
            "PyTorch must use the locked CUDA 12.1 runtime",
            torch_cuda,
        ))
    return tuple(checks)


def _mmdet_checks(mmdet_root: Path) -> tuple[CheckResult, ...]:
    trust = _executable_tree_check(mmdet_root, "mmdetection_source_private")
    if not trust.ok:
        return (trust,)
    if not (mmdet_root / ".git").is_dir():
        return (CheckResult("mmdetection_checkout", False, "missing Git checkout", str(mmdet_root)),)
    commit = _git_output(mmdet_root, "rev-parse", "HEAD")
    tree = _git_output(mmdet_root, "rev-parse", "HEAD^{tree}")
    origin = _git_output(mmdet_root, "remote", "get-url", "origin")
    dirty = _git_output(mmdet_root, "status", "--porcelain", "--untracked-files=all")
    allowed_origins = {
        "https://github.com/open-mmlab/mmdetection.git",
        "git@github.com:open-mmlab/mmdetection.git",
    }
    return (trust,
        CheckResult(
            "mmdetection_commit",
            commit == EXPECTED_MMDET_COMMIT,
            "checkout must be pinned exactly",
            commit,
        ),
        CheckResult(
            "mmdetection_tree",
            tree == EXPECTED_MMDET_TREE,
            "checkout tree must match the audited tree exactly",
            tree,
        ),
        CheckResult(
            "mmdetection_clean",
            dirty == "",
            "checkout must have no tracked, staged, or untracked changes",
            "clean" if not dirty else "dirty",
        ),
        CheckResult(
            "mmdetection_origin",
            origin in allowed_origins,
            "checkout must come from the official repository",
            "official" if origin in allowed_origins else "unexpected",
        ),
    )


def _project_source_checks(project_root: Path) -> tuple[CheckResult, ...]:
    trust = _executable_tree_check(project_root, "project_source_private")
    if not trust.ok:
        return (trust,)
    dirty = _git_output(
        project_root, "status", "--porcelain", "--untracked-files=all")
    protected_roots = (
        project_root / "configs",
        project_root / "ovha_rod",
        project_root / "scripts",
    )
    unsafe: list[str] = []
    for root in protected_roots:
        for path in (root, *root.rglob("*")):
            try:
                stat = path.lstat()
            except OSError as exc:
                unsafe.append(f"{path}: {exc}")
                continue
            if path.is_symlink():
                continue
            if stat.st_uid != os.getuid() or stat.st_mode & 0o022:
                unsafe.append(str(path))
    return (
        trust,
        CheckResult(
            "project_git_clean",
            dirty == "",
            "reviewed project worktree must be clean before execution",
            "clean" if not dirty else dirty[:2000],
        ),
        CheckResult(
            "project_runtime_paths_private",
            not unsafe,
            "configs, scripts, and package must be user-owned and not group/other writable",
            unsafe[:50],
        ),
    )


def _executable_tree_check(root: Path, name: str) -> CheckResult:
    try:
        resolved = _resolve_private_file(root)
    except (OSError, ValueError) as exc:
        return CheckResult(name, False, str(exc), str(root))
    if not resolved.is_dir():
        return CheckResult(name, False, "executable root is not a directory", str(resolved))
    unsafe: list[str] = []
    for path in (resolved, *resolved.rglob("*")):
        try:
            stat = path.lstat()
        except OSError as exc:
            unsafe.append(f"{path}: {exc}")
            continue
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
                target_stat = target.stat()
            except OSError:
                unsafe.append(str(path))
                continue
            internal = target == resolved or target.is_relative_to(resolved)
            target_private = (
                target_stat.st_uid == os.getuid()
                and not target_stat.st_mode & 0o022
            )
            if not internal or not target_private:
                unsafe.append(str(path))
            continue
        if stat.st_uid != os.getuid() or stat.st_mode & 0o022:
            unsafe.append(str(path))
    return CheckResult(
        name,
        not unsafe,
        "executable tree must be private; links may only resolve inside the same trusted root",
        unsafe[:50],
    )


def _project_checks(project_root: Path, dataset: str, data_root: Path,
                    bert_root: Path, checkpoint: Path,
                    checkpoint_trusted: bool) -> tuple[CheckResult, ...]:
    config = project_root / "configs" / CONFIG_FILES[dataset]
    if not config.is_file():
        return (CheckResult("phase1_config", False, "missing config", str(config)),)
    source = config.read_text()
    lowered = source.lower()
    forbidden = [marker for marker in FORBIDDEN_CONFIG_MARKERS if marker in lowered]
    checks = [
        CheckResult("phase1_config", True, "config exists", str(config)),
        CheckResult(
            "val_only_config",
            not forbidden,
            "config must not name held-out or offline-candidate artifacts",
            forbidden,
        ),
        CheckResult(
            "five_epoch_budget",
            "max_epochs=5" in source.replace(" ", ""),
            "strict Phase 1 budget is five epochs",
        ),
        CheckResult(
            "no_separate_freeze_schedule",
            "freeze" not in lowered,
            "main protocol jointly optimizes the configured parameter groups",
        ),
    ]
    try:
        module = importlib.import_module("ovha_rod")
    except Exception as exc:  # pragma: no cover - depends on completed project package
        checks.append(CheckResult("import_ovha_rod", False, str(exc)))
    else:
        checks.append(CheckResult("import_ovha_rod", True, "project registry imports", str(module.__file__)))
    try:
        import torch
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet.registry import METRICS, MODELS
        from ovha_rod.runtime_contracts import find_remote_weight_values

        resolved = Config.fromfile(config)
        resolved.model.language_model.name = str(bert_root)
        remote_weights = find_remote_weight_values({
            "model": resolved.model,
            "load_from": resolved.get("load_from"),
        })
        if remote_weights:
            raise ValueError(
                f"resolved config contains remote model weights: {remote_weights}")
        init_default_scope("mmdet")
        model = MODELS.build(resolved.model)
        resolved.val_evaluator.ann_file = str(
            data_root / "mdetr_annotations" / DATASET_FILES[dataset][1])
        METRICS.build(resolved.val_evaluator)
    except Exception as exc:  # pragma: no cover - depends on installed MMDetection
        checks.append(CheckResult("resolve_phase1_config", False, str(exc)))
    else:
        checks.append(
            CheckResult(
                "resolve_phase1_config",
                resolved.model.get("type") == "OVHAGroundingDINO",
                "MMEngine resolves the pinned base and OVHA detector override",
                resolved.model.get("type"),
            ))
        checks.append(CheckResult(
            "build_phase1_model", True,
            "pinned config builds detector/head/metric registries"))
        checks.append(CheckResult(
            "no_remote_model_weights", True,
            "resolved model init and load sources are local-only"))
        positional_type = type(model.positional_encoding).__name__
        positional_ok = (
            positional_type == "DeterministicSinePositionalEncoding")
        checks.append(CheckResult(
            "deterministic_positional_encoding",
            positional_ok,
            "padded feature masks must avoid floating CUDA cumsum",
            positional_type,
        ))
        try:
            positional_mask = torch.tensor([
                [[False, False, True],
                 [False, True, True]],
            ])
            with torch.no_grad():
                positional_probe = model.positional_encoding(positional_mask)
            positional_probe_ok = (
                positional_type == "DeterministicSinePositionalEncoding"
                and tuple(positional_probe.shape) == (
                    1, int(model.embed_dims), 2, 3)
                and bool(torch.isfinite(positional_probe).all())
            )
        except Exception as exc:
            checks.append(CheckResult(
                "deterministic_positional_encoding_probe", False, str(exc)))
        else:
            checks.append(CheckResult(
                "deterministic_positional_encoding_probe",
                positional_probe_ok,
                "irregular padded mask produces finite parent-shaped encoding",
                {
                    "type": positional_type,
                    "shape": list(positional_probe.shape),
                    "finite": bool(torch.isfinite(positional_probe).all()),
                },
            ))
        zero_tensors = [
            model.bbox_head.referent_head[-1].weight,
            model.bbox_head.referent_head[-1].bias,
            model.seed_operator.seed_head[-1].weight,
            model.seed_operator.seed_head[-1].bias,
        ]
        zero_init = all(torch.count_nonzero(tensor).item() == 0
                        for tensor in zero_tensors)
        checks.append(CheckResult(
            "zero_initialized_residual_heads", zero_init,
            "seed and referent output heads must be exactly zero initialized"))
        try:
            model.eval()
            token_count = max(int(model.num_queries), 1)
            grid_side = math.isqrt(token_count)
            if grid_side * grid_side != token_count:
                raise ValueError(
                    "pre-decoder probe requires a square query grid")
            embed_dims = int(model.embed_dims)
            probe_memory = torch.zeros(1, token_count, embed_dims)
            probe_spatial_shapes = torch.tensor(
                [[grid_side, grid_side]], dtype=torch.long)
            probe_text = torch.zeros(1, 4, embed_dims)
            probe_text_mask = torch.ones(1, 4, dtype=torch.bool)
            with torch.no_grad():
                _, probe_head_inputs = model.pre_decoder(
                    memory=probe_memory,
                    memory_mask=None,
                    spatial_shapes=probe_spatial_shapes,
                    memory_text=probe_text,
                    text_token_mask=probe_text_mask,
                    batch_data_samples=None,
                )
            probe_valid = probe_head_inputs["seed_valid"]
            probe_bias = probe_head_inputs["seed_bias"]
            probe_ok = (
                tuple(probe_valid.shape) == (1, token_count)
                and bool(probe_valid.all())
                and bool(torch.isfinite(probe_bias).all())
            )
        except Exception as exc:
            checks.append(CheckResult(
                "pre_decoder_none_memory_mask", False, str(exc)))
        else:
            checks.append(CheckResult(
                "pre_decoder_none_memory_mask",
                probe_ok,
                "real Phase 1 pre-decoder accepts the official None mask",
                {
                    "tokens": token_count,
                    "all_valid": bool(probe_valid.all()),
                    "finite_bias": bool(torch.isfinite(probe_bias).all()),
                },
            ))
        if not checkpoint_trusted:
            checks.append(CheckResult(
                "checkpoint_base_key_coverage", False,
                "checkpoint digest must pass before safe deserialization"))
        else:
            try:
                payload = _load_locked_checkpoint(
                    checkpoint, checkpoint_trusted=checkpoint_trusted)
                state = payload.get("state_dict", payload)
                state_keys = {
                    key.removeprefix("module.") for key in state.keys()
                }
                model_keys = set(model.state_dict())
                new_prefixes = (
                    "role_encoder.", "seed_operator.",
                    "bbox_head.referent_head.")
                base_keys = {
                    key for key in model_keys
                    if not key.startswith(new_prefixes)
                }
                missing = sorted(base_keys - state_keys)
                backbone_keys = {
                    key for key in model_keys if key.startswith("backbone.")
                }
                missing_backbone = sorted(backbone_keys - state_keys)
                coverage = 1.0 - len(missing) / max(len(base_keys), 1)
                checks.append(CheckResult(
                    "checkpoint_backbone_key_coverage",
                    not missing_backbone,
                    "every backbone parameter key must be present",
                    {
                        "coverage": (
                            1.0 - len(missing_backbone)
                            / max(len(backbone_keys), 1)),
                        "missing_count": len(missing_backbone),
                        "missing_preview": missing_backbone[:20],
                    },
                ))
                checks.append(CheckResult(
                    "checkpoint_base_key_coverage",
                    coverage >= 0.99,
                    "at least 99% of parent parameter keys must be present",
                    {
                        "coverage": round(coverage, 6),
                        "missing_count": len(missing),
                        "missing_preview": missing[:20],
                    },
                ))
            except Exception as exc:
                checks.append(CheckResult(
                    "checkpoint_base_key_coverage", False, str(exc)))
    return tuple(checks)


def _data_checks(data_root: Path, dataset: str) -> tuple[CheckResult, ...]:
    image_root = data_root / "train2014"
    annotation_root = data_root / "mdetr_annotations"
    required = tuple(annotation_root / name for name in DATASET_FILES[dataset])
    image_exists = image_root.is_dir() and _contains_jpeg(image_root)
    checks = [
        CheckResult(
            "coco_train2014_images",
            image_exists,
            "train2014 must contain at least one JPEG",
            str(image_root),
        )
    ]
    checks.extend(
        CheckResult(
            f"annotation_{path.name}",
            path.is_file() and path.stat().st_size > 0,
            "required training/validation annotation",
            str(path),
        )
        for path in required
    )
    return tuple(checks)


def _resolve_trusted_checkpoint(path: Path) -> Path:
    return _resolve_private_file(path)


def _load_locked_checkpoint(
        path: Path, checkpoint_trusted: bool) -> Any:
    if not checkpoint_trusted:
        raise ValueError(
            "checkpoint trust checks must pass before compatibility load")
    trusted_path = _resolve_trusted_checkpoint(path)
    with trusted_path.open("rb") as handle:
        file_stat = os.fstat(handle.fileno())
        if file_stat.st_uid != os.getuid() or file_stat.st_mode & 0o077:
            raise ValueError("checkpoint permissions changed before load")
        if file_stat.st_size != EXPECTED_CHECKPOINT_SIZE:
            raise ValueError("checkpoint size changed before load")
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("checkpoint digest changed before load")
        handle.seek(0)
        import torch
        return torch.load(handle, map_location="cpu", weights_only=False)


def _resolve_private_file(path: Path) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    trusted_home = Path.home().resolve()
    if not lexical.is_relative_to(trusted_home):
        raise ValueError(f"trusted file must stay under the user home: {trusted_home}")
    if lexical.is_symlink():
        raise ValueError("trusted file must not be a symlink")
    for parent in lexical.parents:
        if parent.is_symlink():
            raise ValueError(f"trusted parent must not be a symlink: {parent}")
        stat = parent.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o022:
            raise ValueError(f"trusted parent is not private: {parent}")
        if parent == trusted_home:
            break
    return lexical.resolve(strict=False)


def _checkpoint_checks(path: Path, expected_sha256: str | None) -> tuple[CheckResult, ...]:
    try:
        trusted_path = _resolve_trusted_checkpoint(path)
    except ValueError as exc:
        return (CheckResult(
            "pretrained_checkpoint", False, str(exc), str(path)),)
    exists = trusted_path.is_file() and trusted_path.stat().st_size > 0
    checks = [CheckResult(
        "pretrained_checkpoint", exists,
        "checkpoint exists, is non-empty, and has no symlink component",
        str(trusted_path))]
    if exists:
        stat = trusted_path.stat()
        checks.append(CheckResult(
            "checkpoint_size",
            stat.st_size == EXPECTED_CHECKPOINT_SIZE,
            "checkpoint byte size must match the locked official artifact",
            stat.st_size,
        ))
        private = stat.st_uid == os.getuid() and (stat.st_mode & 0o077) == 0
        checks.append(CheckResult(
            "checkpoint_private_permissions",
            private,
            "checkpoint must be user-owned and inaccessible to group/other",
            oct(stat.st_mode & 0o777),
        ))
    if expected_sha256 is not None:
        normalized = expected_sha256.strip().lower()
        valid_expected = len(normalized) == 64 and all(char in "0123456789abcdef" for char in normalized)
        checks.append(CheckResult("checkpoint_sha256_format", valid_expected, "expected SHA-256 must be 64 hex digits"))
        checks.append(CheckResult(
            "checkpoint_sha256_lock",
            normalized == EXPECTED_CHECKPOINT_SHA256,
            "provided digest must equal the version-controlled official digest"))
        if exists and valid_expected:
            observed = _sha256(trusted_path)
            checks.append(CheckResult(
                "checkpoint_sha256",
                observed == normalized == EXPECTED_CHECKPOINT_SHA256,
                "checkpoint digest matches the locked official artifact",
                observed))
    return tuple(checks)


def _bert_checks(root: Path) -> tuple[CheckResult, ...]:
    safetensors = root / "model.safetensors"
    pytorch_weights = root / "pytorch_model.bin"
    required = (root / "config.json", root / "vocab.txt")
    try:
        trusted_safetensors = _resolve_private_file(safetensors)
    except ValueError as exc:
        trusted_safetensors = None
        weights_detail = str(exc)
    else:
        weights_detail = str(trusted_safetensors)
    weights_ok = bool(
        trusted_safetensors is not None
        and trusted_safetensors.is_file()
        and trusted_safetensors.stat().st_size == EXPECTED_BERT_SAFETENSORS_SIZE
        and _sha256(trusted_safetensors) == EXPECTED_BERT_SAFETENSORS_SHA256
        and trusted_safetensors.stat().st_uid == os.getuid()
        and (trusted_safetensors.stat().st_mode & 0o077) == 0
        and not pytorch_weights.exists()
    )
    checks = [
        CheckResult(
            "bert_directory",
            root.is_dir(),
            "BERT must be stored locally for deterministic server runs",
            str(root),
        ),
        CheckResult(
            "bert_config_tokenizer",
            all(path.is_file() for path in required),
            "local BERT requires config.json and vocab.txt",
            [str(path) for path in required],
        ),
        CheckResult(
            "bert_weights",
            weights_ok,
            "local BERT requires locked safetensors and no pytorch_model.bin",
            weights_detail,
        ),
    ]
    return tuple(checks)


def _cuda_checks() -> tuple[CheckResult, ...]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - handled by package checks too
        return (CheckResult("cuda_available", False, str(exc)),)
    available = bool(torch.cuda.is_available())
    checks = [
        CheckResult(
            "cuda_available",
            available,
            "CUDA is mandatory for MM-Grounding-DINO training",
            {"torch_cuda": torch.version.cuda, "device_count": torch.cuda.device_count()},
        )
    ]
    if available:
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": properties.name,
                "total_memory": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            })
        checks.append(CheckResult("cuda_devices", bool(devices), "visible CUDA devices", devices))
    return tuple(checks)


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *args),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()


def _contains_jpeg(root: Path) -> bool:
    for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        if next(root.glob(pattern), None) is not None:
            return True
    return False


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
