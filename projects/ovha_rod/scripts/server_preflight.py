#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
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
EXPECTED_VERSIONS = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
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
        "dataset": args.dataset,
        "expected_mmdetection_commit": EXPECTED_MMDET_COMMIT,
        "checks": [asdict(check) for check in checks],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        work_root = args.work_root.expanduser().resolve()
        output = args.output.expanduser().resolve()
        if not output.is_relative_to(work_root):
            raise ValueError("preflight output must stay inside work root")
        if output.is_symlink():
            raise ValueError("preflight output must not be a symlink")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    return 0 if payload["ok"] else 2


def run_checks(args: argparse.Namespace) -> tuple[CheckResult, ...]:
    project_root = Path(__file__).resolve().parents[1]
    mmdet_root = args.mmdet_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    bert_root = args.bert_root.expanduser().resolve()

    checks: list[CheckResult] = []
    checks.extend(_python_checks())
    checkpoint_checks = _checkpoint_checks(checkpoint, args.checkpoint_sha256)
    checkpoint_trusted = all(check.ok for check in checkpoint_checks)
    checks.extend(_package_checks(mmdet_root))
    checks.extend(_mmdet_checks(mmdet_root))
    checks.extend(checkpoint_checks)
    checks.extend(_project_checks(
        project_root, args.dataset, data_root, bert_root,
        checkpoint, checkpoint_trusted))
    checks.extend(_data_checks(data_root, args.dataset))
    checks.extend(_bert_checks(bert_root))
    checks.extend(_cuda_checks())
    checks.extend(_storage_checks(args.work_root, args.min_free_gb))
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
    supported = (3, 9) <= version[:2] <= (3, 11)
    return (
        CheckResult(
            "python_version",
            supported,
            "Python 3.9--3.11 is required by the locked stack.",
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
        from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401
    except Exception as exc:  # pragma: no cover - server-specific extension
        checks.append(CheckResult("mmcv_deformable_ops", False, str(exc)))
    else:
        checks.append(CheckResult("mmcv_deformable_ops", True, "compiled MMCV op imports"))
    return tuple(checks)


def _mmdet_checks(mmdet_root: Path) -> tuple[CheckResult, ...]:
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
    return (
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

        resolved = Config.fromfile(config)
        resolved.model.language_model.name = str(bert_root)
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
        if not checkpoint_trusted:
            checks.append(CheckResult(
                "checkpoint_base_key_coverage", False,
                "checkpoint digest must pass before safe deserialization"))
        else:
            try:
                payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=True)
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
                coverage = 1.0 - len(missing) / max(len(base_keys), 1)
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


def _checkpoint_checks(path: Path, expected_sha256: str | None) -> tuple[CheckResult, ...]:
    exists = path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    checks = [CheckResult(
        "pretrained_checkpoint", exists,
        "checkpoint exists, is non-empty, and is not a symlink", str(path))]
    if expected_sha256 is not None:
        normalized = expected_sha256.strip().lower()
        valid_expected = len(normalized) == 64 and all(char in "0123456789abcdef" for char in normalized)
        checks.append(CheckResult("checkpoint_sha256_format", valid_expected, "expected SHA-256 must be 64 hex digits"))
        checks.append(CheckResult(
            "checkpoint_sha256_lock",
            normalized == EXPECTED_CHECKPOINT_SHA256,
            "provided digest must equal the version-controlled official digest"))
        if exists and valid_expected:
            observed = _sha256(path)
            checks.append(CheckResult(
                "checkpoint_sha256",
                observed == normalized == EXPECTED_CHECKPOINT_SHA256,
                "checkpoint digest matches the locked official artifact",
                observed))
    return tuple(checks)


def _bert_checks(root: Path) -> tuple[CheckResult, ...]:
    alternatives = (root / "model.safetensors", root / "pytorch_model.bin")
    required = (root / "config.json", root / "vocab.txt")
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
            any(path.is_file() and path.stat().st_size > 0 for path in alternatives),
            "local BERT requires safetensors or PyTorch weights",
            [str(path) for path in alternatives],
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
