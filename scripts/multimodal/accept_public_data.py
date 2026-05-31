#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
from moat_ovha_torch.data.multimodal.adapters.base import MissingMultimodalDataError
from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout
from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements
from moat_ovha_torch.models.multimodal.baselines import assert_same_feature_baseline_policy
from scripts.multimodal.build_cache import ADAPTERS, _cache_splits_for
from scripts.multimodal.run_public_smoke import _replace_cache_root, _run_public_training_smoke


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Accept staged public multimodal raw data by building the formal cache, "
            "validating controlled-entry evidence, and optionally running a T5 public smoke."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--controlled-report", type=Path, required=True)
    parser.add_argument("--train-smoke-steps", type=int, default=0)
    parser.add_argument("--train-baseline-smoke-steps", type=int, default=0)
    parser.add_argument("--train-all-config-seeds", action="store_true")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-smoke-split")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--memory-tokens", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        payload, exit_code = accept_public_data(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "ok": False,
            "mode": "public_data_acceptance",
            "policy": "fail-fast: public data acceptance must be reproducible before public training",
            "errors": [str(exc)],
            "warnings": [],
        }
        exit_code = 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def accept_public_data(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = MultimodalExperimentConfig.from_file(args.config)
    cache_root = args.cache_root or config.cache_root
    config = _replace_cache_root(config, cache_root)
    assert_same_feature_baseline_policy(config)
    phases: dict[str, dict[str, Any]] = {}

    adapter_type = ADAPTERS.get(config.dataset_name)
    if adapter_type is None:
        return _failure(
            config,
            phases,
            phase="raw_manifest",
            errors=[f"unsupported public dataset adapter: {config.dataset_name}"],
        )
    adapter = adapter_type()

    try:
        manifest = adapter.discover_raw(args.raw_root)
    except MissingMultimodalDataError as exc:
        return _failure(config, phases, phase="raw_manifest", errors=[str(exc)])
    phases["raw_manifest"] = {
        "ok": True,
        "dataset_name": manifest.dataset_name,
        "raw_root": str(manifest.raw_root),
        "file_count": len(manifest.files),
        "files": sorted(manifest.files),
    }

    layout = MultimodalCacheLayout(config.cache_root, adapter.name, config.cache_version)
    try:
        for split in _cache_splits_for(adapter.name):
            adapter.write_cache(manifest, config.cache_root, split, config.cache_version)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _failure(config, phases, phase="cache", errors=[str(exc)])

    validation = adapter.validate_cache(config.cache_root, config.cache_version)
    phases["cache"] = {
        "ok": validation.ok,
        "cache_root": str(layout.root),
        "version": config.cache_version,
        "splits": list(_cache_splits_for(adapter.name)),
        "errors": validation.errors,
        "warnings": validation.warnings,
    }
    if not validation.ok:
        return _failure(config, phases, phase="cache", errors=validation.errors, warnings=validation.warnings)

    controlled_report = json.loads(args.controlled_report.read_text())
    entry_report = validate_public_entry_requirements(
        config.task_type,
        controlled_report,
        require_artifact_files=True,
    )
    phases["public_entry"] = {
        "ok": entry_report.ok,
        "controlled_report": str(args.controlled_report),
        "errors": entry_report.errors,
        "warnings": entry_report.warnings,
    }
    if not entry_report.ok:
        return _failure(config, phases, phase="public_entry", errors=entry_report.errors, warnings=entry_report.warnings)

    if int(args.train_smoke_steps) > 0:
        smoke_payload = _public_smoke_payload(config, layout, args)
        smoke_training = smoke_payload["training"]
        phases["public_smoke"] = {
            "ok": bool(smoke_payload["ok"]),
            "seed_count": len(smoke_training["seeds"]),
            "seeds": list(smoke_training["seeds"]),
            "optimizer_steps": int(smoke_training["optimizer_steps"]),
            "optimizer_steps_per_seed": int(smoke_training["optimizer_steps_per_seed"]),
            "eval_smoke_rows": int(smoke_training["eval_smoke_rows"]),
            "eval_smoke_baseline_rows": int(smoke_training["eval_smoke_baseline_rows"]),
            "baseline_training_status": str(smoke_training["baseline_training_status"]),
            "baseline_smoke_training_steps": int(smoke_training["baseline_smoke_training_steps"]),
            "baseline_optimizer_steps": int(smoke_training["baseline_optimizer_steps"]),
            "artifacts": dict(smoke_training["artifacts"]),
            "artifact_root": str(args.artifact_root) if args.artifact_root else None,
        }
        if args.artifact_root is not None:
            args.artifact_root.mkdir(parents=True, exist_ok=True)
            (args.artifact_root / "public_acceptance_smoke_payload.json").write_text(
                json.dumps(smoke_payload, indent=2, sort_keys=True) + "\n"
            )
        if not smoke_payload["ok"]:
            return _failure(config, phases, phase="public_smoke", errors=["public smoke did not pass optimizer checks"])
    else:
        phases["public_smoke"] = {"ok": True, "skipped": True, "reason": "train-smoke-steps is 0"}

    payload = {
        "ok": True,
        "mode": "public_data_acceptance",
        "policy": "raw manifest, formal cache, controlled entry, and public smoke acceptance completed",
        "config": config.name,
        "dataset_name": config.dataset_name,
        "task_type": config.task_type,
        "phases": phases,
    }
    if args.artifact_root is not None:
        args.artifact_root.mkdir(parents=True, exist_ok=True)
        (args.artifact_root / "public_acceptance_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return payload, 0


def _public_smoke_payload(
    config: MultimodalExperimentConfig,
    layout: MultimodalCacheLayout,
    args: argparse.Namespace,
) -> dict[str, Any]:
    training = _run_public_training_smoke(config, layout, args)
    return {
        "ok": training["ok"],
        "mode": "public_trained_smoke",
        "policy": "cache and controlled gates validated; public T5 train smoke executed",
        "config": config.name,
        "public_entry": "controlled go/no-go report validated",
        "baselines": config.baseline_names,
        "seeds": config.seeds,
        "training": training,
    }


def _failure(
    config: MultimodalExperimentConfig,
    phases: dict[str, dict[str, Any]],
    *,
    phase: str,
    errors: list[str],
    warnings: list[str] | None = None,
) -> tuple[dict[str, Any], int]:
    return (
        {
            "ok": False,
            "mode": "public_data_acceptance",
            "policy": "fail-fast: public data acceptance must pass before public multimodal training",
            "config": config.name,
            "dataset_name": config.dataset_name,
            "task_type": config.task_type,
            "failed_phase": phase,
            "phases": phases,
            "errors": errors,
            "warnings": warnings or [],
        },
        2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
