#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
from moat_ovha_torch.data.multimodal.cache_schema import file_sha256


_SMOKE_RE = re.compile(r"(smoke|not[_-]?topconf|preview[_-]?only)", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that public multimodal main artifacts match a non-smoke "
            "5-seed public main config before top-conference gate bundling."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-metrics", type=Path, action="append", required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--robustness-rows", type=Path, required=True)
    parser.add_argument("--external-sota-references", type=Path, action="append", default=[])
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    try:
        payload, exit_code = validate_public_main_artifacts(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "ok": False,
            "mode": "public_main_artifact_validation",
            "policy": "fail-fast: public main artifacts must be readable JSON/JSONL and config-aligned",
            "errors": [str(exc)],
            "warnings": [],
        }
        exit_code = 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def validate_public_main_artifacts(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = MultimodalExperimentConfig.from_file(args.config)
    raw_rows = _read_jsonl_many(args.raw_metrics)
    diagnostics_rows = _read_jsonl(args.diagnostics)
    robustness_rows = _read_jsonl(args.robustness_rows)
    external_reference_summary = _external_reference_summary(args.external_sota_references)
    errors: list[str] = []
    warnings: list[str] = []

    _validate_non_smoke_config(args.config, config, errors)
    _validate_rows_not_smoke("raw_metrics", raw_rows, errors)
    _validate_rows_not_smoke("diagnostics", diagnostics_rows, errors)
    _validate_rows_not_smoke("robustness_rows", robustness_rows, errors)
    coverage = _validate_raw_metric_coverage(config, raw_rows, split=args.split, errors=errors)
    _validate_diagnostic_seed_coverage(config, diagnostics_rows, split=args.split, errors=errors)
    _validate_robustness_rows(config, robustness_rows, errors)

    payload = {
        "ok": not errors,
        "mode": "public_main_artifact_validation",
        "policy": (
            "public main artifacts must cover the configured seeds and same-feature "
            "model set without smoke/not-topconf markers"
        ),
        "config": str(args.config),
        "dataset": config.dataset_name,
        "task": config.task_type,
        "split": args.split,
        "coverage": coverage,
        "artifacts": {
            "raw_metrics": [_artifact_descriptor(path) for path in args.raw_metrics],
            "diagnostics": _artifact_descriptor(args.diagnostics),
            "robustness_rows": _artifact_descriptor(args.robustness_rows),
        },
        "external_sota_references": external_reference_summary,
        "errors": errors,
        "warnings": warnings,
    }
    return payload, 0 if payload["ok"] else 2


def _external_reference_summary(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        return {
            "provided": False,
            "policy": "external SOTA references are not part of same-feature model coverage",
            "paths": [],
            "reference_count": 0,
        }
    references: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload.get("references", []):
            if not isinstance(row, dict):
                continue
            references.append(
                {
                    "name": str(row.get("name", "")),
                    "dataset": str(row.get("dataset", "")),
                    "evidence_type": str(row.get("evidence_type", "")),
                }
            )
    return {
        "provided": True,
        "policy": "external SOTA references are not part of same-feature model coverage",
        "paths": [str(path) for path in paths],
        "reference_count": len(references),
        "references": references,
    }


def _validate_non_smoke_config(path: Path, config: MultimodalExperimentConfig, errors: list[str]) -> None:
    if _is_smoke_text(path.name):
        errors.append(f"public main config path must not be smoke-scoped: {path}")
    if _is_smoke_text(config.name):
        errors.append(f"public main config name must not be smoke-scoped: {config.name}")
    if _is_smoke_text(str(config.output_dir)):
        errors.append(f"public main output_dir must not be smoke-scoped: {config.output_dir}")
    if len(config.seeds) < 5:
        errors.append("public main config must use at least 5 seeds by default")


def _validate_rows_not_smoke(label: str, rows: list[dict[str, Any]], errors: list[str]) -> None:
    for index, row in enumerate(rows):
        row_name = _row_name(label, index, row)
        if row.get("not_topconf_main_table") is True:
            errors.append(f"{row_name}: not_topconf_main_table rows cannot enter public main artifacts")
        for key in ("artifact_type", "evidence_scope", "public_metrics_scope"):
            value = row.get(key)
            if _is_smoke_text(value):
                errors.append(f"{row_name}: {key} is smoke/not-topconf scoped: {value}")
        raw_metric_path = row.get("raw_metric_path") or row.get("source_raw_metric_path")
        if _is_smoke_text(_basename_text(raw_metric_path)):
            errors.append(f"{row_name}: raw metric path is smoke/not-topconf scoped: {raw_metric_path}")


def _validate_raw_metric_coverage(
    config: MultimodalExperimentConfig,
    rows: list[dict[str, Any]],
    *,
    split: str,
    errors: list[str],
) -> dict[str, Any]:
    expected_models = ("ovha_full", *config.baseline_names)
    expected_model_set = set(expected_models)
    expected_seed_set = set(config.seeds)
    observed: dict[str, set[int]] = {}
    target_rows = []
    for index, row in enumerate(rows):
        row_name = _row_name("raw_metrics", index, row)
        _validate_metric_row_fields(row_name, config, row, errors=errors)
        row_split = str(row.get("split", ""))
        if row_split == split:
            target_rows.append(row)
            model = str(row.get("model", ""))
            seed = _safe_int(row.get("seed"))
            if model:
                observed.setdefault(model, set())
            if seed is not None and model:
                observed[model].add(seed)

    if not target_rows:
        errors.append(f"raw metrics missing target split rows: {split}")

    for model in sorted(set(observed) - expected_model_set):
        errors.append(f"raw metrics contain model not declared by config: {model}")
    for model in expected_models:
        seeds = observed.get(model, set())
        missing = sorted(expected_seed_set - seeds)
        extra = sorted(seeds - expected_seed_set)
        if missing:
            errors.append(f"missing configured seed coverage for model {model}: {', '.join(map(str, missing))}")
        if extra:
            errors.append(f"raw metrics contain unconfigured seeds for model {model}: {', '.join(map(str, extra))}")

    return {
        "models": list(expected_models),
        "seed_count": len(config.seeds),
        "seeds": list(config.seeds),
        "target_split_row_count": len(target_rows),
        "raw_metric_row_count": len(rows),
    }


def _validate_metric_row_fields(
    row_name: str,
    config: MultimodalExperimentConfig,
    row: dict[str, Any],
    *,
    errors: list[str],
) -> None:
    if row.get("dataset") != config.dataset_name:
        errors.append(f"{row_name}: dataset must be {config.dataset_name}")
    if row.get("task") != config.task_type:
        errors.append(f"{row_name}: task must be {config.task_type}")
    row_split = str(row.get("split", ""))
    if row_split not in config.eval_splits:
        errors.append(f"{row_name}: split must be one of {config.eval_splits}")


def _validate_diagnostic_seed_coverage(
    config: MultimodalExperimentConfig,
    diagnostics_rows: list[dict[str, Any]],
    *,
    split: str,
    errors: list[str],
) -> None:
    observed = {
        seed
        for row in diagnostics_rows
        if str(row.get("split", "")) == split
        for seed in [_safe_int(row.get("seed"))]
        if seed is not None
    }
    missing = sorted(set(config.seeds) - observed)
    if missing:
        errors.append(f"diagnostics missing configured seed coverage: {', '.join(map(str, missing))}")


def _validate_robustness_rows(
    config: MultimodalExperimentConfig,
    robustness_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if not robustness_rows:
        errors.append("robustness_rows must contain at least one public main robustness row")
        return
    expected_models = {"ovha_full", *config.baseline_names}
    observed_models = {str(row.get("model", "")) for row in robustness_rows if str(row.get("model", "")).strip()}
    if "ovha_full" not in observed_models:
        errors.append("robustness_rows missing ovha_full")
    if observed_models and not observed_models & (expected_models - {"ovha_full"}):
        errors.append("robustness_rows missing configured same-feature baseline rows")


def _read_jsonl_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} must contain JSONL object rows")
                rows.append(payload)
    if not rows:
        raise ValueError(f"{path} must contain at least one JSONL row")
    return rows


def _artifact_descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _row_name(label: str, index: int, row: dict[str, Any]) -> str:
    return (
        f"{label}[{index}]"
        f"/{row.get('task', '?')}"
        f"/{row.get('split', '?')}"
        f"/{row.get('model', '?')}"
        f"/seed={row.get('seed', '?')}"
    )


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _basename_text(value: Any) -> str:
    return str(value).rsplit("/", 1)[-1] if value is not None else ""


def _is_smoke_text(value: Any) -> bool:
    if value is None:
        return False
    return _SMOKE_RE.search(str(value)) is not None


if __name__ == "__main__":
    raise SystemExit(main())
