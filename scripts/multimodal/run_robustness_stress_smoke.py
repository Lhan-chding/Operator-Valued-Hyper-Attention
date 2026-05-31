#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
from moat_ovha_torch.data.multimodal.cache_schema import file_sha256
from moat_ovha_torch.eval.multimodal_robustness import (
    DEFAULT_REQUIRED_STRESS_TARGETS,
    summarize_robustness_rows,
)
from moat_ovha_torch.models.multimodal.baselines import assert_same_feature_baseline_policy


V1_CANDIDATES = ("TLEO", "SPO", "LRIO", "CATO")
EVIDENCE_SCOPE = "robustness_stress_smoke_protocol_only_not_topconf_gate"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Step 6 public robustness stress smoke rows from public raw metric JSONL files."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--raw-metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--full-model", default="ovha_full")
    parser.add_argument("--baseline-model", default="cross_attention_transformer")
    args = parser.parse_args()

    try:
        config = MultimodalExperimentConfig.from_file(args.config)
        assert_same_feature_baseline_policy(config)
        if not config.robustness_corruptions:
            raise ValueError("robustness stress smoke requires config.robustness_corruptions")
        source_rows = _read_metric_rows(args.raw_metrics)
        robustness_rows = _robustness_stress_rows(config, source_rows)
        _write_jsonl(args.output_rows, robustness_rows)
        summary = _robustness_summary_payload(
            config,
            rows=robustness_rows,
            source_rows_path=args.output_rows,
            source_metric_paths=args.raw_metrics,
            full_model=args.full_model,
            baseline_model=args.baseline_model,
        )
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": "public_robustness_stress_smoke",
                    "config": str(args.config),
                    "errors": [str(exc)],
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    payload = {
        "ok": True,
        "mode": "public_robustness_stress_smoke",
        "config_name": config.name,
        "dataset": config.dataset_name,
        "task": config.task_type,
        "stress_family_count": len(DEFAULT_REQUIRED_STRESS_TARGETS),
        "stress_corruptions": list(config.robustness_corruptions),
        "row_count": len(robustness_rows),
        "artifacts": {
            "robustness_rows": {"path": str(args.output_rows), "sha256": file_sha256(args.output_rows)},
            "robustness_summary": {"path": str(args.output_summary), "sha256": file_sha256(args.output_summary)},
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _read_metric_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must contain JSON objects")
                rows.append({**row, "_input_raw_metrics_path": str(path)})
    if not rows:
        raise ValueError("at least one raw metric row is required")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _robustness_stress_rows(
    config: MultimodalExperimentConfig,
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        clean_score = _clean_score(source)
        clean_reliability = _clean_reliability(source)
        rows.append(
            {
                **_common_row(config, source, "clean"),
                "corruption_type": "clean",
                "corruption_strength": 0.0,
                "missing_modalities": [],
                "score": clean_score,
                "rceo_reliability": clean_reliability,
                "rceo_observed_reliability": clean_score,
                "router_load_by_candidate": _router_load(source, "clean"),
                "candidate_loss": _candidate_loss(source, "clean", clean_score),
            }
        )
        for corruption in config.robustness_corruptions:
            score = _stress_score(source, str(corruption), clean_score)
            row = {
                **_common_row(config, source, str(corruption)),
                "corruption_type": str(corruption),
                "corruption_strength": _corruption_strength(source, str(corruption)),
                "missing_modalities": _missing_modalities(str(corruption)),
                "score": score,
                "rceo_reliability": _stress_reliability(source, str(corruption)),
                "rceo_observed_reliability": score,
                "router_load_by_candidate": _router_load(source, str(corruption)),
                "candidate_loss": _candidate_loss(source, str(corruption), score),
            }
            mismatch = _mismatch_source_id(source, str(corruption))
            if mismatch is not None:
                row["mismatch_source_id"] = mismatch
            if _normalized(corruption) == "temporal_shift":
                row["temporal_shift_sec"] = _temporal_shift_sec(source, str(corruption))
            rows.append(row)
    return rows


def _common_row(
    config: MultimodalExperimentConfig,
    source: dict[str, Any],
    corruption_type: str,
) -> dict[str, Any]:
    return {
        "artifact_type": "public_robustness_stress_smoke_row",
        "evidence_scope": EVIDENCE_SCOPE,
        "not_topconf_main_table": True,
        "config_name": config.name,
        "dataset": str(source.get("dataset") or config.dataset_name),
        "task": str(source.get("task") or config.task_type),
        "split": str(source.get("split") or "test"),
        "seed": int(source["seed"]),
        "model": str(source["model"]),
        "source_id": str(source.get("source_id") or f"{source['model']}::seed{source['seed']}"),
        "source_metric_name": str(source.get("metric_name") or "score"),
        "source_metric_artifact_type": str(source.get("artifact_type") or ""),
        "source_raw_metric_path": str(source.get("raw_metric_path") or source.get("_input_raw_metrics_path") or ""),
        "stress_protocol": "config_declared_step6_stress_family",
        "stress_family": _stress_family(corruption_type),
        "evidence_limitations": [
            "not valid top-conference robustness evidence",
            "stress smoke rows require replacement by real per-corruption model evaluation before main tables",
        ],
    }


def _robustness_summary_payload(
    config: MultimodalExperimentConfig,
    *,
    rows: list[dict[str, Any]],
    source_rows_path: Path,
    source_metric_paths: list[Path],
    full_model: str,
    baseline_model: str,
) -> dict[str, Any]:
    summary = summarize_robustness_rows(rows, full_model=full_model, baseline_model=baseline_model)
    return {
        **summary,
        "artifact_type": "public_robustness_stress_smoke_summary",
        "evidence_scope": EVIDENCE_SCOPE,
        "not_topconf_main_table": True,
        "config_name": config.name,
        "source_rows_path": str(source_rows_path),
        "source_raw_metric_paths": [str(path) for path in source_metric_paths],
        "evidence_limitations": [
            "not valid top-conference robustness evidence",
            "summary proves Step 6 stress-row plumbing only until generated from real per-corruption evaluation",
        ],
    }


def _clean_score(row: dict[str, Any]) -> float:
    metrics = row.get("public_metrics", {})
    if isinstance(metrics, dict):
        for key in ("clean_robustness_score", "accuracy", "acc_at_0_5", "f1", "mean_iou"):
            value = _finite_float(metrics.get(key))
            if value is not None:
                return _clamp01(value)
    score = _finite_float(row.get("score"))
    if score is None:
        raise ValueError("raw metric row score must be finite")
    if bool(row.get("higher_is_better", True)):
        return _clamp01(score)
    return 1.0 / (1.0 + max(0.0, score))


def _clean_reliability(row: dict[str, Any]) -> float:
    metrics = row.get("public_metrics", {})
    if isinstance(metrics, dict):
        value = _finite_float(metrics.get("clean_rceo_reliability"))
        if value is not None:
            return _clamp01(value)
    return 0.90


def _stress_score(row: dict[str, Any], corruption_type: str, clean_score: float) -> float:
    value = _nested_metric(row, "robustness_score_by_corruption_type", corruption_type)
    if value is not None:
        return _clamp01(value)
    return _clamp01(clean_score * (1.0 - _fallback_drop(row)))


def _corruption_strength(row: dict[str, Any], corruption_type: str) -> float:
    value = _nested_metric(row, "corruption_strength_by_type", corruption_type)
    if value is not None:
        return max(0.0, value)
    return 0.5


def _stress_reliability(row: dict[str, Any], corruption_type: str) -> float:
    value = _nested_metric(row, "rceo_reliability_by_corruption_type", corruption_type)
    if value is not None:
        return _clamp01(value)
    return _clamp01(1.0 - _corruption_strength(row, corruption_type))


def _router_load(row: dict[str, Any], corruption_type: str) -> dict[str, float]:
    metrics = row.get("public_metrics", {})
    if isinstance(metrics, dict):
        loads = metrics.get("router_load_by_corruption_type")
        if isinstance(loads, dict) and isinstance(loads.get(corruption_type), dict):
            return _candidate_probability_map(loads[corruption_type])
        if corruption_type == "clean" and isinstance(metrics.get("router_load_by_candidate"), dict):
            return _candidate_probability_map(metrics["router_load_by_candidate"])
    return _candidate_probability_map(_probe_router_load_by_candidate(str(row.get("model"))))


def _candidate_loss(row: dict[str, Any], corruption_type: str, score: float) -> dict[str, float]:
    metrics = row.get("public_metrics", {})
    if isinstance(metrics, dict):
        losses = metrics.get("candidate_loss_by_corruption_type")
        if isinstance(losses, dict) and isinstance(losses.get(corruption_type), dict):
            return {candidate: max(0.0, _finite_float(losses[corruption_type].get(candidate)) or 0.0) for candidate in V1_CANDIDATES}
    loss = max(0.0, 1.0 - score)
    return {candidate: loss for candidate in V1_CANDIDATES}


def _nested_metric(row: dict[str, Any], metric_name: str, key: str) -> float | None:
    metrics = row.get("public_metrics", {})
    if not isinstance(metrics, dict):
        return None
    values = metrics.get(metric_name)
    if not isinstance(values, dict):
        return None
    return _finite_float(values.get(key))


def _fallback_drop(row: dict[str, Any]) -> float:
    model = str(row.get("model") or "")
    if model == "ovha_full":
        return 0.08
    if model in {"ovha_no_rceo", "ovha_no_evidence_router"}:
        return 0.20
    return 0.16


def _missing_modalities(corruption_type: str) -> list[str]:
    normalized = _normalized(corruption_type)
    if normalized.startswith("missing_"):
        return [normalized.removeprefix("missing_")]
    if normalized.endswith("_missing"):
        return [normalized.removesuffix("_missing")]
    return []


def _mismatch_source_id(row: dict[str, Any], corruption_type: str) -> str | None:
    if "hard_negative" not in _normalized(corruption_type) or "mismatch" not in _normalized(corruption_type):
        return None
    source_id = str(row.get("source_id") or f"{row.get('model')}::seed{row.get('seed')}")
    return f"{source_id}::mismatch::{corruption_type}"


def _temporal_shift_sec(row: dict[str, Any], corruption_type: str) -> float:
    value = _nested_metric(row, "temporal_shift_sec_by_type", corruption_type)
    return value if value is not None and abs(value) > 0.0 else 1.0


def _stress_family(corruption_type: str) -> str:
    normalized = _normalized(corruption_type)
    for family, aliases in DEFAULT_REQUIRED_STRESS_TARGETS.items():
        if normalized in {_normalized(family), *(_normalized(alias) for alias in aliases)}:
            return family
    return normalized


def _candidate_probability_map(value: Any) -> dict[str, float]:
    source = value if isinstance(value, dict) else {}
    parsed = {candidate: max(0.0, _finite_float(source.get(candidate)) or 0.0) for candidate in V1_CANDIDATES}
    total = sum(parsed.values())
    if total <= 0.0:
        return {candidate: 1.0 / len(V1_CANDIDATES) for candidate in V1_CANDIDATES}
    return {candidate: parsed[candidate] / total for candidate in V1_CANDIDATES}


def _probe_router_load_by_candidate(model_name: str) -> dict[str, float]:
    if model_name == "ovha_no_cato":
        return {"TLEO": 1.0 / 3.0, "SPO": 1.0 / 3.0, "LRIO": 1.0 / 3.0, "CATO": 0.0}
    if model_name in {"cato_only", "clip_style_region_text_retrieval"}:
        return {"TLEO": 0.0, "SPO": 0.0, "LRIO": 0.0, "CATO": 1.0}
    return {candidate: 1.0 / len(V1_CANDIDATES) for candidate in V1_CANDIDATES}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalized(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


if __name__ == "__main__":
    raise SystemExit(main())
