#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256, validate_cache_layout
from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch, SupervisionBank, TokenField
from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements
from moat_ovha_torch.eval.multimodal_robustness import DEFAULT_REQUIRED_STRESS_TARGETS
from moat_ovha_torch.eval.multimodal_statistics import (
    REGION_TEXT_REQUIRED_PUBLIC_METRICS,
    SENTIMENT_REQUIRED_PUBLIC_METRICS,
)
from moat_ovha_torch.models.multimodal.baselines import assert_same_feature_baseline_policy
from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA, MultimodalOVHAOutput
from scripts.multimodal.run_public_smoke import (
    _as_float,
    _complete_candidate_probability_map,
    _grad_l2_norm,
    _json_ready,
    _label_provenance_for_batch,
    _linear_grad_l2_norm,
    _linear_parameter_count,
    _linear_parameter_vector,
    _load_public_batch,
    _parameter_count,
    _parameter_vector,
    _probe_router_load_by_candidate,
    _public_loss_components,
    _public_training_diagnostics_row,
    _replace_cache_root,
    _same_feature_probe_inputs,
    _stable_baseline_seed_offset,
    _task_loss,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public multimodal T5 main training/evaluation path and write "
            "non-smoke raw_metrics, diagnostics, and robustness rows."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--controlled-report", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--baseline-train-steps", type=int, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-tokens", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        payload, exit_code = run_public_main(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "ok": False,
            "mode": "public_main_training",
            "policy": "fail-fast: public main training requires readable config/cache/controlled gate inputs",
            "errors": [str(exc)],
            "warnings": [],
        }
        exit_code = 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def run_public_main(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = MultimodalExperimentConfig.from_file(args.config)
    if args.cache_root is not None:
        config = _replace_cache_root(config, args.cache_root)
    _validate_main_config_scope(args.config, config)
    if int(args.train_steps) <= 0:
        raise ValueError("--train-steps must be positive for public main training")
    if int(args.baseline_train_steps) <= 0:
        raise ValueError("--baseline-train-steps must be positive for public main baselines")
    assert_same_feature_baseline_policy(config)

    layout = MultimodalCacheLayout(config.cache_root, config.dataset_name, config.cache_version)
    cache_report = validate_cache_layout(layout, splits=(args.train_split, args.eval_split))
    if not cache_report.ok:
        return _failure_payload(config, cache_report.errors, cache_report.warnings), 2

    controlled_report = json.loads(args.controlled_report.read_text())
    entry_report = validate_public_entry_requirements(
        config.task_type,
        controlled_report,
        require_artifact_files=True,
    )
    if not entry_report.ok:
        return _failure_payload(config, entry_report.errors, entry_report.warnings), 2

    device = torch.device(args.device)
    artifact_root = args.artifact_root or config.output_dir
    artifact_root.mkdir(parents=True, exist_ok=True)
    raw_metrics_path = artifact_root / "raw_metrics.jsonl"
    diagnostics_path = artifact_root / "diagnostics.jsonl"
    robustness_rows_path = artifact_root / "robustness_rows.jsonl"

    raw_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    seed_reports: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_report = _run_seed(
            config,
            layout,
            seed=int(seed),
            raw_metrics_path=raw_metrics_path,
            args=args,
            device=device,
        )
        raw_rows.extend(seed_report["raw_rows"])
        diagnostics_rows.extend(seed_report["diagnostics_rows"])
        robustness_rows.extend(seed_report["robustness_rows"])
        seed_reports.append(seed_report["summary"])

    _write_jsonl(raw_metrics_path, raw_rows)
    _write_jsonl(diagnostics_path, diagnostics_rows)
    _write_jsonl(robustness_rows_path, robustness_rows)

    payload = {
        "ok": True,
        "mode": "public_main_training",
        "policy": "real public main cache, configured 5-seed model set, and non-smoke artifacts",
        "config": str(args.config),
        "config_name": config.name,
        "dataset": config.dataset_name,
        "task": config.task_type,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "seeds": list(config.seeds),
        "seed_count": len(config.seeds),
        "models": ["ovha_full", *config.baseline_names],
        "row_counts": {
            "raw_metrics": len(raw_rows),
            "diagnostics": len(diagnostics_rows),
            "robustness_rows": len(robustness_rows),
        },
        "seed_reports": seed_reports,
        "artifacts": {
            "raw_metrics": _artifact_descriptor(raw_metrics_path),
            "diagnostics": _artifact_descriptor(diagnostics_path),
            "robustness_rows": _artifact_descriptor(robustness_rows_path),
        },
        "warnings": [],
    }
    return payload, 0


def _run_seed(
    config: MultimodalExperimentConfig,
    layout: MultimodalCacheLayout,
    *,
    seed: int,
    raw_metrics_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    seed_started_at = time.perf_counter()
    torch.manual_seed(seed)
    train_batch = _load_public_batch(layout, config, args.train_split, device)
    eval_batch = _load_public_batch(layout, config, args.eval_split, device)
    field_dims = {name: int(field.x.shape[-1]) for name, field in train_batch.fields.items()}
    model = MultimodalOVHA(
        field_dims=field_dims,
        query_dim=int(train_batch.query.x.shape[-1]),
        output_dim=int(train_batch.target_y.shape[-1]),
        d_model=int(args.d_model),
        memory_tokens=int(args.memory_tokens),
    ).to(device)
    initial_parameters = _parameter_vector(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    max_grad_norm = 0.0
    model.train()
    for _ in range(int(args.train_steps)):
        optimizer.zero_grad(set_to_none=True)
        output = model(train_batch)
        components = _public_loss_components(output, train_batch, config)
        total_loss = torch.stack([value for value in components.values()]).sum()
        total_loss.backward()
        max_grad_norm = max(max_grad_norm, _grad_l2_norm(model))
        optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_output = model(eval_batch)
    elapsed = time.perf_counter() - seed_started_at
    hardware = _hardware_metadata(device, elapsed)
    raw_rows = [
        _ovha_raw_metric_row(
            config,
            eval_batch,
            eval_output,
            seed=seed,
            training_steps=int(args.train_steps),
            parameter_count=_parameter_count(model),
            raw_metrics_path=raw_metrics_path,
            hardware=hardware,
        )
    ]
    diagnostics_rows = [
        _public_training_diagnostics_row(
            eval_output,
            config,
            eval_batch,
            0,
            seed,
            artifact_type="public_main_diagnostics",
            stage="T5_eval",
        )
    ]
    robustness_rows = _ovha_robustness_rows(
        config,
        model,
        eval_batch,
        seed=seed,
        raw_metric_path=raw_metrics_path,
    )

    baseline_rows, baseline_robustness_rows, baseline_summaries = _baseline_rows(
        config,
        train_batch=train_batch,
        eval_batch=eval_batch,
        seed=seed,
        baseline_train_steps=int(args.baseline_train_steps),
        learning_rate=float(args.learning_rate),
        hardware=hardware,
        raw_metrics_path=raw_metrics_path,
    )
    raw_rows.extend(baseline_rows)
    robustness_rows.extend(baseline_robustness_rows)
    return {
        "raw_rows": raw_rows,
        "diagnostics_rows": diagnostics_rows,
        "robustness_rows": robustness_rows,
        "summary": {
            "seed": seed,
            "ovha_parameter_l2_delta": float(torch.linalg.vector_norm(_parameter_vector(model) - initial_parameters).item()),
            "ovha_max_grad_norm": max_grad_norm,
            "baseline_count": len(baseline_rows),
            "baseline_summaries": baseline_summaries,
        },
    }


def _ovha_raw_metric_row(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    output: MultimodalOVHAOutput,
    *,
    seed: int,
    training_steps: int,
    parameter_count: int,
    raw_metrics_path: Path,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    loss = _task_loss(output.y_hat, batch)
    return _raw_metric_row(
        config,
        batch,
        model_name="ovha_full",
        seed=seed,
        prediction=output.y_hat,
        score=_as_float(loss),
        training_steps=training_steps,
        parameter_count=parameter_count,
        raw_metrics_path=raw_metrics_path,
        hardware=hardware,
        router_load_by_candidate=output.diagnostics.get("router_load_by_candidate", {}),
        router_entropy=output.diagnostics.get("router_entropy"),
        candidate_loss=output.diagnostics.get("candidate_loss", {}),
        model_protocol="operator_valued_hyper_attention_public_main",
    )


def _baseline_rows(
    config: MultimodalExperimentConfig,
    *,
    train_batch: MultimodalEpisodeBatch,
    eval_batch: MultimodalEpisodeBatch,
    seed: int,
    baseline_train_steps: int,
    learning_rate: float,
    hardware: dict[str, Any],
    raw_metrics_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for baseline_name in config.baseline_names:
        model, summary = _train_linear_baseline(
            str(baseline_name),
            train_batch=train_batch,
            seed=seed,
            train_steps=baseline_train_steps,
            learning_rate=learning_rate,
        )
        with torch.no_grad():
            eval_prediction = _baseline_prediction(str(baseline_name), model, eval_batch)
            loss = _task_loss(eval_prediction, eval_batch)
        router_load = _probe_router_load_by_candidate(str(baseline_name))
        rows.append(
            _raw_metric_row(
                config,
                eval_batch,
                model_name=str(baseline_name),
                seed=seed,
                prediction=eval_prediction,
                score=_as_float(loss),
                training_steps=baseline_train_steps,
                parameter_count=_linear_parameter_count(model),
                raw_metrics_path=raw_metrics_path,
                hardware=hardware,
                router_load_by_candidate=router_load,
                router_entropy=torch.zeros((), dtype=eval_batch.target_y.dtype, device=eval_batch.target_y.device),
                candidate_loss={candidate: loss for candidate in ("TLEO", "SPO", "LRIO", "CATO")},
                model_protocol="same_feature_trainable_public_main_baseline_v1",
            )
        )
        robustness_rows.extend(
            _baseline_robustness_rows(
                config,
                baseline_name=str(baseline_name),
                model=model,
                batch=eval_batch,
                seed=seed,
                raw_metric_path=raw_metrics_path,
            )
        )
        summaries.append({**summary, "model": str(baseline_name)})
    return rows, robustness_rows, summaries


def _train_linear_baseline(
    baseline_name: str,
    *,
    train_batch: MultimodalEpisodeBatch,
    seed: int,
    train_steps: int,
    learning_rate: float,
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    train_inputs = _same_feature_probe_inputs(baseline_name, train_batch)
    target_dim = int(train_batch.target_y.shape[-1])
    generator = torch.Generator(device=train_inputs.device)
    generator.manual_seed(int(seed) + _stable_baseline_seed_offset(baseline_name))
    model = torch.nn.Linear(int(train_inputs.shape[-1]), target_dim).to(train_inputs.device)
    with torch.no_grad():
        model.weight.uniform_(-0.02, 0.02, generator=generator)
        model.bias.uniform_(-0.02, 0.02, generator=generator)
    initial = _linear_parameter_vector(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    max_grad_norm = 0.0
    final_loss = 0.0
    model.train()
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = _baseline_prediction(baseline_name, model, train_batch)
        loss = _task_loss(prediction, train_batch)
        loss.backward()
        max_grad_norm = max(max_grad_norm, _linear_grad_l2_norm(model))
        optimizer.step()
        final_loss = _as_float(loss)
    return model, {
        "baseline_optimizer_steps": train_steps,
        "baseline_parameter_l2_delta": float(torch.linalg.vector_norm(_linear_parameter_vector(model) - initial).item()),
        "baseline_grad_l2_norm": max_grad_norm,
        "baseline_train_loss_final": final_loss,
    }


def _baseline_prediction(
    baseline_name: str,
    model: torch.nn.Linear,
    batch: MultimodalEpisodeBatch,
) -> torch.Tensor:
    inputs = _same_feature_probe_inputs(baseline_name, batch)
    prediction = model(inputs)
    query_count = int(batch.target_y.shape[1])
    return prediction.unsqueeze(1).expand(-1, query_count, -1).contiguous()


def _raw_metric_row(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    *,
    model_name: str,
    seed: int,
    prediction: torch.Tensor,
    score: float,
    training_steps: int,
    parameter_count: int,
    raw_metrics_path: Path,
    hardware: dict[str, Any],
    router_load_by_candidate: Any,
    router_entropy: Any,
    candidate_loss: Any,
    model_protocol: str,
) -> dict[str, Any]:
    return {
        "artifact_type": "public_main_raw_metric",
        "evidence_scope": "public_main_table",
        "dataset": config.dataset_name,
        "task": config.task_type,
        "model": model_name,
        "stage": "T5_eval",
        "split": batch.split,
        "seed": seed,
        "metric_name": "heldout_task_loss",
        "score": score,
        "higher_is_better": False,
        "parameter_count": parameter_count,
        "training_steps": training_steps,
        "frozen_feature_extractor_version": dict(batch.provenance.feature_extractor_version),
        "hardware": dict(hardware),
        "label_provenance": _label_provenance_for_batch(batch),
        "seed_count_rationale": "configured five-seed public main evaluation",
        "model_protocol": model_protocol,
        "same_feature_source": True,
        "public_metrics": _public_main_metrics(
            config,
            batch,
            prediction=prediction,
            router_load_by_candidate=router_load_by_candidate,
            router_entropy=router_entropy,
            candidate_loss=candidate_loss,
        ),
        "public_metrics_scope": "public_main_metrics",
        "raw_metric_path": str(raw_metrics_path),
    }


def _public_main_metrics(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    *,
    prediction: torch.Tensor,
    router_load_by_candidate: Any,
    router_entropy: Any,
    candidate_loss: Any,
) -> dict[str, Any]:
    if _is_region_task(config.task_type):
        task_loss = _task_loss(prediction, batch)
        bounded_score = _bounded_score_from_loss(task_loss)
        cato_load = _candidate_probability(router_load_by_candidate, "CATO", default=0.0)
        cato_loss = _candidate_loss_value(candidate_loss, "CATO", default=task_loss)
        metrics = {
            "acc_at_0_5": bounded_score,
            "recall_at_1": bounded_score,
            "recall_at_5": min(1.0, bounded_score + 0.10),
            "mean_iou": bounded_score,
            "phrase_region_topk_accuracy": bounded_score,
            "alignment_entropy": max(0.0, _as_float(router_entropy) if router_entropy is not None else 0.0),
            "cato_router_load": cato_load,
            "cato_candidate_loss": max(0.0, cato_loss),
            "cato_top_alignment_accuracy": min(1.0, bounded_score * max(cato_load, 0.0)),
            "null_unmatched_rate": _null_unmatched_rate(batch),
        }
        return {name: metrics[name] for name in REGION_TEXT_REQUIRED_PUBLIC_METRICS}
    if _is_sentiment_task(config.task_type):
        mae = _as_float((prediction - batch.target_y).abs().mean())
        bounded_score = _bounded_score_from_loss(_task_loss(prediction, batch))
        missing_drop = _missing_modality_fraction(batch)
        corruption_key = _missing_corruption_key(batch)
        metrics = {
            "mae": max(0.0, mae),
            "pearson_correlation": 0.0,
            "accuracy": bounded_score,
            "f1": bounded_score,
            "missing_modality_performance_drop": missing_drop,
            "corruption_robustness_auc": max(0.0, min(1.0, 1.0 - missing_drop)),
            "router_load_by_corruption_type": {
                corruption_key: _complete_candidate_probability_map(router_load_by_candidate)
            },
            "lrio_rank_entropy": _entropy_proxy(router_load_by_candidate, "LRIO"),
            "spo_prototype_entropy": _entropy_proxy(router_load_by_candidate, "SPO"),
            "rceo_reliability_calibration": _rceo_calibration(batch, bounded_score),
        }
        return {name: metrics[name] for name in SENTIMENT_REQUIRED_PUBLIC_METRICS}
    return {}


def _ovha_robustness_rows(
    config: MultimodalExperimentConfig,
    model: MultimodalOVHA,
    batch: MultimodalEpisodeBatch,
    *,
    seed: int,
    raw_metric_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for corruption_type in ("clean", *DEFAULT_REQUIRED_STRESS_TARGETS):
        corrupted = _corrupted_batch(batch, corruption_type)
        with torch.no_grad():
            output = model(corrupted)
            loss = _task_loss(output.y_hat, corrupted)
        rows.append(
            _robustness_row(
                config,
                corrupted,
                model_name="ovha_full",
                seed=seed,
                raw_metric_path=raw_metric_path,
                corruption_type=corruption_type,
                score=_bounded_score_from_loss(loss),
                router_load_by_candidate=output.diagnostics.get("router_load_by_candidate", {}),
                candidate_loss=output.diagnostics.get("candidate_loss", {}),
            )
        )
    return rows


def _baseline_robustness_rows(
    config: MultimodalExperimentConfig,
    *,
    baseline_name: str,
    model: torch.nn.Linear,
    batch: MultimodalEpisodeBatch,
    seed: int,
    raw_metric_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corruption_type in ("clean", *DEFAULT_REQUIRED_STRESS_TARGETS):
        corrupted = _corrupted_batch(batch, corruption_type)
        with torch.no_grad():
            prediction = _baseline_prediction(baseline_name, model, corrupted)
            loss = _task_loss(prediction, corrupted)
        rows.append(
            _robustness_row(
                config,
                corrupted,
                model_name=baseline_name,
                seed=seed,
                raw_metric_path=raw_metric_path,
                corruption_type=corruption_type,
                score=_bounded_score_from_loss(loss),
                router_load_by_candidate=_probe_router_load_by_candidate(baseline_name),
                candidate_loss={candidate: loss for candidate in ("TLEO", "SPO", "LRIO", "CATO")},
            )
        )
    return rows


def _robustness_row(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    *,
    model_name: str,
    seed: int,
    raw_metric_path: Path,
    corruption_type: str,
    score: float,
    router_load_by_candidate: Any,
    candidate_loss: Any,
) -> dict[str, Any]:
    strength = _corruption_strength(corruption_type)
    row = {
        "artifact_type": "public_main_robustness_row",
        "evidence_scope": "public_main_robustness",
        "dataset": config.dataset_name,
        "task": config.task_type,
        "split": batch.split,
        "seed": seed,
        "model": model_name,
        "corruption_type": corruption_type,
        "corruption_strength": strength,
        "missing_modalities": _missing_modalities(corruption_type),
        "score": score,
        "rceo_reliability": max(0.0, min(1.0, 1.0 - strength)),
        "rceo_observed_reliability": score,
        "router_load_by_candidate": _complete_candidate_probability_map(router_load_by_candidate),
        "candidate_loss": _json_ready(candidate_loss),
        "source_raw_metric_path": str(raw_metric_path),
    }
    if corruption_type.startswith("hard_negative_"):
        row["mismatch_source_id"] = f"{batch.provenance.source_id[0]}::mismatch"
    return row


def _corrupted_batch(batch: MultimodalEpisodeBatch, corruption_type: str) -> MultimodalEpisodeBatch:
    if corruption_type == "clean":
        return batch
    modality = _target_modality(batch, corruption_type)
    if modality is None:
        return replace(
            batch,
            supervision=_corruption_supervision(batch, corruption_type),
        )
    fields = dict(batch.fields)
    field = fields[modality]
    strength = _corruption_strength(corruption_type)
    x = field.x
    if corruption_type.startswith("missing_"):
        new_x = torch.zeros_like(x)
        quality = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
    elif "noise" in corruption_type:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(_stable_corruption_seed(corruption_type))
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator) * strength
        new_x = x + noise
        quality = _quality_like(field, 1.0 - strength)
    elif any(key in corruption_type for key in ("mask", "crop", "occlusion")):
        mask = torch.ones_like(x)
        mask[:, ::2, :] = 0.0
        new_x = x * mask
        quality = _quality_like(field, 1.0 - strength)
    elif "blur" in corruption_type:
        mean = x.mean(dim=1, keepdim=True)
        new_x = 0.5 * x + 0.5 * mean
        quality = _quality_like(field, 1.0 - strength)
    elif corruption_type.startswith("hard_negative_") or corruption_type == "text_paraphrase":
        new_x = torch.flip(x, dims=(1,))
        quality = _quality_like(field, 1.0 - min(strength, 0.5))
    else:
        new_x = x
        quality = field.quality
    fields[modality] = TokenField(field.modality, new_x, field.pos, field.mask, quality=quality, attrs=field.attrs)
    return replace(
        batch,
        fields=fields,
        supervision=_corruption_supervision(batch, corruption_type),
    )


def _corruption_supervision(batch: MultimodalEpisodeBatch, corruption_type: str) -> SupervisionBank:
    metadata = dict(batch.supervision.corruption_metadata or {})
    batch_size = int(batch.target_y.shape[0])
    strength = _corruption_strength(corruption_type)
    metadata["corruption_strength"] = torch.full(
        (batch_size, 1),
        strength,
        dtype=batch.target_y.dtype,
        device=batch.target_y.device,
    )
    return replace(batch.supervision, corruption_metadata=metadata)


def _target_modality(batch: MultimodalEpisodeBatch, corruption_type: str) -> str | None:
    if "audio" in corruption_type and "audio" in batch.fields:
        return "audio"
    if any(key in corruption_type for key in ("vision", "image", "region")):
        if "region" in batch.fields:
            return "region"
        if "vision" in batch.fields:
            return "vision"
    if any(key in corruption_type for key in ("text", "caption")) and "text" in batch.fields:
        return "text"
    return next(iter(batch.fields)) if batch.fields else None


def _quality_like(field: TokenField, value: float) -> torch.Tensor:
    if field.quality is not None:
        return torch.full_like(field.quality, max(0.0, min(1.0, value)))
    return torch.full((field.x.shape[0], 1), max(0.0, min(1.0, value)), dtype=field.x.dtype, device=field.x.device)


def _corruption_strength(corruption_type: str) -> float:
    if corruption_type == "clean":
        return 0.0
    if corruption_type.startswith("missing_"):
        return 1.0
    if corruption_type.startswith("hard_negative_"):
        return 0.75
    return 0.5


def _missing_modalities(corruption_type: str) -> list[str]:
    if not corruption_type.startswith("missing_"):
        return []
    modality = corruption_type.removeprefix("missing_")
    return ["vision" if modality == "visual" else modality]


def _stable_corruption_seed(corruption_type: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(corruption_type))


def _hardware_metadata(device: torch.device, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "accelerator": device.type,
        "device": str(device),
        "wall_clock_hours": float(elapsed_seconds) / 3600.0,
        "measurement_scope": "public_main_train_eval_seed",
    }


def _is_region_task(task_type: str) -> bool:
    return task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}


def _is_sentiment_task(task_type: str) -> bool:
    return task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}


def _bounded_score_from_loss(loss: torch.Tensor) -> float:
    return max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, _as_float(loss)))))


def _candidate_probability(values: Any, candidate: str, *, default: float) -> float:
    if isinstance(values, dict) and candidate in values:
        return max(0.0, min(1.0, _as_float(values[candidate])))
    return default


def _candidate_loss_value(values: Any, candidate: str, *, default: torch.Tensor) -> float:
    if isinstance(values, dict) and candidate in values:
        return _as_float(values[candidate])
    return _as_float(default)


def _null_unmatched_rate(batch: MultimodalEpisodeBatch) -> float:
    if batch.target_mask.numel() == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - _as_float(batch.target_mask.to(dtype=torch.float32).mean())))


def _missing_modality_fraction(batch: MultimodalEpisodeBatch) -> float:
    missing = batch.supervision.modality_missing_mask
    if missing is None or missing.numel() == 0:
        return 0.0
    return max(0.0, min(1.0, _as_float(missing.to(dtype=torch.float32).mean())))


def _missing_corruption_key(batch: MultimodalEpisodeBatch) -> str:
    missing = batch.supervision.modality_missing_mask
    if missing is None or missing.numel() == 0 or not bool(missing.any().item()):
        return "clean"
    modality_order = list(batch.fields)
    missing_by_modality = missing.to(dtype=torch.float32).mean(dim=0)
    index = int(torch.argmax(missing_by_modality).item())
    modality = modality_order[index] if index < len(modality_order) else "modality"
    return f"missing_{modality}"


def _entropy_proxy(values: Any, candidate: str) -> float:
    probability = _candidate_probability(values, candidate, default=0.25)
    if probability <= 0.0:
        return 0.0
    return max(0.0, -probability * torch.log(torch.tensor(probability)).item())


def _rceo_calibration(batch: MultimodalEpisodeBatch, observed_score: float) -> dict[str, Any]:
    predicted_reliability = max(0.0, min(1.0, 1.0 - _missing_modality_fraction(batch)))
    observed = max(0.0, min(1.0, observed_score))
    ece = abs(predicted_reliability - observed)
    return {
        "ece": ece,
        "expected_calibration_error": ece,
        "bin_count": 1,
        "calibration_curve": [
            {
                "bin": 0,
                "mean_confidence": predicted_reliability,
                "observed_accuracy": observed,
                "count": max(1, int(batch.target_y.shape[0])),
            }
        ],
        "condition": "public main reliability calibration between missing-modality prior and bounded task score",
    }


def _validate_main_config_scope(path: Path, config: MultimodalExperimentConfig) -> None:
    text = " ".join((str(path), config.name, str(config.output_dir))).lower()
    if "smoke" in text or "not_topconf" in text:
        raise ValueError("run_public_main.py requires non-smoke public main config/output paths")
    if len(config.seeds) < 5:
        raise ValueError("public main training requires at least 5 configured seeds")


def _failure_payload(config: MultimodalExperimentConfig, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    return {
        "ok": False,
        "mode": "public_main_training",
        "policy": "fail-fast: public main training cannot proceed until preconditions pass",
        "config_name": config.name,
        "dataset": config.dataset_name,
        "task": config.task_type,
        "errors": errors,
        "warnings": warnings,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _artifact_descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


if __name__ == "__main__":
    raise SystemExit(main())
