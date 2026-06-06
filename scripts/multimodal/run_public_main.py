#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import fields as dataclass_fields, is_dataclass, replace
import json
import math
import os
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
from moat_ovha_torch.eval.grounding_metrics import METRICS_SOURCE, grounding_candidate_metrics
from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements
from moat_ovha_torch.eval.mosei_standard_metrics import mosei_standard_metrics
from moat_ovha_torch.eval.multimodal_robustness import DEFAULT_REQUIRED_STRESS_TARGETS
from moat_ovha_torch.eval.multimodal_statistics import (
    REGION_TEXT_REQUIRED_PUBLIC_METRICS,
    SENTIMENT_REQUIRED_PUBLIC_METRICS,
)
from moat_ovha_torch.models.multimodal.baselines import (
    assert_same_feature_baseline_policy,
    baseline_protocol_for_name,
    ovha_ablation_names_for_task,
)
from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA, MultimodalOVHAOutput
from moat_ovha_torch.models.multimodal.primitives.base import CandidateOutput
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
    _ovha_composition_kwargs,
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
    parser.add_argument(
        "--skip-cache-validation",
        action="store_true",
        help=(
            "Trust an already verified public cache and skip the expensive startup cache "
            "layout preflight. Use only after the cache has been validated separately."
        ),
    )
    parser.add_argument("--controlled-report", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--baseline-train-steps", type=int, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-tokens", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--preload-batches-to-device",
        action="store_true",
        help=(
            "Load train/selection/eval public batches directly onto --device instead of CPU. "
            "This can reduce host-to-device transfer overhead on large-memory GPUs, but it "
            "requires enough GPU memory for all three splits plus model state."
        ),
    )
    parser.add_argument(
        "--pilot-seed",
        type=int,
        help=(
            "Run only one configured seed for a pilot fit. The config must still contain "
            "the formal five-seed public-main plan, and outputs are marked as a pilot subset."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=None,
        help=(
            "Print public-main training progress every N optimizer steps to stderr. "
            "Defaults to OVHA_PUBLIC_MAIN_PROGRESS_INTERVAL or 500; set 0 to disable periodic rows."
        ),
    )
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
    if not args.skip_cache_validation:
        cache_report = validate_cache_layout(
            layout,
            splits=tuple(dict.fromkeys((args.train_split, args.selection_split, args.eval_split))),
        )
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
    progress_interval = _progress_interval(args.progress_interval)
    artifact_root = args.artifact_root or config.output_dir
    artifact_root.mkdir(parents=True, exist_ok=True)
    raw_metrics_path = artifact_root / "raw_metrics.jsonl"
    diagnostics_path = artifact_root / "diagnostics.jsonl"
    robustness_rows_path = artifact_root / "robustness_rows.jsonl"
    per_sample_predictions_path = artifact_root / "per_sample_predictions.jsonl"

    raw_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    per_sample_prediction_rows: list[dict[str, Any]] = []
    seed_reports: list[dict[str, Any]] = []
    selected_seeds = _selected_seeds(config, args.pilot_seed)
    pilot_seed_subset = len(selected_seeds) != len(config.seeds)
    _print_progress(
        "start",
        dataset=config.dataset_name,
        task=config.task_type,
        seeds=",".join(str(seed) for seed in selected_seeds),
        configured_seeds=",".join(str(seed) for seed in config.seeds),
        pilot_seed_subset=pilot_seed_subset,
        model_count=1 + len(config.baseline_names),
        train_steps=int(args.train_steps),
        baseline_train_steps=int(args.baseline_train_steps),
        device=str(device),
        progress_interval=progress_interval,
        d_model=int(args.d_model),
        memory_tokens=int(args.memory_tokens),
    )
    for seed in selected_seeds:
        seed_report = _run_seed(
            config,
            layout,
            seed=int(seed),
            raw_metrics_path=raw_metrics_path,
            args=args,
            device=device,
            progress_interval=progress_interval,
        )
        raw_rows.extend(seed_report["raw_rows"])
        diagnostics_rows.extend(seed_report["diagnostics_rows"])
        robustness_rows.extend(seed_report["robustness_rows"])
        per_sample_prediction_rows.extend(seed_report["per_sample_prediction_rows"])
        seed_reports.append(seed_report["summary"])

    _write_jsonl(raw_metrics_path, raw_rows)
    _write_jsonl(diagnostics_path, diagnostics_rows)
    _write_jsonl(robustness_rows_path, robustness_rows)
    _write_jsonl(per_sample_predictions_path, per_sample_prediction_rows)

    payload = {
        "ok": True,
        "mode": "public_main_training",
        "policy": "real public main cache, configured 5-seed model set, and non-smoke artifacts",
        "config": str(args.config),
        "config_name": config.name,
        "dataset": config.dataset_name,
        "task": config.task_type,
        "train_split": args.train_split,
        "selection_split": args.selection_split,
        "eval_split": args.eval_split,
        "seeds": list(selected_seeds),
        "seed_count": len(selected_seeds),
        "configured_seeds": list(config.seeds),
        "configured_seed_count": len(config.seeds),
        "pilot_seed_subset": pilot_seed_subset,
        "models": [config.main_model_name, *config.baseline_names],
        "row_counts": {
            "raw_metrics": len(raw_rows),
            "diagnostics": len(diagnostics_rows),
            "robustness_rows": len(robustness_rows),
            "per_sample_predictions": len(per_sample_prediction_rows),
        },
        "seed_reports": seed_reports,
        "artifacts": {
            "raw_metrics": _artifact_descriptor(raw_metrics_path),
            "diagnostics": _artifact_descriptor(diagnostics_path),
            "robustness_rows": _artifact_descriptor(robustness_rows_path),
            "per_sample_predictions": _artifact_descriptor(per_sample_predictions_path),
        },
        "warnings": [],
    }
    return payload, 0


def _selected_seeds(config: MultimodalExperimentConfig, pilot_seed: int | None) -> tuple[int, ...]:
    if pilot_seed is None:
        return config.seeds
    seed = int(pilot_seed)
    if seed not in set(config.seeds):
        raise ValueError(f"--pilot-seed must be one of the configured seeds: {', '.join(str(value) for value in config.seeds)}")
    return (seed,)


def _run_seed(
    config: MultimodalExperimentConfig,
    layout: MultimodalCacheLayout,
    *,
    seed: int,
    raw_metrics_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    progress_interval: int,
) -> dict[str, Any]:
    seed_started_at = time.perf_counter()
    torch.manual_seed(seed)
    _print_progress("seed:start", seed=seed)
    host_device = device if args.preload_batches_to_device else torch.device("cpu")
    train_batch = _load_public_batch_with_progress(layout, config, args.train_split, host_device, seed)
    selection_batch = _load_public_batch_with_progress(layout, config, args.selection_split, host_device, seed)
    eval_batch = _load_public_batch_with_progress(layout, config, args.eval_split, host_device, seed)
    target_mean, target_std = _target_standardizer(train_batch)
    fit_batch_std = _standardize_batch_targets(train_batch, target_mean, target_std)
    val_batch_std = _standardize_batch_targets(selection_batch, target_mean, target_std)
    eval_batch_std = _standardize_batch_targets(eval_batch, target_mean, target_std)
    field_dims = {name: int(field.x.shape[-1]) for name, field in train_batch.fields.items()}
    model = MultimodalOVHA(
        field_dims=field_dims,
        query_dim=int(train_batch.query.x.shape[-1]),
        output_dim=int(train_batch.target_y.shape[-1]),
        d_model=int(args.d_model),
        memory_tokens=int(args.memory_tokens),
        candidate_names=config.candidate_names,
        use_evidence_router=config.use_evidence_router,
        use_reliability_prior=config.use_reliability_prior,
        lrio_pairs=config.lrio_pairs or None,
        **_ovha_composition_kwargs(config, config.candidate_names),
    ).to(device)
    initial_parameters = _parameter_vector(model)
    fit_summary = _fit_public_ovha_model(
        config,
        model,
        fit_batch_std=fit_batch_std,
        val_batch_std=val_batch_std,
        seed=seed,
        sampling_seed=int(seed) + 17,
        model_name=config.main_model_name,
        train_steps=int(args.train_steps),
        learning_rate=float(args.learning_rate),
        raw_val_batch=selection_batch,
        target_mean=target_mean,
        target_std=target_std,
        progress_interval=progress_interval,
        batch_size=int(args.batch_size),
        eval_interval=int(args.eval_interval),
        early_stopping_patience=int(args.early_stopping_patience),
        weight_decay=float(args.weight_decay),
        device=device,
    )
    model.eval()
    with torch.no_grad():
        inference_batch_size = _eval_batch_size(int(args.batch_size))
        val_output = _destandardize_output(
            _predict_ovha_on_device(model, val_batch_std, device, batch_size=inference_batch_size),
            target_mean,
            target_std,
        )
        calibrator = _fit_task_calibrator(config, val_output.y_hat.detach().cpu(), selection_batch)
        eval_output = _destandardize_output(
            _predict_ovha_on_device(model, eval_batch_std, device, batch_size=inference_batch_size),
            target_mean,
            target_std,
        )
        eval_output = _with_raw_space_candidate_diagnostics(eval_output, eval_batch_std, eval_batch)
        eval_output = _apply_task_calibration_to_output(eval_output, calibrator, config, eval_batch)
    elapsed = time.perf_counter() - seed_started_at
    hardware = _hardware_metadata(device, elapsed)
    raw_rows = [
        _ovha_raw_metric_row(
            config,
            eval_batch,
            eval_output,
            seed=seed,
            training_steps=int(fit_summary["optimizer_steps"]),
            parameter_count=_parameter_count(model),
            raw_metrics_path=raw_metrics_path,
            hardware=hardware,
            model_name=config.main_model_name,
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
        model_name=config.main_model_name,
        raw_metric_path=raw_metrics_path,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        batch_size=int(args.batch_size),
    )
    per_sample_prediction_rows = _per_sample_prediction_rows(
        config,
        eval_batch,
        eval_output,
        model_name=config.main_model_name,
        seed=seed,
    )

    baseline_rows, baseline_robustness_rows, baseline_summaries, baseline_prediction_rows = _baseline_rows(
        config,
        train_batch=train_batch,
        selection_batch=selection_batch,
        eval_batch=eval_batch,
        seed=seed,
        baseline_train_steps=int(args.baseline_train_steps),
        learning_rate=float(args.learning_rate),
        hardware=hardware,
        raw_metrics_path=raw_metrics_path,
        progress_interval=progress_interval,
        d_model=int(args.d_model),
        memory_tokens=int(args.memory_tokens),
        batch_size=int(args.batch_size),
        validation_fraction=float(args.validation_fraction),
        eval_interval=int(args.eval_interval),
        early_stopping_patience=int(args.early_stopping_patience),
        weight_decay=float(args.weight_decay),
        device=device,
    )
    raw_rows.extend(baseline_rows)
    robustness_rows.extend(baseline_robustness_rows)
    per_sample_prediction_rows.extend(baseline_prediction_rows)
    _print_progress(
        "seed:done",
        seed=seed,
        elapsed=f"{time.perf_counter() - seed_started_at:.1f}s",
        raw_rows=len(raw_rows),
        robustness_rows=len(robustness_rows),
    )
    return {
        "raw_rows": raw_rows,
        "diagnostics_rows": diagnostics_rows,
        "robustness_rows": robustness_rows,
        "per_sample_prediction_rows": per_sample_prediction_rows,
        "summary": {
            "seed": seed,
            "ovha_parameter_l2_delta": float(torch.linalg.vector_norm(_parameter_vector(model) - initial_parameters).item()),
            "ovha_max_grad_norm": float(fit_summary["max_grad_norm"]),
            "ovha_best_val_task_loss": float(fit_summary["best_val_loss"]),
            "ovha_best_val_selection_score": float(fit_summary["best_val_score"]),
            "ovha_best_checkpoint_step": int(fit_summary["best_step"]),
            "ovha_training_protocol": str(fit_summary["training_protocol"]),
            "ovha_checkpoint_selection_protocol": "official_val_selection_best_checkpoint",
            "ovha_checkpoint_selection_metric": str(fit_summary["selection_metric"]),
            "ovha_optimizer_steps": int(fit_summary["optimizer_steps"]),
            "ovha_stage_history": fit_summary["stage_history"],
            "ovha_lr_schedule": str(fit_summary["lr_schedule"]),
            "ovha_warmup_steps": int(fit_summary["warmup_steps"]),
            "ovha_min_lr_ratio": float(fit_summary["min_lr_ratio"]),
            "ovha_batch_size": int(args.batch_size),
            "ovha_selection_split": args.selection_split,
            "ovha_target_standardized": True,
            "baseline_count": len(baseline_rows),
            "baseline_summaries": baseline_summaries,
        },
    }


def _load_public_batch_with_progress(
    layout: MultimodalCacheLayout,
    config: MultimodalExperimentConfig,
    split: str,
    device: torch.device,
    seed: int,
) -> MultimodalEpisodeBatch:
    started_at = time.perf_counter()
    _print_progress("seed:load:start", seed=seed, split=split, device=str(device))
    batch = _load_public_batch(layout, config, split, device)
    _print_progress(
        "seed:load:done",
        seed=seed,
        split=split,
        samples=int(batch.target_y.shape[0]),
        elapsed=f"{time.perf_counter() - started_at:.1f}s",
    )
    return batch


def _fit_public_ovha_model(
    config: MultimodalExperimentConfig,
    model: MultimodalOVHA,
    *,
    fit_batch_std: MultimodalEpisodeBatch,
    val_batch_std: MultimodalEpisodeBatch,
    seed: int,
    sampling_seed: int,
    model_name: str,
    train_steps: int,
    learning_rate: float,
    raw_val_batch: MultimodalEpisodeBatch,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
    progress_interval: int,
    batch_size: int,
    eval_interval: int,
    early_stopping_patience: int,
    weight_decay: float,
    device: torch.device,
) -> dict[str, Any]:
    return _fit_single_stage_ovha_model(
        config,
        model,
        fit_batch_std=fit_batch_std,
        val_batch_std=val_batch_std,
        seed=seed,
        sampling_seed=sampling_seed,
        model_name=model_name,
        train_steps=train_steps,
        learning_rate=learning_rate,
        raw_val_batch=raw_val_batch,
        target_mean=target_mean,
        target_std=target_std,
        progress_interval=progress_interval,
        batch_size=batch_size,
        eval_interval=eval_interval,
        early_stopping_patience=early_stopping_patience,
        weight_decay=weight_decay,
        device=device,
    )


def _fit_single_stage_ovha_model(
    config: MultimodalExperimentConfig,
    model: MultimodalOVHA,
    *,
    fit_batch_std: MultimodalEpisodeBatch,
    val_batch_std: MultimodalEpisodeBatch,
    seed: int,
    sampling_seed: int,
    model_name: str,
    train_steps: int,
    learning_rate: float,
    raw_val_batch: MultimodalEpisodeBatch,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
    progress_interval: int,
    batch_size: int,
    eval_interval: int,
    early_stopping_patience: int,
    weight_decay: float,
    device: torch.device,
) -> dict[str, Any]:
    return _run_ovha_training_stages(
        config,
        model,
        stages=(
            {
                "stage": "joint",
                "steps": int(train_steps),
                "loss_mode": "public",
                "trainable_scope": "all",
            },
        ),
        fit_batch_std=fit_batch_std,
        val_batch_std=val_batch_std,
        seed=seed,
        sampling_seed=sampling_seed,
        model_name=model_name,
        train_steps=train_steps,
        learning_rate=learning_rate,
        raw_val_batch=raw_val_batch,
        target_mean=target_mean,
        target_std=target_std,
        progress_interval=progress_interval,
        batch_size=batch_size,
        eval_interval=eval_interval,
        early_stopping_patience=early_stopping_patience,
        weight_decay=weight_decay,
        device=device,
        training_protocol="mini_batch_validation_best_checkpoint",
    )


def _run_ovha_training_stages(
    config: MultimodalExperimentConfig,
    model: MultimodalOVHA,
    *,
    stages: tuple[dict[str, Any], ...],
    fit_batch_std: MultimodalEpisodeBatch,
    val_batch_std: MultimodalEpisodeBatch,
    seed: int,
    sampling_seed: int,
    model_name: str,
    train_steps: int,
    learning_rate: float,
    raw_val_batch: MultimodalEpisodeBatch,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
    progress_interval: int,
    batch_size: int,
    eval_interval: int,
    early_stopping_patience: int,
    weight_decay: float,
    device: torch.device,
    training_protocol: str,
) -> dict[str, Any]:
    max_grad_norm = 0.0
    final_loss = 0.0
    best_val_loss = float("inf")
    best_val_score = float("inf")
    best_step = 0
    best_state = _clone_state_dict(model)
    optimizer_steps = 0
    stage_history: list[dict[str, Any]] = []
    batch_generator = torch.Generator(device=fit_batch_std.target_y.device)
    batch_generator.manual_seed(int(sampling_seed))
    started_at = time.perf_counter()
    _print_progress(
        "model:start",
        seed=seed,
        model=model_name,
        steps=train_steps,
        learning_rate=learning_rate,
        lr_schedule=config.lr_schedule,
        selection_metric=config.checkpoint_selection_metric,
        protocol=training_protocol,
    )
    for stage in stages:
        stage_name = str(stage["stage"])
        stage_steps = int(stage["steps"])
        if stage_steps <= 0:
            continue
        _set_all_parameters_trainable(model)
        trainable_count = _trainable_parameter_count(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        stage_started_at = time.perf_counter()
        stage_grad_norm = 0.0
        stage_loss = 0.0
        stage_best_val_loss = float("inf")
        stage_best_val_score = float("inf")
        stage_best_step = 0
        stale_evals = 0
        model.train()
        _print_progress(
            "model:stage_start",
            seed=seed,
            model=model_name,
            stage=stage_name,
            steps=stage_steps,
            trainable_parameters=trainable_count,
        )
        for stage_step in range(1, stage_steps + 1):
            optimizer_steps += 1
            current_lr = _lr_for_step(
                optimizer_steps,
                learning_rate,
                config.warmup_steps,
                train_steps,
                config.min_lr_ratio,
            ) if config.lr_schedule == "warmup_cosine" else learning_rate
            _set_optimizer_lr(optimizer, current_lr)
            train_step_batch = _move_batch_to_device(_sample_batch(fit_batch_std, batch_size, batch_generator), device)
            optimizer.zero_grad(set_to_none=True)
            output = model(train_step_batch)
            components = _public_loss_components(output, train_step_batch, config)
            total_loss = torch.stack([value for value in components.values()]).sum()
            total_loss.backward()
            grad_norm = _grad_l2_norm(model)
            max_grad_norm = max(max_grad_norm, grad_norm)
            stage_grad_norm = max(stage_grad_norm, grad_norm)
            optimizer.step()
            final_loss = _as_float(total_loss)
            stage_loss = final_loss
            if _should_validate(optimizer_steps, train_steps, eval_interval) or stage_step == stage_steps:
                val_loss = _evaluate_task_loss_on_device(
                    model,
                    val_batch_std,
                    device,
                    batch_size=_eval_batch_size(batch_size),
                )
                val_score = _evaluate_selection_score_on_device(
                    config,
                    model,
                    val_batch_std=val_batch_std,
                    raw_val_batch=raw_val_batch,
                    target_mean=target_mean,
                    target_std=target_std,
                    device=device,
                    batch_size=_eval_batch_size(batch_size),
                )
                if val_score < best_val_score:
                    best_val_score = val_score
                    best_val_loss = val_loss
                    best_step = optimizer_steps
                    best_state = _clone_state_dict(model)
                    stale_evals = 0
                else:
                    stale_evals += 1
                if val_score < stage_best_val_score:
                    stage_best_val_score = val_score
                    stage_best_val_loss = val_loss
                    stage_best_step = optimizer_steps
                if early_stopping_patience > 0 and stale_evals >= early_stopping_patience:
                    _print_progress(
                        "model:early_stop",
                        seed=seed,
                        model=model_name,
                        stage=stage_name,
                        step=optimizer_steps,
                        best_step=best_step,
                        best_val_loss=f"{best_val_loss:.6g}",
                    )
                    break
            if _should_log_progress(optimizer_steps, train_steps, progress_interval):
                _print_step_progress(
                    seed=seed,
                    model=model_name,
                    step=optimizer_steps,
                    total_steps=train_steps,
                    loss=final_loss,
                    started_at=started_at,
                )
        stage_history.append(
            {
                "stage": stage_name,
                "optimizer_steps": stage_step,
                "global_optimizer_step_end": optimizer_steps,
                "loss_mode": str(stage.get("loss_mode")),
                "trainable_scope": str(stage.get("trainable_scope")),
                "trainable_parameter_count": trainable_count,
                "max_grad_norm": stage_grad_norm,
                "final_loss": stage_loss,
                "best_val_task_loss": stage_best_val_loss,
                "best_val_selection_score": stage_best_val_score,
                "best_checkpoint_step": stage_best_step,
                "selection_metric": config.checkpoint_selection_metric,
                "lr_schedule": config.lr_schedule,
                "warmup_steps": int(config.warmup_steps),
                "min_lr_ratio": float(config.min_lr_ratio),
                "elapsed_seconds": time.perf_counter() - stage_started_at,
            }
        )
    model.load_state_dict(best_state)
    _set_all_parameters_trainable(model)
    _print_progress(
        "model:done",
        seed=seed,
        model=model_name,
        steps=optimizer_steps,
        loss=f"{final_loss:.6g}",
        elapsed=f"{time.perf_counter() - started_at:.1f}s",
        protocol=training_protocol,
    )
    return {
        "optimizer_steps": optimizer_steps,
        "max_grad_norm": max_grad_norm,
        "final_loss": final_loss,
        "best_val_loss": best_val_loss,
        "best_val_score": best_val_score,
        "best_step": best_step,
        "selection_metric": config.checkpoint_selection_metric,
        "lr_schedule": config.lr_schedule,
        "warmup_steps": int(config.warmup_steps),
        "min_lr_ratio": float(config.min_lr_ratio),
        "training_protocol": training_protocol,
        "stage_history": stage_history,
    }


def _set_all_parameters_trainable(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def _trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)


def _split_train_val_batch(
    batch: MultimodalEpisodeBatch,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[MultimodalEpisodeBatch, MultimodalEpisodeBatch]:
    batch_size = int(batch.target_y.shape[0])
    if batch_size <= 1 or validation_fraction <= 0.0:
        return batch, batch
    val_count = max(1, min(batch_size - 1, int(round(batch_size * validation_fraction))))
    generator = torch.Generator(device=batch.target_y.device)
    generator.manual_seed(int(seed) + 104729)
    order = torch.randperm(batch_size, generator=generator, device=batch.target_y.device)
    val_indices = order[:val_count]
    fit_indices = order[val_count:]
    return _slice_batch(batch, fit_indices), _slice_batch(batch, val_indices)


def _sample_batch(
    batch: MultimodalEpisodeBatch,
    batch_size: int,
    generator: torch.Generator,
) -> MultimodalEpisodeBatch:
    total = int(batch.target_y.shape[0])
    if batch_size <= 0 or batch_size >= total:
        return batch
    indices = torch.randint(total, (batch_size,), generator=generator, device=batch.target_y.device)
    return _slice_batch(batch, indices)


def _slice_batch(batch: MultimodalEpisodeBatch, indices: torch.Tensor) -> MultimodalEpisodeBatch:
    fields = {
        name: replace(
            field,
            x=field.x.index_select(0, indices),
            pos=field.pos.index_select(0, indices),
            mask=field.mask.index_select(0, indices),
            quality=_slice_optional_tensor(field.quality, indices),
        )
        for name, field in batch.fields.items()
    }
    query = replace(
        batch.query,
        x=batch.query.x.index_select(0, indices),
        pos=batch.query.pos.index_select(0, indices),
        query_type=batch.query.query_type.index_select(0, indices),
        mask=batch.query.mask.index_select(0, indices),
    )
    supervision = replace(
        batch.supervision,
        task_label=_slice_optional_tensor(batch.supervision.task_label, indices),
        alignment_pairs=_slice_optional_tensor(batch.supervision.alignment_pairs, indices),
        alignment_weights=_slice_optional_tensor(batch.supervision.alignment_weights, indices),
        bbox_targets=_slice_optional_tensor(batch.supervision.bbox_targets, indices),
        region_targets=_slice_optional_tensor(batch.supervision.region_targets, indices),
        timestamp_targets=_slice_optional_tensor(batch.supervision.timestamp_targets, indices),
        modality_missing_mask=_slice_optional_tensor(batch.supervision.modality_missing_mask, indices),
        corruption_metadata=_slice_optional_tensor_dict(batch.supervision.corruption_metadata, indices),
        weak_labels=_slice_optional_tensor_dict(batch.supervision.weak_labels, indices),
        weak_label_confidence=_slice_optional_tensor_dict(batch.supervision.weak_label_confidence, indices),
    )
    provenance_indices = [int(index) for index in indices.detach().cpu().tolist()]
    provenance = replace(
        batch.provenance,
        source_id=[batch.provenance.source_id[index] for index in provenance_indices],
        original_split=[batch.provenance.original_split[index] for index in provenance_indices],
        raw_ref=[batch.provenance.raw_ref[index] for index in provenance_indices],
        license_tag=[batch.provenance.license_tag[index] for index in provenance_indices],
    )
    hidden = _slice_hidden(batch.hidden, indices)
    return replace(
        batch,
        fields=fields,
        query=query,
        target_y=batch.target_y.index_select(0, indices),
        target_mask=batch.target_mask.index_select(0, indices),
        supervision=supervision,
        provenance=provenance,
        hidden=hidden,
    )


def _move_batch_to_device(batch: MultimodalEpisodeBatch, device: torch.device) -> MultimodalEpisodeBatch:
    fields = {
        name: replace(
            field,
            x=_move_nested_to_device(field.x, device),
            pos=_move_nested_to_device(field.pos, device),
            mask=_move_nested_to_device(field.mask, device),
            quality=_move_nested_to_device(field.quality, device),
        )
        for name, field in batch.fields.items()
    }
    query = replace(
        batch.query,
        x=_move_nested_to_device(batch.query.x, device),
        pos=_move_nested_to_device(batch.query.pos, device),
        query_type=_move_nested_to_device(batch.query.query_type, device),
        mask=_move_nested_to_device(batch.query.mask, device),
    )
    supervision = replace(
        batch.supervision,
        task_label=_move_nested_to_device(batch.supervision.task_label, device),
        alignment_pairs=_move_nested_to_device(batch.supervision.alignment_pairs, device),
        alignment_weights=_move_nested_to_device(batch.supervision.alignment_weights, device),
        bbox_targets=_move_nested_to_device(batch.supervision.bbox_targets, device),
        region_targets=_move_nested_to_device(batch.supervision.region_targets, device),
        timestamp_targets=_move_nested_to_device(batch.supervision.timestamp_targets, device),
        modality_missing_mask=_move_nested_to_device(batch.supervision.modality_missing_mask, device),
        corruption_metadata=_move_nested_to_device(batch.supervision.corruption_metadata, device),
        weak_labels=_move_nested_to_device(batch.supervision.weak_labels, device),
        weak_label_confidence=_move_nested_to_device(batch.supervision.weak_label_confidence, device),
    )
    return replace(
        batch,
        fields=fields,
        query=query,
        target_y=_move_nested_to_device(batch.target_y, device),
        target_mask=_move_nested_to_device(batch.target_mask, device),
        supervision=supervision,
        hidden=_move_nested_to_device(batch.hidden, device),
    )


def _move_ovha_output_to_device(output: MultimodalOVHAOutput, device: torch.device) -> MultimodalOVHAOutput:
    return _move_nested_to_device(output, device)


def _move_nested_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _move_nested_to_device(getattr(value, field.name), device)
                for field in dataclass_fields(value)
            },
        )
    return value


def _slice_optional_tensor(value: Any, indices: torch.Tensor) -> Any:
    if value is None or not hasattr(value, "index_select"):
        return value
    return value.index_select(0, indices)


def _slice_optional_tensor_dict(values: dict[str, Any] | None, indices: torch.Tensor) -> dict[str, Any] | None:
    if values is None:
        return None
    return {key: _slice_optional_tensor(value, indices) for key, value in values.items()}


def _slice_hidden(hidden: dict[str, Any] | None, indices: torch.Tensor) -> dict[str, Any] | None:
    if hidden is None:
        return None
    return {key: _slice_optional_tensor(value, indices) for key, value in hidden.items()}


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _evaluate_task_loss(model: MultimodalOVHA, batch: MultimodalEpisodeBatch) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = _task_loss(model(batch).y_hat, batch)
    if was_training:
        model.train()
    return _as_float(loss)


def _evaluate_task_loss_on_device(
    model: MultimodalOVHA,
    batch: MultimodalEpisodeBatch,
    device: torch.device,
    *,
    batch_size: int,
) -> float:
    total = int(batch.target_y.shape[0])
    if total <= 0:
        return 0.0
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, total, max(1, int(batch_size))):
            stop = min(total, start + max(1, int(batch_size)))
            indices = torch.arange(start, stop, device=batch.target_y.device)
            chunk = _move_batch_to_device(_slice_batch(batch, indices), device)
            loss = _task_loss(model(chunk).y_hat, chunk)
            weight = float(chunk.target_mask.to(dtype=torch.float32).sum().detach().cpu())
            weighted_loss += _as_float(loss) * weight
            total_weight += weight
    if was_training:
        model.train()
    return weighted_loss / max(1.0, total_weight)


def _evaluate_selection_score_on_device(
    config: MultimodalExperimentConfig,
    model: MultimodalOVHA,
    *,
    val_batch_std: MultimodalEpisodeBatch,
    raw_val_batch: MultimodalEpisodeBatch,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
    device: torch.device,
    batch_size: int,
) -> float:
    if config.checkpoint_selection_metric == "standardized_mse":
        return _evaluate_task_loss_on_device(model, val_batch_std, device, batch_size=batch_size)
    if config.checkpoint_selection_metric != "mosei_composite":
        raise ValueError(f"unknown checkpoint_selection_metric: {config.checkpoint_selection_metric}")
    output_std = _predict_ovha_on_device(model, val_batch_std, device, batch_size=batch_size)
    output_raw = _destandardize_output(output_std, target_mean, target_std)
    metrics = mosei_standard_metrics(output_raw.y_hat, raw_val_batch.target_y, raw_val_batch.target_mask)
    return _mosei_composite_score(metrics, config.checkpoint_selection_weights or {})


def _mosei_composite_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return (
        float(weights["mae"]) * float(metrics["mae"])
        + float(weights["pearson_correlation"]) * (1.0 - float(metrics["pearson_correlation"]))
        + float(weights["acc7"]) * (1.0 - float(metrics["acc7"]))
        + float(weights["acc5"]) * (1.0 - float(metrics["acc5"]))
        + float(weights["acc2_excl0"]) * (1.0 - float(metrics["acc2_excl0"]))
        + float(weights["acc2_nonneg"]) * (1.0 - float(metrics["acc2_nonneg"]))
    )


def _lr_for_step(
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    step = max(0, int(step))
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * float(step) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
    progress = max(0.0, min(1.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def _evaluate_linear_task_loss_on_device(
    baseline_name: str,
    model: torch.nn.Linear,
    batch: MultimodalEpisodeBatch,
    device: torch.device,
    *,
    batch_size: int,
) -> float:
    total = int(batch.target_y.shape[0])
    if total <= 0:
        return 0.0
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, total, max(1, int(batch_size))):
            stop = min(total, start + max(1, int(batch_size)))
            indices = torch.arange(start, stop, device=batch.target_y.device)
            chunk = _move_batch_to_device(_slice_batch(batch, indices), device)
            loss = _task_loss(_baseline_prediction(baseline_name, model, chunk), chunk)
            weight = float(chunk.target_mask.to(dtype=torch.float32).sum().detach().cpu())
            weighted_loss += _as_float(loss) * weight
            total_weight += weight
    if was_training:
        model.train()
    return weighted_loss / max(1.0, total_weight)


def _predict_ovha_on_device(
    model: MultimodalOVHA,
    batch: MultimodalEpisodeBatch,
    device: torch.device,
    *,
    batch_size: int,
) -> MultimodalOVHAOutput:
    total = int(batch.target_y.shape[0])
    if total <= 0:
        raise ValueError("cannot run OVHA inference on an empty batch")
    was_training = model.training
    model.eval()
    outputs: list[MultimodalOVHAOutput] = []
    sizes: list[int] = []
    with torch.no_grad():
        for start in range(0, total, max(1, int(batch_size))):
            stop = min(total, start + max(1, int(batch_size)))
            indices = torch.arange(start, stop, device=batch.target_y.device)
            chunk = _move_batch_to_device(_slice_batch(batch, indices), device)
            output = _move_ovha_output_to_device(model(chunk), torch.device("cpu"))
            outputs.append(output)
            sizes.append(int(stop - start))
    if was_training:
        model.train()
    return _merge_ovha_outputs(outputs, sizes)


def _merge_ovha_outputs(outputs: list[MultimodalOVHAOutput], sizes: list[int]) -> MultimodalOVHAOutput:
    if not outputs:
        raise ValueError("cannot merge empty OVHA output list")
    first = outputs[0]
    candidate_names = tuple(first.candidate_outputs)
    return MultimodalOVHAOutput(
        y_hat=torch.cat([output.y_hat for output in outputs], dim=0),
        candidate_values=torch.cat([output.candidate_values for output in outputs], dim=0),
        router_weights=torch.cat([output.router_weights for output in outputs], dim=0),
        router_logits=torch.cat([output.router_logits for output in outputs], dim=0),
        router_logit_parts=_merge_chunk_values([output.router_logit_parts for output in outputs], sizes),
        candidate_outputs={
            name: _merge_candidate_outputs([output.candidate_outputs[name] for output in outputs], sizes)
            for name in candidate_names
        },
        reliability_prior=None,
        diagnostics=_merge_chunk_values([output.diagnostics for output in outputs], sizes),
        evidence=None,
    )


def _merge_candidate_outputs(outputs: list[CandidateOutput], sizes: list[int]) -> CandidateOutput:
    return CandidateOutput(
        value=torch.cat([output.value for output in outputs], dim=0),
        feature=torch.cat([output.feature for output in outputs], dim=0),
        diagnostics=_merge_chunk_values([output.diagnostics for output in outputs], sizes),
    )


def _merge_chunk_values(values: list[Any], sizes: list[int]) -> Any:
    first = values[0]
    if torch.is_tensor(first):
        tensors = [value.detach().cpu() for value in values]
        if first.ndim > 0 and all(tensor.shape[:1] == (size,) for tensor, size in zip(tensors, sizes)):
            return torch.cat(tensors, dim=0)
        if all(tensor.numel() == 1 for tensor in tensors):
            total = float(sum(sizes))
            return sum(tensor.reshape(()) * (float(size) / total) for tensor, size in zip(tensors, sizes))
        return tensors[0]
    if isinstance(first, dict):
        keys = set().union(*(value.keys() for value in values if isinstance(value, dict)))
        merged: dict[str, Any] = {}
        for key in keys:
            key_pairs = [
                (value[key], size)
                for value, size in zip(values, sizes)
                if isinstance(value, dict) and key in value
            ]
            merged[key] = _merge_chunk_values(
                [value for value, _ in key_pairs],
                [size for _, size in key_pairs],
            )
        return merged
    if isinstance(first, (int, float)):
        total = float(sum(sizes))
        return sum(float(value) * (float(size) / total) for value, size in zip(values, sizes))
    return first


def _eval_batch_size(train_batch_size: int) -> int:
    return max(1, int(train_batch_size) * 8)


def _should_validate(step: int, train_steps: int, eval_interval: int) -> bool:
    interval = max(1, int(eval_interval))
    return step == 1 or step == train_steps or step % interval == 0


def _target_standardizer(batch: MultimodalEpisodeBatch) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if _is_region_task(batch.task_type):
        return None, None
    mask = batch.target_mask.to(dtype=batch.target_y.dtype, device=batch.target_y.device).unsqueeze(-1)
    denom = mask.sum(dim=(0, 1), keepdim=True).clamp_min(1.0)
    mean = (batch.target_y * mask).sum(dim=(0, 1), keepdim=True) / denom
    variance = ((batch.target_y - mean).square() * mask).sum(dim=(0, 1), keepdim=True) / denom
    std = variance.sqrt().clamp_min(1e-6)
    return mean.detach(), std.detach()


def _standardize_batch_targets(
    batch: MultimodalEpisodeBatch,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
) -> MultimodalEpisodeBatch:
    if target_mean is None or target_std is None:
        return batch
    target_y = (batch.target_y - target_mean) / target_std
    supervision = replace(
        batch.supervision,
        task_label=_standardize_optional_tensor(batch.supervision.task_label, target_mean, target_std),
    )
    return replace(batch, target_y=target_y, supervision=supervision)


def _standardize_optional_tensor(value: Any, target_mean: torch.Tensor, target_std: torch.Tensor) -> Any:
    if value is None or not hasattr(value, "shape"):
        return value
    try:
        return (value - target_mean) / target_std
    except RuntimeError:
        return value


def _destandardize_output(
    output: MultimodalOVHAOutput,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
) -> MultimodalOVHAOutput:
    if target_mean is None or target_std is None:
        return output
    return _transform_ovha_output_target_space(
        output,
        prediction_transform=lambda value: value * target_std + target_mean,
        delta_transform=lambda value: value * target_std,
        target_space={
            "prediction": "raw",
            "candidate_values": "raw",
            "candidate_outputs": "raw_prediction_or_delta",
            "residual_delta": "raw_delta",
        },
    )


def _transform_ovha_output_target_space(
    output: MultimodalOVHAOutput,
    *,
    prediction_transform: Any,
    delta_transform: Any,
    target_space: dict[str, Any],
) -> MultimodalOVHAOutput:
    composition = output.diagnostics.get("composition", {}) if isinstance(output.diagnostics, dict) else {}
    mode = composition.get("mode")
    base_candidate = str(composition.get("base_candidate", ""))
    residual_candidates = {str(name) for name in composition.get("residual_candidates", ())}
    transformed_candidate_outputs = {
        name: replace(
            candidate_output,
            value=(
                delta_transform(candidate_output.value)
                if mode == "base_plus_residual" and name in residual_candidates and name != base_candidate
                else prediction_transform(candidate_output.value)
            ),
        )
        for name, candidate_output in output.candidate_outputs.items()
    }
    transformed_composition = _transform_composition_target_space(
        composition,
        prediction_transform=prediction_transform,
        delta_transform=delta_transform,
    )
    diagnostics = {
        **output.diagnostics,
        "composition": transformed_composition,
        "target_space": {
            **output.diagnostics.get("target_space", {}),
            **target_space,
        },
    }
    return replace(
        output,
        y_hat=prediction_transform(output.y_hat),
        candidate_values=prediction_transform(output.candidate_values),
        candidate_outputs=transformed_candidate_outputs,
        diagnostics=diagnostics,
    )


def _transform_composition_target_space(
    composition: Any,
    *,
    prediction_transform: Any,
    delta_transform: Any,
) -> Any:
    if not isinstance(composition, dict):
        return composition
    transformed = dict(composition)
    for key in ("raw_delta_by_candidate", "gated_delta_by_candidate"):
        transformed[key] = _transform_tensor_dict(transformed.get(key), delta_transform)
    for key in ("gated_corrected_candidate_values_by_candidate", "ungated_corrected_candidate_values_by_candidate"):
        transformed[key] = _transform_tensor_dict(transformed.get(key), prediction_transform)
    return transformed


def _transform_tensor_dict(values: Any, transform: Any) -> Any:
    if not isinstance(values, dict):
        return values
    return {
        key: transform(value) if hasattr(value, "shape") else value
        for key, value in values.items()
    }


def _with_raw_space_candidate_diagnostics(
    output: MultimodalOVHAOutput,
    standardized_batch: MultimodalEpisodeBatch,
    raw_batch: MultimodalEpisodeBatch,
) -> MultimodalOVHAOutput:
    candidate_names = tuple(output.candidate_outputs)
    raw_losses = _candidate_losses_from_values(output.candidate_values, candidate_names, raw_batch)
    standardized_losses = output.diagnostics.get("candidate_loss", {})
    diagnostics = {
        **output.diagnostics,
        "standardized_candidate_loss": standardized_losses,
        "raw_candidate_loss": raw_losses,
        "candidate_loss": raw_losses,
        "raw_candidate_value_stats": _candidate_value_stats_from_values(output.candidate_values, candidate_names),
    }
    if "candidate_value_stats" in output.diagnostics:
        diagnostics["standardized_candidate_value_stats"] = output.diagnostics["candidate_value_stats"]
    candidate_diagnostics = output.diagnostics.get("candidate_diagnostics")
    if isinstance(candidate_diagnostics, dict):
        diagnostics["candidate_diagnostics"] = {
            name: _raw_candidate_diagnostic_entry(candidate_diagnostics.get(name, {}), name, raw_losses, standardized_losses)
            for name in candidate_diagnostics
        }
    if "candidate_loss_by_sample" in output.diagnostics:
        diagnostics["standardized_candidate_loss_by_sample"] = output.diagnostics["candidate_loss_by_sample"]
        diagnostics["raw_candidate_loss_by_sample"] = _candidate_losses_by_sample_from_values(
            output.candidate_values,
            candidate_names,
            raw_batch,
        )
    diagnostics["target_space"] = {
        "prediction": "raw",
        "candidate_values": "raw",
        "candidate_loss": "raw",
        "standardized_reference_available": True,
        "standardized_target_mean": _as_float(standardized_batch.target_y.mean()),
    }
    return replace(output, diagnostics=diagnostics)


def _raw_candidate_diagnostic_entry(
    values: Any,
    name: str,
    raw_losses: dict[str, torch.Tensor],
    standardized_losses: Any,
) -> dict[str, Any]:
    entry = dict(values) if isinstance(values, dict) else {}
    if name in raw_losses:
        entry["raw_candidate_loss"] = raw_losses[name]
        entry["candidate_loss"] = raw_losses[name]
    if isinstance(standardized_losses, dict) and name in standardized_losses:
        entry["standardized_candidate_loss"] = standardized_losses[name]
    return entry


def _candidate_losses_from_values(
    candidate_values: torch.Tensor,
    candidate_names: tuple[str, ...],
    batch: MultimodalEpisodeBatch,
) -> dict[str, torch.Tensor]:
    if _is_region_task(batch.task_type):
        return {
            name: _task_loss(candidate_values[..., index, :], batch)
            for index, name in enumerate(candidate_names)
        }
    mask = batch.target_mask.to(device=candidate_values.device, dtype=candidate_values.dtype).unsqueeze(-1)
    truth = batch.target_y.to(device=candidate_values.device, dtype=candidate_values.dtype).unsqueeze(-2)
    losses: dict[str, torch.Tensor] = {}
    for index, name in enumerate(candidate_names):
        error = (candidate_values[..., index, :] - truth[..., 0, :]).square()
        losses[name] = (error * mask).sum() / mask.sum().clamp_min(1.0)
    return losses


def _candidate_losses_by_sample_from_values(
    candidate_values: torch.Tensor,
    candidate_names: tuple[str, ...],
    batch: MultimodalEpisodeBatch,
) -> dict[str, torch.Tensor]:
    mask = batch.target_mask.to(device=candidate_values.device, dtype=candidate_values.dtype)
    losses: dict[str, torch.Tensor] = {}
    if _is_region_task(batch.task_type):
        labels = batch.supervision.region_targets
        if labels is None:
            return {name: torch.zeros(mask.shape[0], dtype=candidate_values.dtype, device=candidate_values.device) for name in candidate_names}
        labels = labels.to(device=candidate_values.device, dtype=torch.long)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        if labels.ndim > 2:
            labels = labels.reshape(labels.shape[0], -1)
        if labels.shape[1] == 1 and candidate_values.shape[1] > 1:
            labels = labels.expand(-1, candidate_values.shape[1])
        labels = labels[:, : candidate_values.shape[1]].contiguous()
        for index, name in enumerate(candidate_names):
            logits = candidate_values[..., index, :]
            ce = torch.nn.functional.cross_entropy(
                logits[:, : labels.shape[1], :].reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            losses[name] = (ce * mask[:, : labels.shape[1]]).sum(dim=1) / mask[:, : labels.shape[1]].sum(dim=1).clamp_min(1.0)
        return losses
    truth = batch.target_y.to(device=candidate_values.device, dtype=candidate_values.dtype)
    for index, name in enumerate(candidate_names):
        error = (candidate_values[..., index, :] - truth).square().mean(dim=-1)
        losses[name] = (error * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return losses


def _candidate_value_stats_from_values(
    candidate_values: torch.Tensor,
    candidate_names: tuple[str, ...],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "mean": candidate_values[..., index, :].mean(),
            "std": candidate_values[..., index, :].std(unbiased=False),
        }
        for index, name in enumerate(candidate_names)
    }


def _destandardize_prediction(
    prediction: torch.Tensor,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
) -> torch.Tensor:
    if target_mean is None or target_std is None:
        return prediction
    return prediction * target_std + target_mean


def _fit_task_calibrator(
    config: MultimodalExperimentConfig,
    prediction: torch.Tensor,
    batch: MultimodalEpisodeBatch,
) -> dict[str, Any]:
    if _is_region_task(config.task_type):
        return {
            "method": "none",
            "objective": "region_classification_logits_uncalibrated",
            "sample_count": int(batch.target_mask.to(dtype=torch.bool).sum().item()),
            "metrics_scope": METRICS_SOURCE,
        }
    mse = _fit_affine_calibrator(prediction, batch)
    huber = _fit_affine_huber_calibrator(prediction, batch)
    composite = _fit_affine_mosei_composite_calibrator(
        prediction,
        batch,
        weights=config.checkpoint_selection_weights,
    )
    selected = (
        "composite_affine_calibration"
        if config.checkpoint_selection_metric == "mosei_composite"
        else "mse_affine_calibration"
    )
    calibrators = {
        "mse_affine_calibration": mse,
        "huber_affine_calibration": huber,
        "composite_affine_calibration": composite,
    }
    return {
        "method": "affine_bank",
        "selected": selected,
        "calibrators": calibrators,
        "validation_metrics": {
            "pre_calibration": mosei_standard_metrics(prediction, batch.target_y, batch.target_mask),
            **{
                name: mosei_standard_metrics(
                    _apply_affine_calibration_to_prediction(prediction, calibrator),
                    batch.target_y,
                    batch.target_mask,
                )
                for name, calibrator in calibrators.items()
            },
        },
        "sample_count": int(batch.target_mask.to(dtype=torch.bool).sum().item()),
    }


def _fit_affine_calibrator(
    prediction: torch.Tensor,
    batch: MultimodalEpisodeBatch,
) -> dict[str, float]:
    pred, truth = _masked_flat_pair(prediction, batch.target_y, batch.target_mask)
    if pred.numel() < 2:
        return {"a": 1.0, "b": 0.0, "objective": "validation_mse_closed_form", "sample_count": int(pred.numel())}
    pred = pred.to(dtype=torch.float64)
    truth = truth.to(dtype=torch.float64)
    pred_centered = pred - pred.mean()
    denom = (pred_centered.square()).mean().clamp_min(1e-12)
    a = ((pred_centered * (truth - truth.mean())).mean() / denom).clamp(-5.0, 5.0)
    b = truth.mean() - a * pred.mean()
    return {
        "a": float(a.item()),
        "b": float(b.item()),
        "objective": "validation_mse_closed_form",
        "sample_count": int(pred.numel()),
    }


def _fit_affine_huber_calibrator(
    prediction: torch.Tensor,
    batch: MultimodalEpisodeBatch,
    *,
    steps: int = 120,
    lr: float = 0.05,
) -> dict[str, float]:
    pred, truth = _masked_flat_pair(prediction, batch.target_y, batch.target_mask)
    if pred.numel() < 2:
        return {"a": 1.0, "b": 0.0, "objective": "validation_huber_affine", "sample_count": int(pred.numel())}
    pred = pred.detach().to(dtype=torch.float64)
    truth = truth.detach().to(dtype=torch.float64)
    a = torch.nn.Parameter(torch.ones((), dtype=torch.float64, device=pred.device))
    b = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device=pred.device))
    optimizer = torch.optim.Adam([a, b], lr=float(lr))
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.smooth_l1_loss(a * pred + b, truth)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            a.clamp_(-5.0, 5.0)
            b.clamp_(-5.0, 5.0)
    return {
        "a": float(a.detach().cpu().item()),
        "b": float(b.detach().cpu().item()),
        "objective": "validation_huber_affine",
        "sample_count": int(pred.numel()),
    }


def _fit_affine_mosei_composite_calibrator(
    prediction: torch.Tensor,
    batch: MultimodalEpisodeBatch,
    *,
    weights: dict[str, float] | None,
) -> dict[str, float]:
    pred, truth = _masked_flat_pair(prediction, batch.target_y, batch.target_mask)
    if pred.numel() < 2:
        return {"a": 1.0, "b": 0.0, "objective": "validation_mosei_composite_grid", "sample_count": int(pred.numel())}
    composite_weights = weights or {
        "mae": 0.35,
        "pearson_correlation": 0.25,
        "acc7": 0.15,
        "acc5": 0.15,
        "acc2_excl0": 0.05,
        "acc2_nonneg": 0.05,
    }
    best = {"a": 1.0, "b": 0.0, "score": float("inf")}
    for a in torch.linspace(0.80, 1.20, 41, dtype=prediction.dtype):
        for b in torch.linspace(-0.20, 0.20, 41, dtype=prediction.dtype):
            calibrated = float(a.item()) * prediction + float(b.item())
            metrics = mosei_standard_metrics(calibrated, batch.target_y, batch.target_mask)
            score = _mosei_composite_score(metrics, composite_weights)
            if score < best["score"]:
                best = {"a": float(a.item()), "b": float(b.item()), "score": float(score)}
    return {
        "a": best["a"],
        "b": best["b"],
        "objective": "validation_mosei_composite_grid",
        "validation_composite_score": best["score"],
        "sample_count": int(pred.numel()),
    }


def _apply_task_calibration(prediction: torch.Tensor, calibrator: dict[str, Any]) -> torch.Tensor:
    if calibrator.get("method") == "none":
        return prediction
    return _apply_affine_calibration_to_prediction(prediction, _selected_affine_calibrator(calibrator))


def _apply_task_calibration_to_output(
    output: MultimodalOVHAOutput,
    calibrator: dict[str, Any],
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
) -> MultimodalOVHAOutput:
    if calibrator.get("method") == "none":
        diagnostics = {
            **output.diagnostics,
            "calibration": dict(calibrator),
            "pre_calibration_public_metrics": _public_main_metrics(
                config,
                batch,
                prediction=output.y_hat,
                router_load_by_candidate={},
                router_entropy=None,
                candidate_loss={},
                diagnostics=None,
            ),
            "post_calibration_public_metrics": _public_main_metrics(
                config,
                batch,
                prediction=output.y_hat,
                router_load_by_candidate={},
                router_entropy=None,
                candidate_loss={},
                diagnostics=None,
            ),
        }
        return replace(output, diagnostics=diagnostics)
    return _apply_affine_calibration(output, calibrator, config, batch)


def _task_calibration_diagnostics(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    *,
    pre_prediction: torch.Tensor,
    post_prediction: torch.Tensor,
    calibrator: dict[str, Any],
) -> dict[str, Any]:
    if calibrator.get("method") == "none":
        return {
            "calibration": dict(calibrator),
            "pre_calibration_public_metrics": _public_main_metrics(
                config,
                batch,
                prediction=pre_prediction,
                router_load_by_candidate={},
                router_entropy=None,
                candidate_loss={},
                diagnostics=None,
            ),
            "post_calibration_public_metrics": _public_main_metrics(
                config,
                batch,
                prediction=post_prediction,
                router_load_by_candidate={},
                router_entropy=None,
                candidate_loss={},
                diagnostics=None,
            ),
        }
    return _affine_calibration_diagnostics(
        config,
        batch,
        pre_prediction=pre_prediction,
        post_prediction=post_prediction,
        calibrator=calibrator,
    )


def _apply_affine_calibration(
    output: MultimodalOVHAOutput,
    calibrator: dict[str, Any],
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
) -> MultimodalOVHAOutput:
    pre_prediction = output.y_hat
    selected_calibrator = _selected_affine_calibrator(calibrator)
    scale = float(selected_calibrator.get("a", 1.0))
    offset = float(selected_calibrator.get("b", 0.0))
    calibrated_output = _transform_ovha_output_target_space(
        output,
        prediction_transform=lambda value: scale * value + offset,
        delta_transform=lambda value: scale * value,
        target_space={
            "prediction": "raw_calibrated",
            "candidate_values": "raw_calibrated",
            "candidate_outputs": "raw_calibrated_prediction_or_delta",
            "residual_delta": "raw_calibrated_delta",
        },
    )
    post_prediction = calibrated_output.y_hat
    candidate_names = tuple(calibrated_output.candidate_outputs)
    candidate_loss = _candidate_losses_from_values(calibrated_output.candidate_values, candidate_names, batch)
    diagnostics = {
        **calibrated_output.diagnostics,
        "pre_calibration_candidate_loss": output.diagnostics.get("candidate_loss", {}),
        "candidate_loss": candidate_loss,
        "raw_calibrated_candidate_loss": candidate_loss,
        "raw_calibrated_candidate_value_stats": _candidate_value_stats_from_values(calibrated_output.candidate_values, candidate_names),
        **_affine_calibration_diagnostics(
            config,
            batch,
            pre_prediction=pre_prediction,
            post_prediction=post_prediction,
            calibrator=calibrator,
        ),
    }
    if "candidate_loss_by_sample" in output.diagnostics:
        diagnostics["pre_calibration_candidate_loss_by_sample"] = output.diagnostics["candidate_loss_by_sample"]
        diagnostics["candidate_loss_by_sample"] = _candidate_losses_by_sample_from_values(
            calibrated_output.candidate_values,
            candidate_names,
            batch,
        )
    return replace(calibrated_output, diagnostics=diagnostics)


def _apply_affine_calibration_to_prediction(
    prediction: torch.Tensor,
    calibrator: dict[str, Any],
) -> torch.Tensor:
    return float(calibrator.get("a", 1.0)) * prediction + float(calibrator.get("b", 0.0))


def _selected_affine_calibrator(calibrator: dict[str, Any]) -> dict[str, Any]:
    if calibrator.get("method") != "affine_bank":
        return calibrator
    calibrators = calibrator.get("calibrators", {})
    selected = str(calibrator.get("selected", "mse_affine_calibration"))
    if isinstance(calibrators, dict) and selected in calibrators and isinstance(calibrators[selected], dict):
        return calibrators[selected]
    return {"a": 1.0, "b": 0.0, "objective": "missing_selected_affine_fallback"}


def _affine_calibration_diagnostics(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    *,
    pre_prediction: torch.Tensor,
    post_prediction: torch.Tensor,
    calibrator: dict[str, Any],
) -> dict[str, Any]:
    selected = _selected_affine_calibrator(calibrator)
    diagnostics: dict[str, Any] = {"affine_calibration": dict(selected)}
    if calibrator.get("method") == "affine_bank":
        diagnostics["calibration"] = dict(calibrator)
        diagnostics["mse_affine_calibration"] = dict(calibrator["calibrators"]["mse_affine_calibration"])
        diagnostics["huber_affine_calibration"] = dict(calibrator["calibrators"]["huber_affine_calibration"])
        diagnostics["composite_affine_calibration"] = dict(calibrator["calibrators"]["composite_affine_calibration"])
    try:
        diagnostics["pre_calibration_public_metrics"] = _public_main_metrics(
            config,
            batch,
            prediction=pre_prediction,
            router_load_by_candidate={},
            router_entropy=None,
            candidate_loss={},
            diagnostics=None,
        )
        diagnostics["post_calibration_public_metrics"] = _public_main_metrics(
            config,
            batch,
            prediction=post_prediction,
            router_load_by_candidate={},
            router_entropy=None,
            candidate_loss={},
            diagnostics=None,
        )
    except RuntimeError as exc:
        diagnostics["calibration_metric_status"] = f"skipped_shape_mismatch: {exc}"
    return diagnostics


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
    model_name: str = "ovha_full",
) -> dict[str, Any]:
    loss = _task_loss(output.y_hat, batch)
    return _raw_metric_row(
        config,
        batch,
        model_name=model_name,
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
        diagnostics=output.diagnostics,
        model_protocol="operator_valued_hyper_attention_public_main",
    )


def _per_sample_prediction_rows(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    output: MultimodalOVHAOutput | None,
    *,
    model_name: str,
    seed: int,
    prediction: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    prediction_tensor = output.y_hat if output is not None else prediction
    if prediction_tensor is None:
        return []
    prediction_tensor = prediction_tensor.detach().cpu()
    target = batch.target_y.detach().cpu()
    mask = batch.target_mask.detach().cpu()
    rows: list[dict[str, Any]] = []
    composition = output.diagnostics.get("composition", {}) if output is not None and isinstance(output.diagnostics, dict) else {}
    output_target_space = output.diagnostics.get("target_space", {}) if output is not None and isinstance(output.diagnostics, dict) else {}
    base_candidate = str(composition.get("base_candidate", ""))
    base_prediction = None
    prediction_by_candidate: dict[str, torch.Tensor] = {}
    if output is not None:
        prediction_by_candidate = {
            name: candidate_output.value.detach().cpu()
            for name, candidate_output in output.candidate_outputs.items()
        }
        if base_candidate in output.candidate_outputs:
            base_prediction = output.candidate_outputs[base_candidate].value.detach().cpu()
    raw_delta = _sample_tensor_map(composition.get("raw_delta_by_candidate", {}))
    gated_delta = _sample_tensor_map(composition.get("gated_delta_by_candidate", {}))
    residual_gate = _sample_tensor_map(composition.get("residual_gate_tensor_by_candidate", {}))
    for index, sample_id in enumerate(batch.provenance.source_id):
        row = {
            "artifact_type": "public_main_per_sample_prediction",
            "evidence_scope": "public_main_per_sample_diagnostics",
            "dataset": config.dataset_name,
            "task": config.task_type,
            "split": batch.split,
            "seed": int(seed),
            "model": model_name,
            "sample_index": index,
            "sample_id": str(sample_id),
            "target_mask": _tensor_payload(mask[index]),
            "truth": _tensor_payload(target[index]),
            "prediction_full": _tensor_payload(prediction_tensor[index]),
            "prediction_by_model": {model_name: _tensor_payload(prediction_tensor[index])},
            "prediction_by_candidate": {
                name: _tensor_payload(value[index])
                for name, value in prediction_by_candidate.items()
                if value.shape[0] > index
            },
            "target_space": _per_sample_target_space(output_target_space, output is not None),
            "base_candidate": base_candidate or None,
            "base_prediction": _tensor_payload(base_prediction[index]) if base_prediction is not None and base_prediction.shape[0] > index else None,
            "raw_delta_by_candidate": _indexed_tensor_map(raw_delta, index),
            "gated_delta_by_candidate": _indexed_tensor_map(gated_delta, index),
            "residual_gate_by_candidate": _indexed_tensor_map(residual_gate, index),
        }
        rows.append(row)
    return rows


def _per_sample_target_space(target_space: Any, has_ovha_output: bool) -> dict[str, str]:
    if isinstance(target_space, dict) and target_space:
        prediction_space = str(target_space.get("prediction", "model_output"))
        residual_delta_space = str(target_space.get("residual_delta", "model_output_delta"))
        candidate_space = str(target_space.get("candidate_outputs", target_space.get("candidate_values", prediction_space)))
    else:
        prediction_space = "model_output" if has_ovha_output else "raw_calibrated"
        residual_delta_space = "model_output_delta" if has_ovha_output else "not_applicable"
        candidate_space = prediction_space
    return {
        "prediction_full": prediction_space,
        "prediction_by_candidate": candidate_space,
        "base_prediction": prediction_space,
        "raw_delta_by_candidate": residual_delta_space,
        "gated_delta_by_candidate": residual_delta_space,
    }


def _sample_tensor_map(values: Any) -> dict[str, torch.Tensor]:
    if not isinstance(values, dict):
        return {}
    return {
        str(key): value.detach().cpu()
        for key, value in values.items()
        if hasattr(value, "detach")
    }


def _indexed_tensor_map(values: dict[str, torch.Tensor], index: int) -> dict[str, Any]:
    return {
        name: _tensor_payload(value[index])
        for name, value in values.items()
        if value.shape[0] > index
    }


def _tensor_payload(value: torch.Tensor) -> Any:
    payload = value.detach().cpu().tolist()
    return payload


def _baseline_rows(
    config: MultimodalExperimentConfig,
    *,
    train_batch: MultimodalEpisodeBatch,
    selection_batch: MultimodalEpisodeBatch,
    eval_batch: MultimodalEpisodeBatch,
    seed: int,
    baseline_train_steps: int,
    learning_rate: float,
    hardware: dict[str, Any],
    raw_metrics_path: Path,
    progress_interval: int,
    d_model: int,
    memory_tokens: int,
    batch_size: int,
    validation_fraction: float,
    eval_interval: int,
    early_stopping_patience: int,
    weight_decay: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    per_sample_prediction_rows: list[dict[str, Any]] = []
    for baseline_name in config.baseline_names:
        baseline_protocol = baseline_protocol_for_name(config.task_type, str(baseline_name))
        if str(baseline_name) in ovha_ablation_names_for_task(config.task_type):
            model, eval_output, summary = _train_ovha_ablation(
                config,
                str(baseline_name),
                train_batch=train_batch,
                eval_batch=eval_batch,
                selection_batch=selection_batch,
                seed=seed,
                train_steps=baseline_train_steps,
                learning_rate=learning_rate,
                progress_interval=progress_interval,
                d_model=d_model,
                memory_tokens=memory_tokens,
                batch_size=batch_size,
                validation_fraction=validation_fraction,
                eval_interval=eval_interval,
                early_stopping_patience=early_stopping_patience,
                weight_decay=weight_decay,
                device=device,
            )
            loss = _task_loss(eval_output.y_hat, eval_batch)
            rows.append(
                _raw_metric_row(
                    config,
                    eval_batch,
                    model_name=str(baseline_name),
                    seed=seed,
                    prediction=eval_output.y_hat,
                    score=_as_float(loss),
                    training_steps=baseline_train_steps,
                    parameter_count=_parameter_count(model),
                    raw_metrics_path=raw_metrics_path,
                    hardware=hardware,
                    router_load_by_candidate=eval_output.diagnostics.get("router_load_by_candidate", {}),
                    router_entropy=eval_output.diagnostics.get("router_entropy"),
                    candidate_loss=eval_output.diagnostics.get("candidate_loss", {}),
                    diagnostics=eval_output.diagnostics,
                    model_protocol=f"{baseline_protocol}_public_main_v1",
                )
            )
            robustness_rows.extend(
                _ovha_robustness_rows(
                    config,
                    model,
                    eval_batch,
                    seed=seed,
                    raw_metric_path=raw_metrics_path,
                    model_name=str(baseline_name),
                    device=device,
                    batch_size=batch_size,
                )
            )
            per_sample_prediction_rows.extend(
                _per_sample_prediction_rows(
                    config,
                    eval_batch,
                    eval_output,
                    model_name=str(baseline_name),
                    seed=seed,
                )
            )
            summaries.append({**summary, "model": str(baseline_name)})
            continue

        model, summary = _train_linear_baseline(
            config,
            str(baseline_name),
            train_batch=train_batch,
            selection_batch=selection_batch,
            seed=seed,
            train_steps=baseline_train_steps,
            learning_rate=learning_rate,
            progress_interval=progress_interval,
            batch_size=batch_size,
            validation_fraction=validation_fraction,
            eval_interval=eval_interval,
            early_stopping_patience=early_stopping_patience,
            weight_decay=weight_decay,
            device=device,
        )
        target_mean = summary.pop("_target_mean")
        target_std = summary.pop("_target_std")
        with torch.no_grad():
            eval_batch_std = _standardize_batch_targets(eval_batch, target_mean, target_std)
            selection_batch_std = _standardize_batch_targets(selection_batch, target_mean, target_std)
            selection_batch_device = _move_batch_to_device(selection_batch_std, device)
            selection_prediction = _destandardize_prediction(
                _baseline_prediction(str(baseline_name), model, selection_batch_device),
                target_mean.to(device=device) if target_mean is not None else None,
                target_std.to(device=device) if target_std is not None else None,
            )
            calibrator = _fit_task_calibrator(config, selection_prediction.detach().cpu(), selection_batch)
            eval_batch_device = _move_batch_to_device(eval_batch_std, device)
            pre_calibration_prediction = _destandardize_prediction(
                _baseline_prediction(str(baseline_name), model, eval_batch_device),
                target_mean.to(device=device) if target_mean is not None else None,
                target_std.to(device=device) if target_std is not None else None,
            )
            eval_prediction = _apply_task_calibration(pre_calibration_prediction, calibrator)
            loss = _task_loss(eval_prediction, _move_batch_to_device(eval_batch, device))
            eval_prediction = eval_prediction.detach().cpu()
            calibration_diagnostics = _task_calibration_diagnostics(
                config,
                eval_batch,
                pre_prediction=pre_calibration_prediction.detach().cpu(),
                post_prediction=eval_prediction,
                calibrator=calibrator,
            )
        router_load = _uniform_candidate_load(config.candidate_names)
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
                candidate_loss={candidate: loss for candidate in config.candidate_names},
                diagnostics=calibration_diagnostics,
                model_protocol=f"{baseline_protocol}_public_main_v1",
            )
        )
        per_sample_prediction_rows.extend(
            _per_sample_prediction_rows(
                config,
                eval_batch,
                None,
                model_name=str(baseline_name),
                seed=seed,
                prediction=eval_prediction,
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
                device=device,
                target_mean=target_mean,
                target_std=target_std,
            )
        )
        summaries.append({**summary, "model": str(baseline_name)})
    return rows, robustness_rows, summaries, per_sample_prediction_rows


def _train_ovha_ablation(
    config: MultimodalExperimentConfig,
    baseline_name: str,
    *,
    train_batch: MultimodalEpisodeBatch,
    eval_batch: MultimodalEpisodeBatch,
    selection_batch: MultimodalEpisodeBatch | None = None,
    seed: int,
    train_steps: int,
    learning_rate: float,
    progress_interval: int,
    d_model: int,
    memory_tokens: int,
    batch_size: int,
    validation_fraction: float,
    eval_interval: int,
    early_stopping_patience: int,
    weight_decay: float,
    device: torch.device | None = None,
) -> tuple[MultimodalOVHA, MultimodalOVHAOutput, dict[str, Any]]:
    device = device or train_batch.target_y.device
    selection_batch = selection_batch or train_batch
    target_mean, target_std = _target_standardizer(train_batch)
    fit_batch_std = _standardize_batch_targets(train_batch, target_mean, target_std)
    val_batch_std = _standardize_batch_targets(selection_batch, target_mean, target_std)
    eval_batch_std = _standardize_batch_targets(eval_batch, target_mean, target_std)
    field_dims = {name: int(field.x.shape[-1]) for name, field in train_batch.fields.items()}
    torch.manual_seed(int(seed) + _stable_baseline_seed_offset(baseline_name))
    variant_kwargs = _ovha_variant_kwargs(baseline_name, config.candidate_pool_names)
    active_candidate_names = variant_kwargs.pop("candidate_names", config.candidate_names)
    model_kwargs = {
        "use_evidence_router": config.use_evidence_router,
        "use_reliability_prior": config.use_reliability_prior,
        **_ovha_composition_kwargs(config, active_candidate_names),
        **variant_kwargs,
    }
    model = MultimodalOVHA(
        field_dims=field_dims,
        query_dim=int(train_batch.query.x.shape[-1]),
        output_dim=int(train_batch.target_y.shape[-1]),
        d_model=d_model,
        memory_tokens=memory_tokens,
        candidate_names=active_candidate_names,
        lrio_pairs=config.lrio_pairs or None,
        **model_kwargs,
    ).to(device)
    initial = _parameter_vector(model)
    fit_summary = _fit_public_ovha_model(
        config,
        model,
        fit_batch_std=fit_batch_std,
        val_batch_std=val_batch_std,
        seed=seed,
        sampling_seed=int(seed) + _stable_baseline_seed_offset(baseline_name) + 17,
        model_name=baseline_name,
        train_steps=train_steps,
        learning_rate=learning_rate,
        raw_val_batch=selection_batch,
        target_mean=target_mean,
        target_std=target_std,
        progress_interval=progress_interval,
        batch_size=batch_size,
        eval_interval=eval_interval,
        early_stopping_patience=early_stopping_patience,
        weight_decay=weight_decay,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        inference_batch_size = _eval_batch_size(int(batch_size))
        val_output = _destandardize_output(
            _predict_ovha_on_device(model, val_batch_std, device, batch_size=inference_batch_size),
            target_mean,
            target_std,
        )
        calibrator = _fit_task_calibrator(config, val_output.y_hat.detach().cpu(), selection_batch)
        eval_output = _destandardize_output(
            _predict_ovha_on_device(model, eval_batch_std, device, batch_size=inference_batch_size),
            target_mean,
            target_std,
        )
        eval_output = _with_raw_space_candidate_diagnostics(eval_output, eval_batch_std, eval_batch)
        eval_output = _apply_task_calibration_to_output(eval_output, calibrator, config, eval_batch)
    return model, eval_output, {
        "baseline_optimizer_steps": int(fit_summary["optimizer_steps"]),
        "baseline_parameter_l2_delta": float(torch.linalg.vector_norm(_parameter_vector(model) - initial).item()),
        "baseline_grad_l2_norm": float(fit_summary["max_grad_norm"]),
        "baseline_train_loss_final": float(fit_summary["final_loss"]),
        "baseline_best_val_task_loss": float(fit_summary["best_val_loss"]),
        "baseline_best_val_selection_score": float(fit_summary["best_val_score"]),
        "baseline_best_checkpoint_step": int(fit_summary["best_step"]),
        "baseline_training_protocol": str(fit_summary["training_protocol"]),
        "baseline_checkpoint_selection_protocol": "official_val_selection_best_checkpoint",
        "baseline_checkpoint_selection_metric": str(fit_summary["selection_metric"]),
        "baseline_stage_history": fit_summary["stage_history"],
        "baseline_lr_schedule": str(fit_summary["lr_schedule"]),
        "baseline_warmup_steps": int(fit_summary["warmup_steps"]),
        "baseline_min_lr_ratio": float(fit_summary["min_lr_ratio"]),
        "baseline_batch_size": batch_size,
        "baseline_validation_fraction": validation_fraction,
        "baseline_selection_split": selection_batch.split,
        "baseline_target_standardized": target_mean is not None and target_std is not None,
        "baseline_variant": {"candidate_names": active_candidate_names, **variant_kwargs},
    }


def _ovha_variant_kwargs(
    baseline_name: str,
    active_candidate_names: tuple[str, ...] = ("TLEO", "SPO", "LRIO", "CATO"),
) -> dict[str, Any]:
    if baseline_name == "ovha_no_rceo":
        return {"use_reliability_prior": False}
    if baseline_name == "ovha_no_evidence_router":
        return {"use_evidence_router": False}
    if baseline_name == "ovha_with_evidence_router":
        return {"use_evidence_router": True}
    if baseline_name == "cato_only":
        return {"candidate_names": ("CATO",)}
    if baseline_name == "spo_only":
        return {"candidate_names": _require_candidates(active_candidate_names, ("SPO",))}
    if baseline_name == "lrio_only":
        return {"candidate_names": _require_candidates(active_candidate_names, ("LRIO",))}
    if baseline_name == "ovha_tanso_only":
        if "TANSOBase" in active_candidate_names:
            return {"candidate_names": ("TANSOBase",), "composition_mode": "tanso_base"}
        return {"candidate_names": _require_candidates(active_candidate_names, ("TANSO",))}
    if baseline_name == "ovha_tanso_base_only":
        return {"candidate_names": _require_candidates(active_candidate_names, ("TANSOBase",)), "composition_mode": "tanso_base"}
    if baseline_name == "ovha_tanso_shift_only":
        return {"candidate_names": _require_candidates(active_candidate_names, ("TANSOShift",))}
    if baseline_name == "ovha_no_tanso":
        return {"candidate_names": _require_candidates(active_candidate_names, ("SPO",))}
    if baseline_name == "ovha_spo_lrio":
        return {
            "candidate_names": _require_candidates(active_candidate_names, ("SPO", "LRIO")),
            "composition_mode": "base_plus_residual",
            "base_candidate": "SPO",
            "residual_candidates": ("LRIO",),
        }
    if baseline_name == "ovha_spo_tanso":
        return {
            "candidate_names": _require_candidates(active_candidate_names, ("SPO", "TANSO")),
            "composition_mode": "base_plus_residual",
            "base_candidate": "SPO",
            "residual_candidates": ("TANSO",),
        }
    if baseline_name == "ovha_lrio_tanso":
        tanso_name = "TANSOBase" if "TANSOBase" in active_candidate_names else "TANSO"
        return {
            "candidate_names": _require_candidates(active_candidate_names, ("LRIO", tanso_name)),
            "composition_mode": "base_plus_residual",
            "base_candidate": tanso_name,
            "residual_candidates": ("LRIO",),
        }
    if baseline_name == "ovha_all_candidates_exploratory":
        return {
            "candidate_names": _require_candidates(active_candidate_names, ("SPO", "LRIO", "TANSO")),
            "composition_mode": "base_plus_residual",
            "base_candidate": "SPO",
            "residual_candidates": ("LRIO", "TANSO"),
        }
    if baseline_name == "ovha_no_cato":
        return {"candidate_names": _drop_candidate(active_candidate_names, "CATO")}
    if baseline_name == "ovha_no_lrio":
        return {"candidate_names": _drop_candidate(active_candidate_names, "LRIO")}
    if baseline_name == "ovha_no_spo":
        return {"candidate_names": _drop_candidate(active_candidate_names, "SPO")}
    raise ValueError(f"unknown OVHA ablation baseline: {baseline_name}")


def _require_candidates(active_candidate_names: tuple[str, ...], required: tuple[str, ...]) -> tuple[str, ...]:
    missing = tuple(candidate for candidate in required if candidate not in active_candidate_names)
    if missing:
        raise ValueError(f"structural ablation requires active candidates {missing}; active={active_candidate_names}")
    return required


def _drop_candidate(active_candidate_names: tuple[str, ...], candidate: str) -> tuple[str, ...]:
    kept = tuple(name for name in active_candidate_names if name != candidate)
    if not kept:
        raise ValueError(f"structural ablation would remove all active candidates: {candidate}")
    return kept


def _train_linear_baseline(
    config: MultimodalExperimentConfig,
    baseline_name: str,
    *,
    train_batch: MultimodalEpisodeBatch,
    selection_batch: MultimodalEpisodeBatch,
    seed: int,
    train_steps: int,
    learning_rate: float,
    progress_interval: int,
    batch_size: int,
    validation_fraction: float,
    eval_interval: int,
    early_stopping_patience: int,
    weight_decay: float,
    device: torch.device,
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    target_mean, target_std = _target_standardizer(train_batch)
    fit_batch_std = _standardize_batch_targets(train_batch, target_mean, target_std)
    val_batch_std = _standardize_batch_targets(selection_batch, target_mean, target_std)
    input_probe = _move_batch_to_device(
        _slice_batch(fit_batch_std, torch.arange(0, 1, device=fit_batch_std.target_y.device)),
        device,
    )
    train_inputs = _same_feature_probe_inputs(baseline_name, input_probe)
    target_dim = int(train_batch.target_y.shape[-1])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + _stable_baseline_seed_offset(baseline_name))
    model = torch.nn.Linear(int(train_inputs.shape[-1]), target_dim).to(device)
    with torch.no_grad():
        model.weight.uniform_(-0.02, 0.02, generator=generator)
        model.bias.uniform_(-0.02, 0.02, generator=generator)
    initial = _linear_parameter_vector(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    max_grad_norm = 0.0
    final_loss = 0.0
    best_val_loss = float("inf")
    best_step = 0
    best_state = _clone_state_dict(model)
    stale_evals = 0
    model.train()
    started_at = time.perf_counter()
    _print_progress(
        "model:start",
        seed=seed,
        model=baseline_name,
        steps=train_steps,
        learning_rate=learning_rate,
    )
    batch_generator = torch.Generator(device=fit_batch_std.target_y.device)
    batch_generator.manual_seed(int(seed) + _stable_baseline_seed_offset(baseline_name) + 19)
    for step in range(1, train_steps + 1):
        current_lr = _lr_for_step(
            step,
            learning_rate,
            config.warmup_steps,
            train_steps,
            config.min_lr_ratio,
        ) if config.lr_schedule == "warmup_cosine" else learning_rate
        _set_optimizer_lr(optimizer, current_lr)
        train_step_batch = _move_batch_to_device(_sample_batch(fit_batch_std, batch_size, batch_generator), device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _baseline_prediction(baseline_name, model, train_step_batch)
        loss = _task_loss(prediction, train_step_batch)
        loss.backward()
        max_grad_norm = max(max_grad_norm, _linear_grad_l2_norm(model))
        optimizer.step()
        final_loss = _as_float(loss)
        if _should_validate(step, train_steps, eval_interval):
            val_loss = _evaluate_linear_task_loss_on_device(
                baseline_name,
                model,
                val_batch_std,
                device,
                batch_size=_eval_batch_size(batch_size),
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_state = _clone_state_dict(model)
                stale_evals = 0
            else:
                stale_evals += 1
            if early_stopping_patience > 0 and stale_evals >= early_stopping_patience:
                _print_progress(
                    "model:early_stop",
                    seed=seed,
                    model=baseline_name,
                    step=step,
                    best_step=best_step,
                    best_val_loss=f"{best_val_loss:.6g}",
                )
                break
        if _should_log_progress(step, train_steps, progress_interval):
            _print_step_progress(
                seed=seed,
                model=baseline_name,
                step=step,
                total_steps=train_steps,
                loss=final_loss,
                started_at=started_at,
            )
    _print_progress(
        "model:done",
        seed=seed,
        model=baseline_name,
        steps=train_steps,
        loss=f"{final_loss:.6g}",
        elapsed=f"{time.perf_counter() - started_at:.1f}s",
    )
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "baseline_optimizer_steps": train_steps,
        "baseline_parameter_l2_delta": float(torch.linalg.vector_norm(_linear_parameter_vector(model) - initial).item()),
        "baseline_grad_l2_norm": max_grad_norm,
        "baseline_train_loss_final": final_loss,
        "baseline_best_val_task_loss": best_val_loss,
        "baseline_best_checkpoint_step": best_step,
        "baseline_training_protocol": "mini_batch_validation_best_checkpoint",
        "baseline_checkpoint_selection_protocol": "official_val_selection_best_checkpoint",
        "baseline_batch_size": batch_size,
        "baseline_validation_fraction": validation_fraction,
        "baseline_selection_split": selection_batch.split,
        "baseline_target_standardized": target_mean is not None and target_std is not None,
        "baseline_weight_decay": weight_decay,
        "baseline_lr_schedule": config.lr_schedule,
        "baseline_warmup_steps": int(config.warmup_steps),
        "baseline_min_lr_ratio": float(config.min_lr_ratio),
        "_target_mean": target_mean,
        "_target_std": target_std,
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
    diagnostics: Any,
) -> dict[str, Any]:
    standard = mosei_standard_metrics(prediction, batch.target_y, batch.target_mask) if _is_sentiment_task(config.task_type) else {}
    public_metrics = _public_main_metrics(
        config,
        batch,
        prediction=prediction,
        router_load_by_candidate=router_load_by_candidate,
        router_entropy=router_entropy,
        candidate_loss=candidate_loss,
        diagnostics=diagnostics,
    )
    task_loss = float(score)
    if _is_region_task(config.task_type):
        metric_name = "acc_at_0_5"
        primary_score = float(public_metrics[metric_name])
        higher_is_better = True
    else:
        metric_name = "mse_loss" if _is_sentiment_task(config.task_type) else "heldout_task_loss"
        primary_score = task_loss
        higher_is_better = False
    row = {
        "artifact_type": "public_main_raw_metric",
        "evidence_scope": "public_main_table",
        "dataset": config.dataset_name,
        "task": config.task_type,
        "model": model_name,
        "stage": "T5_eval",
        "split": batch.split,
        "seed": seed,
        "metric_name": metric_name,
        "score": primary_score,
        "task_loss": task_loss,
        "mse_loss": standard.get("mse_loss", score),
        "l1_loss": standard.get("l1_loss"),
        "higher_is_better": higher_is_better,
        "parameter_count": parameter_count,
        "training_steps": training_steps,
        "frozen_feature_extractor_version": dict(batch.provenance.feature_extractor_version),
        "hardware": dict(hardware),
        "label_provenance": _label_provenance_for_batch(batch),
        "seed_count_rationale": "configured five-seed public main evaluation",
        "model_protocol": model_protocol,
        "same_feature_source": True,
        "public_metrics": public_metrics,
        "public_metrics_scope": "public_main_metrics",
        "raw_metric_path": str(raw_metrics_path),
    }
    if isinstance(diagnostics, dict):
        for key in (
            "calibration",
            "affine_calibration",
            "mse_affine_calibration",
            "huber_affine_calibration",
            "composite_affine_calibration",
            "pre_calibration_public_metrics",
            "post_calibration_public_metrics",
        ):
            if key in diagnostics:
                row[key] = _json_ready(diagnostics[key])
    return row


def _public_main_metrics(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    *,
    prediction: torch.Tensor,
    router_load_by_candidate: Any,
    router_entropy: Any,
    candidate_loss: Any,
    diagnostics: Any = None,
) -> dict[str, Any]:
    if _is_region_task(config.task_type):
        task_loss = _task_loss(prediction, batch)
        region_metrics = _region_text_metrics(prediction, batch)
        cato_load = _candidate_probability(router_load_by_candidate, "CATO", default=0.0)
        cato_loss = _candidate_loss_value(candidate_loss, "CATO", default=task_loss)
        metrics = {
            "metrics_source": region_metrics["metrics_source"],
            "acc_at_0_5": region_metrics["acc_at_0_5"],
            "region_recall_at_1": region_metrics["region_recall_at_1"],
            "region_recall_at_5": region_metrics["region_recall_at_5"],
            "candidate_iou_at_0_5": region_metrics["candidate_iou_at_0_5"],
            "mean_candidate_iou": region_metrics["mean_candidate_iou"],
            "recall_at_1": region_metrics["recall_at_1"],
            "recall_at_5": region_metrics["recall_at_5"],
            "mean_iou": region_metrics["mean_iou"],
            "phrase_region_topk_accuracy": region_metrics["phrase_region_topk_accuracy"],
            "alignment_entropy": max(0.0, _as_float(router_entropy) if router_entropy is not None else 0.0),
            "cato_router_load": cato_load,
            "cato_candidate_loss": max(0.0, cato_loss),
            "cato_top_alignment_accuracy": region_metrics["phrase_region_topk_accuracy"],
            "null_unmatched_rate": _null_unmatched_rate(batch),
        }
        required = {name: metrics[name] for name in REGION_TEXT_REQUIRED_PUBLIC_METRICS}
        return {
            **required,
            "metrics_source": metrics["metrics_source"],
            "region_recall_at_1": metrics["region_recall_at_1"],
            "region_recall_at_5": metrics["region_recall_at_5"],
            "candidate_iou_at_0_5": metrics["candidate_iou_at_0_5"],
            "mean_candidate_iou": metrics["mean_candidate_iou"],
        }
    if _is_sentiment_task(config.task_type):
        standard = mosei_standard_metrics(prediction, batch.target_y, batch.target_mask)
        missing_drop = _missing_modality_fraction(batch)
        metrics = {
            "mae": max(0.0, standard["mae"]),
            "mse_loss": max(0.0, standard["mse_loss"]),
            "l1_loss": max(0.0, standard["l1_loss"]),
            "pearson_correlation": standard["pearson_correlation"],
            "acc7": standard["acc7"],
            "acc5": standard["acc5"],
            "acc2_excl0": standard["acc2_excl0"],
            "f1_excl0": standard["f1_excl0"],
            "acc2_nonneg": standard["acc2_nonneg"],
            "f1_nonneg": standard["f1_nonneg"],
            "accuracy": standard["acc2_excl0"],
            "f1": standard["f1_excl0"],
            "missing_modality_performance_drop": missing_drop,
            "corruption_robustness_auc": max(0.0, min(1.0, 1.0 - missing_drop)),
            "router_load_by_corruption_type": {
                "clean": _complete_candidate_probability_map(router_load_by_candidate, candidate_names=config.candidate_names)
            },
            "lrio_rank_entropy": _candidate_diagnostic_metric(diagnostics, "LRIO", "rank_entropy"),
            "spo_prototype_entropy": _candidate_diagnostic_metric(diagnostics, "SPO", "prototype_entropy"),
            "rceo_reliability_calibration": _rceo_calibration(batch, prediction, diagnostics),
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
    model_name: str = "ovha_full",
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    device: torch.device | None = None,
    batch_size: int = 128,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = device or next(model.parameters()).device
    model.eval()
    for corruption_type in ("clean", *DEFAULT_REQUIRED_STRESS_TARGETS):
        corrupted = _corrupted_batch(batch, corruption_type)
        model_batch = corrupted
        if target_mean is not None and target_std is not None:
            model_batch = _standardize_batch_targets(
                corrupted,
                target_mean,
                target_std,
            )
        with torch.no_grad():
            output = _predict_ovha_on_device(
                model,
                model_batch,
                device,
                batch_size=_eval_batch_size(int(batch_size)),
            )
            if target_mean is not None and target_std is not None:
                output = _destandardize_output(output, target_mean, target_std)
                output = _with_raw_space_candidate_diagnostics(output, model_batch, corrupted)
            loss = _task_loss(output.y_hat, corrupted)
            score = _robustness_score(config, corrupted, output.y_hat, loss)
        rows.append(
            _robustness_row(
                config,
                corrupted,
                model_name=model_name,
                seed=seed,
                raw_metric_path=raw_metric_path,
                corruption_type=corruption_type,
                score=score,
                task_loss=_as_float(loss),
                router_load_by_candidate=output.diagnostics.get("router_load_by_candidate", {}),
                candidate_loss=output.diagnostics.get("candidate_loss", {}),
                model_reliability=_model_reliability_mean(output.diagnostics),
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
    device: torch.device,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corruption_type in ("clean", *DEFAULT_REQUIRED_STRESS_TARGETS):
        corrupted = _corrupted_batch(batch, corruption_type)
        corrupted_device = _move_batch_to_device(corrupted, device)
        model_batch = corrupted_device
        if target_mean is not None and target_std is not None:
            model_batch = _standardize_batch_targets(
                corrupted_device,
                target_mean.to(device=device),
                target_std.to(device=device),
            )
        with torch.no_grad():
            prediction = _baseline_prediction(baseline_name, model, model_batch)
            if target_mean is not None and target_std is not None:
                prediction = _destandardize_prediction(
                    prediction,
                    target_mean.to(device=device),
                    target_std.to(device=device),
                )
            loss = _task_loss(prediction, corrupted_device)
            score = _robustness_score(config, corrupted_device, prediction, loss)
        rows.append(
            _robustness_row(
                config,
                corrupted,
                model_name=baseline_name,
                seed=seed,
                raw_metric_path=raw_metric_path,
                corruption_type=corruption_type,
                score=score,
                task_loss=_as_float(loss),
                router_load_by_candidate=_uniform_candidate_load(config.candidate_names),
                candidate_loss={candidate: loss for candidate in config.candidate_names},
                model_reliability=None,
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
    task_loss: float,
    router_load_by_candidate: Any,
    candidate_loss: Any,
    model_reliability: float | None,
) -> dict[str, Any]:
    strength = _corruption_strength(corruption_type)
    rceo_reliability = 0.0 if model_reliability is None else max(0.0, min(1.0, float(model_reliability)))
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
        "task_loss": task_loss,
        "rceo_reliability": rceo_reliability,
        "rceo_reliability_source": "model_reliability_prior" if model_reliability is not None else "not_applicable_no_model_reliability",
        "rceo_observed_reliability": score,
        "router_load_by_candidate": _complete_candidate_probability_map(router_load_by_candidate, candidate_names=config.candidate_names),
        "candidate_loss": _json_ready(candidate_loss),
        "source_raw_metric_path": str(raw_metric_path),
    }
    if corruption_type.startswith("hard_negative_"):
        row["mismatch_source_id"] = f"{batch.provenance.source_id[0]}::mismatch"
    return row


def _uniform_candidate_load(candidate_names: tuple[str, ...]) -> dict[str, float]:
    if not candidate_names:
        return {}
    weight = 1.0 / float(len(candidate_names))
    return {candidate: weight for candidate in candidate_names}


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


def _robustness_score(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    prediction: torch.Tensor,
    loss: torch.Tensor,
) -> float:
    if _is_region_task(config.task_type):
        return float(_region_text_metrics(prediction, batch)["acc_at_0_5"])
    return _bounded_score_from_loss(loss)


def _region_text_metrics(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> dict[str, float]:
    metrics = grounding_candidate_metrics(
        prediction,
        batch.supervision.region_targets,
        batch.supervision.candidate_region_boxes,
        batch.supervision.bbox_targets,
        batch.target_mask,
    )
    return {
        "metrics_source": METRICS_SOURCE,
        "acc_at_0_5": float(metrics["acc_at_0_5"]),
        "region_recall_at_1": float(metrics["recall_at_1"]),
        "region_recall_at_5": float(metrics["recall_at_5"]),
        "candidate_iou_at_0_5": float(metrics["acc_at_0_5"]),
        "mean_candidate_iou": float(metrics["mean_iou"]),
        "recall_at_1": float(metrics["recall_at_1"]),
        "recall_at_5": float(metrics["recall_at_5"]),
        "mean_iou": float(metrics["mean_iou"]),
        "phrase_region_topk_accuracy": float(metrics["recall_at_1"]),
        "cross_entropy": float(metrics["cross_entropy"]),
        "mrr": float(metrics["mrr"]),
    }


def _candidate_box_iou_metrics(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> dict[str, float]:
    candidate_boxes = batch.supervision.candidate_region_boxes
    target = batch.supervision.bbox_targets
    if candidate_boxes is None or target is None or prediction.shape[-1] <= 1:
        mean_iou = _bbox_mean_iou(prediction, batch)
        return {"candidate_iou_at_0_5": float(mean_iou >= 0.5) if mean_iou > 0.0 else 0.0, "mean_candidate_iou": mean_iou}
    boxes = candidate_boxes.to(device=prediction.device, dtype=torch.float32)
    truth = target.to(device=prediction.device, dtype=torch.float32)
    if truth.ndim == 2:
        truth = truth.unsqueeze(1)
    if truth.shape[1] == 1 and prediction.shape[1] > 1:
        truth = truth.expand(-1, prediction.shape[1], -1)
    query_count = min(int(prediction.shape[1]), int(truth.shape[1]))
    valid = batch.target_mask.to(dtype=torch.bool, device=prediction.device)[:, :query_count]
    if not bool(valid.any()):
        return {"candidate_iou_at_0_5": 0.0, "mean_candidate_iou": 0.0}
    top1 = prediction[:, :query_count, :].argmax(dim=-1)
    gather_index = top1.unsqueeze(-1).expand(-1, -1, 4)
    pred_boxes = torch.gather(boxes, dim=1, index=gather_index)
    iou = _box_iou(pred_boxes[valid], truth[:, :query_count, :][valid])
    return {
        "candidate_iou_at_0_5": _as_float((iou >= 0.5).to(dtype=torch.float32).mean()),
        "mean_candidate_iou": _as_float(iou.mean()),
    }


def _bbox_mean_iou(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> float:
    target = batch.supervision.bbox_targets
    if target is None or prediction.shape[-1] != 4:
        return 0.0
    pred = prediction.to(dtype=torch.float32)
    truth = target.to(device=prediction.device, dtype=torch.float32)
    if truth.ndim == 2:
        truth = truth.unsqueeze(1)
    if truth.shape[1] == 1 and pred.shape[1] > 1:
        truth = truth.expand(-1, pred.shape[1], -1)
    truth = truth[:, : pred.shape[1], :]
    valid = batch.target_mask.to(dtype=torch.bool, device=prediction.device)[:, : truth.shape[1]]
    if not bool(valid.any()):
        return 0.0
    return _as_float(_box_iou(pred[:, : truth.shape[1], :][valid], truth[valid]).mean())


def _box_iou(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    pred_min = torch.minimum(pred[..., :2], pred[..., 2:])
    pred_max = torch.maximum(pred[..., :2], pred[..., 2:])
    truth_min = torch.minimum(truth[..., :2], truth[..., 2:])
    truth_max = torch.maximum(truth[..., :2], truth[..., 2:])
    inter_min = torch.maximum(pred_min, truth_min)
    inter_max = torch.minimum(pred_max, truth_max)
    inter = (inter_max - inter_min).clamp_min(0.0)
    inter_area = inter[..., 0] * inter[..., 1]
    pred_area = ((pred_max - pred_min).clamp_min(0.0)).prod(dim=-1)
    truth_area = ((truth_max - truth_min).clamp_min(0.0)).prod(dim=-1)
    return inter_area / (pred_area + truth_area - inter_area).clamp_min(1e-12)


def _pearson_correlation(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred, truth = _masked_flat_pair(prediction, target, mask)
    if pred.numel() < 2:
        return 0.0
    pred_centered = pred - pred.mean()
    truth_centered = truth - truth.mean()
    denom = pred_centered.norm() * truth_centered.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, _as_float((pred_centered * truth_centered).sum() / denom)))


def _binary_sign_accuracy_f1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    pred, truth = _masked_flat_pair(prediction, target, mask)
    if pred.numel() == 0:
        return 0.0, 0.0
    pred_positive = pred >= 0
    truth_positive = truth >= 0
    accuracy = (pred_positive == truth_positive).to(dtype=torch.float32).mean()
    true_positive = (pred_positive & truth_positive).to(dtype=torch.float32).sum()
    false_positive = (pred_positive & ~truth_positive).to(dtype=torch.float32).sum()
    false_negative = (~pred_positive & truth_positive).to(dtype=torch.float32).sum()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return _as_float(accuracy), _as_float(f1)


def _masked_flat_pair(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.to(dtype=torch.bool, device=prediction.device)
    pred = prediction[..., 0][valid].reshape(-1)
    truth = target[..., 0].to(device=prediction.device, dtype=prediction.dtype)[valid].reshape(-1)
    return pred, truth


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


def _candidate_diagnostic_metric(diagnostics: Any, candidate: str, key: str) -> float:
    if not isinstance(diagnostics, dict):
        return 0.0
    candidate_diagnostics = diagnostics.get("candidate_diagnostics")
    if not isinstance(candidate_diagnostics, dict):
        return 0.0
    values = candidate_diagnostics.get(candidate)
    if not isinstance(values, dict):
        return 0.0
    return max(0.0, _as_float(values.get(key)))


def _rceo_calibration(
    batch: MultimodalEpisodeBatch,
    prediction: torch.Tensor,
    diagnostics: Any,
) -> dict[str, Any]:
    predicted_values, source = _model_reliability_by_sample(diagnostics, batch, prediction)
    observed_values = _bounded_observed_reliability_by_sample(batch, prediction)
    ece, curve = _binned_ece(predicted_values, observed_values, bin_count=10)
    return {
        "ece": ece,
        "expected_calibration_error": ece,
        "bin_count": 10,
        "source": source,
        "mean_predicted_reliability": _as_float(predicted_values.mean()),
        "mean_observed_reliability": _as_float(observed_values.mean()),
        "calibration_curve": curve,
        "condition": "public main reliability calibration between model RCEO reliability and bounded per-sample performance",
    }


def _model_reliability_mean(diagnostics: Any) -> float | None:
    if not isinstance(diagnostics, dict):
        return None
    reliability = diagnostics.get("reliability")
    if not isinstance(reliability, dict):
        return None
    if "modality_reliability_mean" in reliability:
        return _as_float(reliability["modality_reliability_mean"])
    candidate_diagnostics = diagnostics.get("candidate_diagnostics")
    if isinstance(candidate_diagnostics, dict):
        rceo = candidate_diagnostics.get("RCEO")
        if isinstance(rceo, dict) and "modality_reliability" in rceo:
            return _as_float(rceo["modality_reliability"])
    return None


def _bounded_observed_reliability(batch: MultimodalEpisodeBatch, prediction: torch.Tensor) -> float:
    return max(0.0, min(1.0, _as_float(_bounded_observed_reliability_by_sample(batch, prediction).mean())))


def _bounded_observed_reliability_by_sample(batch: MultimodalEpisodeBatch, prediction: torch.Tensor) -> torch.Tensor:
    error = (prediction - batch.target_y.to(device=prediction.device, dtype=prediction.dtype)).square()
    mask = batch.target_mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
    per_sample_error = (error * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
    return (1.0 / (1.0 + per_sample_error)).clamp(0.0, 1.0)


def _model_reliability_by_sample(
    diagnostics: Any,
    batch: MultimodalEpisodeBatch,
    prediction: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    batch_size = int(batch.target_y.shape[0])
    device = prediction.device
    dtype = prediction.dtype
    if isinstance(diagnostics, dict):
        reliability = diagnostics.get("reliability")
        if isinstance(reliability, dict):
            sample_values = reliability.get("sample_modality_reliability_mean")
            if sample_values is not None:
                values = torch.as_tensor(sample_values, dtype=dtype, device=device).reshape(-1)
                if values.numel() == batch_size:
                    return values.clamp(0.0, 1.0), "model_reliability_prior"
    predicted = _model_reliability_mean(diagnostics)
    if predicted is None:
        return torch.zeros(batch_size, dtype=dtype, device=device), "not_applicable_no_model_reliability"
    return torch.full((batch_size,), max(0.0, min(1.0, predicted)), dtype=dtype, device=device), "model_reliability_prior"


def _binned_ece(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    *,
    bin_count: int,
) -> tuple[float, list[dict[str, float | int]]]:
    predicted = predicted.detach().flatten().clamp(0.0, 1.0)
    observed = observed.detach().flatten().clamp(0.0, 1.0).to(device=predicted.device, dtype=predicted.dtype)
    total = max(int(predicted.numel()), 1)
    ece = 0.0
    curve = []
    for index in range(bin_count):
        lower = float(index) / float(bin_count)
        upper = float(index + 1) / float(bin_count)
        if index == bin_count - 1:
            mask = (predicted >= lower) & (predicted <= upper)
        else:
            mask = (predicted >= lower) & (predicted < upper)
        count = int(mask.sum().item())
        if count > 0:
            confidence = _as_float(predicted[mask].mean())
            accuracy = _as_float(observed[mask].mean())
        else:
            confidence = 0.0
            accuracy = 0.0
        ece += (count / total) * abs(confidence - accuracy)
        curve.append({"bin": index, "mean_confidence": confidence, "observed_accuracy": accuracy, "count": count})
    return ece, curve


def _validate_main_config_scope(path: Path, config: MultimodalExperimentConfig) -> None:
    text = " ".join((str(path), config.name, str(config.output_dir))).lower()
    if "smoke" in text or "not_topconf" in text:
        raise ValueError("run_public_main.py requires non-smoke public main config/output paths")
    if len(config.seeds) < 5:
        raise ValueError("public main training requires at least 5 configured seeds")


def _progress_interval(cli_value: int | None) -> int:
    if cli_value is not None:
        return max(0, int(cli_value))
    value = os.environ.get("OVHA_PUBLIC_MAIN_PROGRESS_INTERVAL")
    if value is None or value.strip() == "":
        return 500
    return max(0, int(value))


def _should_log_progress(current: int, total: int, interval: int) -> bool:
    if interval <= 0:
        return False
    return current == 1 or current == total or current % interval == 0


def _print_step_progress(
    *,
    seed: int,
    model: str,
    step: int,
    total_steps: int,
    loss: float,
    started_at: float,
) -> None:
    elapsed = time.perf_counter() - started_at
    steps_per_second = float(step) / max(elapsed, 1e-6)
    remaining = float(total_steps - step) / max(steps_per_second, 1e-6)
    percent = 100.0 * float(step) / max(float(total_steps), 1.0)
    _print_progress(
        "train",
        seed=seed,
        model=model,
        step=f"{step}/{total_steps}",
        percent=f"{percent:.1f}",
        loss=f"{loss:.6g}",
        elapsed=f"{elapsed:.1f}s",
        eta=f"{remaining:.1f}s",
        speed=f"{steps_per_second:.2f}step/s",
    )


def _print_progress(event: str, **fields: Any) -> None:
    parts = [f"[public-main:{event}]"]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print(" ".join(parts), file=sys.stderr, flush=True)


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
