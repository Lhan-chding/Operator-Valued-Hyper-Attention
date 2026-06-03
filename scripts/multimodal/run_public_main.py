#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import fields as dataclass_fields, is_dataclass, replace
import json
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
from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
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
    progress_interval = _progress_interval(args.progress_interval)
    artifact_root = args.artifact_root or config.output_dir
    artifact_root.mkdir(parents=True, exist_ok=True)
    raw_metrics_path = artifact_root / "raw_metrics.jsonl"
    diagnostics_path = artifact_root / "diagnostics.jsonl"
    robustness_rows_path = artifact_root / "robustness_rows.jsonl"

    raw_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
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
        "seeds": list(selected_seeds),
        "seed_count": len(selected_seeds),
        "configured_seeds": list(config.seeds),
        "configured_seed_count": len(config.seeds),
        "pilot_seed_subset": pilot_seed_subset,
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
    host_device = torch.device("cpu")
    train_batch = _load_public_batch(layout, config, args.train_split, host_device)
    eval_batch = _load_public_batch(layout, config, args.eval_split, host_device)
    fit_batch, val_batch = _split_train_val_batch(
        train_batch,
        validation_fraction=float(args.validation_fraction),
        seed=seed,
    )
    target_mean, target_std = _target_standardizer(fit_batch)
    fit_batch_std = _standardize_batch_targets(fit_batch, target_mean, target_std)
    val_batch_std = _standardize_batch_targets(val_batch, target_mean, target_std)
    eval_batch_std = _standardize_batch_targets(eval_batch, target_mean, target_std)
    field_dims = {name: int(field.x.shape[-1]) for name, field in train_batch.fields.items()}
    model = MultimodalOVHA(
        field_dims=field_dims,
        query_dim=int(train_batch.query.x.shape[-1]),
        output_dim=int(train_batch.target_y.shape[-1]),
        d_model=int(args.d_model),
        memory_tokens=int(args.memory_tokens),
        candidate_names=config.candidate_names,
        lrio_pairs=config.lrio_pairs or None,
    ).to(device)
    initial_parameters = _parameter_vector(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    max_grad_norm = 0.0
    final_loss = 0.0
    best_val_loss = float("inf")
    best_step = 0
    best_state = _clone_state_dict(model)
    stale_evals = 0
    batch_generator = torch.Generator(device=fit_batch_std.target_y.device)
    batch_generator.manual_seed(int(seed) + 17)
    model.train()
    train_started_at = time.perf_counter()
    _print_progress(
        "model:start",
        seed=seed,
        model="ovha_full",
        steps=int(args.train_steps),
        learning_rate=float(args.learning_rate),
    )
    for step in range(1, int(args.train_steps) + 1):
        train_step_batch = _move_batch_to_device(
            _sample_batch(fit_batch_std, int(args.batch_size), batch_generator),
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(train_step_batch)
        components = _public_loss_components(output, train_step_batch, config)
        total_loss = torch.stack([value for value in components.values()]).sum()
        total_loss.backward()
        max_grad_norm = max(max_grad_norm, _grad_l2_norm(model))
        optimizer.step()
        final_loss = _as_float(total_loss)
        if _should_validate(step, int(args.train_steps), int(args.eval_interval)):
            val_loss = _evaluate_task_loss_on_device(
                model,
                val_batch_std,
                device,
                batch_size=_eval_batch_size(int(args.batch_size)),
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_state = _clone_state_dict(model)
                stale_evals = 0
            else:
                stale_evals += 1
            if int(args.early_stopping_patience) > 0 and stale_evals >= int(args.early_stopping_patience):
                _print_progress(
                    "model:early_stop",
                    seed=seed,
                    model="ovha_full",
                    step=step,
                    best_step=best_step,
                    best_val_loss=f"{best_val_loss:.6g}",
                )
                break
        if _should_log_progress(step, int(args.train_steps), progress_interval):
            _print_step_progress(
                seed=seed,
                model="ovha_full",
                step=step,
                total_steps=int(args.train_steps),
                loss=final_loss,
                started_at=train_started_at,
            )
    _print_progress(
        "model:done",
        seed=seed,
        model="ovha_full",
        steps=int(args.train_steps),
        loss=f"{final_loss:.6g}",
        elapsed=f"{time.perf_counter() - train_started_at:.1f}s",
    )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        eval_batch_std_device = _move_batch_to_device(eval_batch_std, device)
        eval_output = _destandardize_output(
            model(eval_batch_std_device),
            target_mean.to(device=device),
            target_std.to(device=device),
        )
        eval_output = _move_ovha_output_to_device(eval_output, torch.device("cpu"))
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
        target_mean=target_mean,
        target_std=target_std,
        device=device,
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
        "summary": {
            "seed": seed,
            "ovha_parameter_l2_delta": float(torch.linalg.vector_norm(_parameter_vector(model) - initial_parameters).item()),
            "ovha_max_grad_norm": max_grad_norm,
            "ovha_best_val_task_loss": best_val_loss,
            "ovha_best_checkpoint_step": best_step,
            "ovha_training_protocol": "mini_batch_validation_best_checkpoint",
            "ovha_batch_size": int(args.batch_size),
            "ovha_validation_fraction": float(args.validation_fraction),
            "ovha_target_standardized": True,
            "baseline_count": len(baseline_rows),
            "baseline_summaries": baseline_summaries,
        },
    }


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


def _eval_batch_size(train_batch_size: int) -> int:
    return max(1, int(train_batch_size) * 8)


def _should_validate(step: int, train_steps: int, eval_interval: int) -> bool:
    interval = max(1, int(eval_interval))
    return step == 1 or step == train_steps or step % interval == 0


def _target_standardizer(batch: MultimodalEpisodeBatch) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch.target_mask.to(dtype=batch.target_y.dtype, device=batch.target_y.device).unsqueeze(-1)
    denom = mask.sum(dim=(0, 1), keepdim=True).clamp_min(1.0)
    mean = (batch.target_y * mask).sum(dim=(0, 1), keepdim=True) / denom
    variance = ((batch.target_y - mean).square() * mask).sum(dim=(0, 1), keepdim=True) / denom
    std = variance.sqrt().clamp_min(1e-6)
    return mean.detach(), std.detach()


def _standardize_batch_targets(
    batch: MultimodalEpisodeBatch,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> MultimodalEpisodeBatch:
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
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> MultimodalOVHAOutput:
    return replace(
        output,
        y_hat=output.y_hat * target_std + target_mean,
        candidate_values=output.candidate_values * target_std.unsqueeze(-2) + target_mean.unsqueeze(-2),
    )


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
    progress_interval: int,
    d_model: int,
    memory_tokens: int,
    batch_size: int,
    validation_fraction: float,
    eval_interval: int,
    early_stopping_patience: int,
    weight_decay: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for baseline_name in config.baseline_names:
        baseline_protocol = baseline_protocol_for_name(config.task_type, str(baseline_name))
        if str(baseline_name) in ovha_ablation_names_for_task(config.task_type):
            model, eval_output, summary = _train_ovha_ablation(
                config,
                str(baseline_name),
                train_batch=train_batch,
                eval_batch=eval_batch,
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
                )
            )
            summaries.append({**summary, "model": str(baseline_name)})
            continue

        model, summary = _train_linear_baseline(
            str(baseline_name),
            train_batch=train_batch,
            seed=seed,
            train_steps=baseline_train_steps,
            learning_rate=learning_rate,
            progress_interval=progress_interval,
            batch_size=batch_size,
            device=device,
        )
        with torch.no_grad():
            eval_batch_device = _move_batch_to_device(eval_batch, device)
            eval_prediction = _baseline_prediction(str(baseline_name), model, eval_batch_device)
            loss = _task_loss(eval_prediction, eval_batch_device)
            eval_prediction = eval_prediction.detach().cpu()
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
                model_protocol=f"{baseline_protocol}_public_main_v1",
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
            )
        )
        summaries.append({**summary, "model": str(baseline_name)})
    return rows, robustness_rows, summaries


def _train_ovha_ablation(
    config: MultimodalExperimentConfig,
    baseline_name: str,
    *,
    train_batch: MultimodalEpisodeBatch,
    eval_batch: MultimodalEpisodeBatch,
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
    fit_batch, val_batch = _split_train_val_batch(
        train_batch,
        validation_fraction=validation_fraction,
        seed=int(seed) + _stable_baseline_seed_offset(baseline_name),
    )
    target_mean, target_std = _target_standardizer(fit_batch)
    fit_batch_std = _standardize_batch_targets(fit_batch, target_mean, target_std)
    val_batch_std = _standardize_batch_targets(val_batch, target_mean, target_std)
    eval_batch_std = _standardize_batch_targets(eval_batch, target_mean, target_std)
    field_dims = {name: int(field.x.shape[-1]) for name, field in train_batch.fields.items()}
    torch.manual_seed(int(seed) + _stable_baseline_seed_offset(baseline_name))
    variant_kwargs = _ovha_variant_kwargs(baseline_name, config.candidate_names)
    active_candidate_names = variant_kwargs.pop("candidate_names", config.candidate_names)
    model = MultimodalOVHA(
        field_dims=field_dims,
        query_dim=int(train_batch.query.x.shape[-1]),
        output_dim=int(train_batch.target_y.shape[-1]),
        d_model=d_model,
        memory_tokens=memory_tokens,
        candidate_names=active_candidate_names,
        lrio_pairs=config.lrio_pairs or None,
        **variant_kwargs,
    ).to(device)
    initial = _parameter_vector(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    max_grad_norm = 0.0
    final_loss = 0.0
    best_val_loss = float("inf")
    best_step = 0
    best_state = _clone_state_dict(model)
    stale_evals = 0
    batch_generator = torch.Generator(device=fit_batch_std.target_y.device)
    batch_generator.manual_seed(int(seed) + _stable_baseline_seed_offset(baseline_name) + 17)
    model.train()
    started_at = time.perf_counter()
    _print_progress(
        "model:start",
        seed=seed,
        model=baseline_name,
        steps=train_steps,
        learning_rate=learning_rate,
        protocol="mini_batch_validation_best_checkpoint",
    )
    for step in range(1, train_steps + 1):
        train_step_batch = _move_batch_to_device(_sample_batch(fit_batch_std, batch_size, batch_generator), device)
        optimizer.zero_grad(set_to_none=True)
        output = model(train_step_batch)
        components = _public_loss_components(output, train_step_batch, config)
        total_loss = torch.stack([value for value in components.values()]).sum()
        total_loss.backward()
        max_grad_norm = max(max_grad_norm, _grad_l2_norm(model))
        optimizer.step()
        final_loss = _as_float(total_loss)
        if _should_validate(step, train_steps, eval_interval):
            val_loss = _evaluate_task_loss_on_device(
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
    with torch.no_grad():
        eval_batch_std_device = _move_batch_to_device(eval_batch_std, device)
        eval_output = _destandardize_output(
            model(eval_batch_std_device),
            target_mean.to(device=device),
            target_std.to(device=device),
        )
        eval_output = _move_ovha_output_to_device(eval_output, torch.device("cpu"))
    return model, eval_output, {
        "baseline_optimizer_steps": train_steps,
        "baseline_parameter_l2_delta": float(torch.linalg.vector_norm(_parameter_vector(model) - initial).item()),
        "baseline_grad_l2_norm": max_grad_norm,
        "baseline_train_loss_final": final_loss,
        "baseline_best_val_task_loss": best_val_loss,
        "baseline_best_checkpoint_step": best_step,
        "baseline_training_protocol": "mini_batch_validation_best_checkpoint",
        "baseline_batch_size": batch_size,
        "baseline_validation_fraction": validation_fraction,
        "baseline_target_standardized": True,
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
    if baseline_name == "cato_only":
        return {"candidate_names": ("CATO",)}
    if baseline_name == "ovha_no_cato":
        return {"candidate_names": _drop_candidate(active_candidate_names, "CATO")}
    if baseline_name == "ovha_no_lrio":
        return {"candidate_names": _drop_candidate(active_candidate_names, "LRIO")}
    if baseline_name == "ovha_no_spo":
        return {"candidate_names": _drop_candidate(active_candidate_names, "SPO")}
    raise ValueError(f"unknown OVHA ablation baseline: {baseline_name}")


def _drop_candidate(active_candidate_names: tuple[str, ...], candidate: str) -> tuple[str, ...]:
    kept = tuple(name for name in active_candidate_names if name != candidate)
    if not kept:
        raise ValueError(f"structural ablation would remove all active candidates: {candidate}")
    return kept


def _train_linear_baseline(
    baseline_name: str,
    *,
    train_batch: MultimodalEpisodeBatch,
    seed: int,
    train_steps: int,
    learning_rate: float,
    progress_interval: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    input_probe = _move_batch_to_device(
        _slice_batch(train_batch, torch.arange(0, 1, device=train_batch.target_y.device)),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    max_grad_norm = 0.0
    final_loss = 0.0
    model.train()
    started_at = time.perf_counter()
    _print_progress(
        "model:start",
        seed=seed,
        model=baseline_name,
        steps=train_steps,
        learning_rate=learning_rate,
    )
    batch_generator = torch.Generator(device=train_batch.target_y.device)
    batch_generator.manual_seed(int(seed) + _stable_baseline_seed_offset(baseline_name) + 19)
    for step in range(1, train_steps + 1):
        train_step_batch = _move_batch_to_device(_sample_batch(train_batch, batch_size, batch_generator), device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _baseline_prediction(baseline_name, model, train_step_batch)
        loss = _task_loss(prediction, train_step_batch)
        loss.backward()
        max_grad_norm = max(max_grad_norm, _linear_grad_l2_norm(model))
        optimizer.step()
        final_loss = _as_float(loss)
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
    return model, {
        "baseline_optimizer_steps": train_steps,
        "baseline_parameter_l2_delta": float(torch.linalg.vector_norm(_linear_parameter_vector(model) - initial).item()),
        "baseline_grad_l2_norm": max_grad_norm,
        "baseline_train_loss_final": final_loss,
        "baseline_training_protocol": "mini_batch",
        "baseline_batch_size": batch_size,
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
        region_metrics = _region_text_metrics(prediction, batch)
        cato_load = _candidate_probability(router_load_by_candidate, "CATO", default=0.0)
        cato_loss = _candidate_loss_value(candidate_loss, "CATO", default=task_loss)
        metrics = {
            "acc_at_0_5": region_metrics["acc_at_0_5"],
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
        return {name: metrics[name] for name in REGION_TEXT_REQUIRED_PUBLIC_METRICS}
    if _is_sentiment_task(config.task_type):
        mae = _as_float((prediction - batch.target_y).abs().mean())
        pearson = _pearson_correlation(prediction, batch.target_y, batch.target_mask)
        accuracy, f1 = _binary_sign_accuracy_f1(prediction, batch.target_y, batch.target_mask)
        missing_drop = _missing_modality_fraction(batch)
        corruption_key = _missing_corruption_key(batch)
        metrics = {
            "mae": max(0.0, mae),
            "pearson_correlation": pearson,
            "accuracy": accuracy,
            "f1": f1,
            "missing_modality_performance_drop": missing_drop,
            "corruption_robustness_auc": max(0.0, min(1.0, 1.0 - missing_drop)),
            "router_load_by_corruption_type": {
                corruption_key: _complete_candidate_probability_map(router_load_by_candidate)
            },
            "lrio_rank_entropy": _entropy_proxy(router_load_by_candidate, "LRIO"),
            "spo_prototype_entropy": _entropy_proxy(router_load_by_candidate, "SPO"),
            "rceo_reliability_calibration": _rceo_calibration(batch, accuracy),
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = device or next(model.parameters()).device
    model.eval()
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
            output = model(model_batch)
            if target_mean is not None and target_std is not None:
                output = _destandardize_output(output, target_mean.to(device=device), target_std.to(device=device))
            loss = _task_loss(output.y_hat, corrupted_device)
        rows.append(
            _robustness_row(
                config,
                corrupted,
                model_name=model_name,
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
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corruption_type in ("clean", *DEFAULT_REQUIRED_STRESS_TARGETS):
        corrupted = _corrupted_batch(batch, corruption_type)
        corrupted_device = _move_batch_to_device(corrupted, device)
        with torch.no_grad():
            prediction = _baseline_prediction(baseline_name, model, corrupted_device)
            loss = _task_loss(prediction, corrupted_device)
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


def _region_text_metrics(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> dict[str, float]:
    region_targets = batch.supervision.region_targets
    if region_targets is not None and prediction.shape[-1] > 1:
        labels = region_targets.to(device=prediction.device, dtype=torch.long)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        if labels.ndim > 2:
            labels = labels.reshape(labels.shape[0], -1)
        if labels.shape[1] == 1 and prediction.shape[1] > 1:
            labels = labels.expand(-1, prediction.shape[1])
        labels = labels[:, : prediction.shape[1]]
        valid = batch.target_mask.to(dtype=torch.bool, device=prediction.device)[:, : labels.shape[1]]
        top1 = prediction[:, : labels.shape[1]].argmax(dim=-1)
        topk = torch.topk(prediction[:, : labels.shape[1]], k=min(5, prediction.shape[-1]), dim=-1).indices
        if bool(valid.any()):
            recall1 = (top1[valid] == labels[valid]).to(dtype=torch.float32).mean()
            recall5 = (topk[valid] == labels[valid].unsqueeze(-1)).any(dim=-1).to(dtype=torch.float32).mean()
            return {
                "acc_at_0_5": _as_float(recall1),
                "recall_at_1": _as_float(recall1),
                "recall_at_5": _as_float(recall5),
                "mean_iou": _bbox_mean_iou(prediction, batch),
                "phrase_region_topk_accuracy": _as_float(recall1),
            }
    return {
        "acc_at_0_5": 0.0,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "mean_iou": _bbox_mean_iou(prediction, batch),
        "phrase_region_topk_accuracy": 0.0,
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
