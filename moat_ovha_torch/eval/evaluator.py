from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from moat_ovha_torch.config import Phase15Config
from moat_ovha_torch.data.episodes import MetaOperatorBatch, hash_context, hash_model_inputs, hash_target
from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
from moat_ovha_torch.eval.diagnostics import primitive_load
from moat_ovha_torch.eval.oracle_metrics import controlled_oracle_metrics
from moat_ovha_torch.models.baselines import build_model
from moat_ovha_torch.models.ovha import OVHAMetaOperator
from moat_ovha_torch.runtime import require_torch, write_environment
from moat_ovha_torch.train.checkpoints import (
    checkpoint_path,
    diagnostics_path,
    eval_metrics_path,
    load_checkpoint_for_eval,
    parameter_count,
)
from moat_ovha_torch.train.metrics import relative_l2, summarize_relative_l2


def run_evaluation(config: Phase15Config) -> Path:
    torch = require_torch()
    eval_seed = config.seed + config.eval_seed_offset
    torch.manual_seed(eval_seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir, config.device, config.config_hash())
    zoo = MetadataFreeOperatorZoo(seed=eval_seed)
    metrics_path = output_dir / "eval_metrics.jsonl"
    legacy_diagnostics_path = output_dir / "diagnostics.jsonl"
    rows = []
    diagnostics = []
    start = time.time()
    progress_interval = _progress_interval("OVHA_EVAL_PROGRESS_INTERVAL", 128)

    for model_name in config.evaluation_model_names():
        model = build_model(
            model_name,
            d_model=config.d_model,
            memory_tokens=config.memory_tokens,
            top_k=config.top_k,
            controlled_generator_variant=config.controlled_generator_variant,
        )
        ckpt = checkpoint_path(output_dir, model_name, config.seed)
        checkpoint_payload = None
        checkpoint_loaded = False
        if ckpt.exists():
            checkpoint_payload = load_checkpoint_for_eval(model, ckpt, strict=True)
            if checkpoint_payload.get("model_name") not in {None, model_name}:
                raise ValueError(f"checkpoint model_name mismatch for {ckpt}: {checkpoint_payload.get('model_name')}")
            checkpoint_loaded = True
        elif config.require_checkpoint:
            raise FileNotFoundError(f"required evaluation checkpoint not found for {model_name}: {ckpt}")

        model = model.to(config.device)
        model.eval()
        params = parameter_count(model)
        model_rows: list[dict[str, object]] = []
        model_diagnostics: list[dict[str, object]] = []
        primitive_names = _primitive_names(model)
        total_eval_rows = len(config.eval_splits) * len(config.families) * config.eval_episode_count
        model_eval_index = 0
        print(
            f"[eval:start] seed={config.seed} eval_seed={eval_seed} model={model_name} "
            f"rows={total_eval_rows} checkpoint_loaded={checkpoint_loaded} progress_interval={progress_interval}",
            flush=True,
        )

        for split in config.eval_splits:
            resolution_multiplier = 2 if split == "resolution_transfer" else 1
            for family in config.families:
                for episode_offset in range(config.eval_episode_count):
                    episode_id = config.eval_episode_base + episode_offset
                    with torch.no_grad():
                        batch, hidden = zoo.sample_batch(
                            batch_size=config.batch_size,
                            num_demos=config.num_demos,
                            context_points=config.context_points,
                            support_points=config.support_points,
                            query_points=config.query_points,
                            family=family,
                            split=split,
                            mode=config.mode,
                            device=config.device,
                            resolution_multiplier=resolution_multiplier,
                            episode_id=episode_id,
                            controlled_generator_variant=config.controlled_generator_variant,
                        )
                        output = model(batch)
                        rel = relative_l2(output.y_hat, batch.target_y)
                        oracle_metrics = controlled_oracle_metrics(output, batch.target_y, hidden, primitive_names, batch=batch)
                        memory_swap_delta = _memory_swap_delta(model, batch, output)
                        row = {
                            "episode_id": episode_id,
                            "split": split,
                            "family": hidden.family,
                            "model": model_name,
                            "model_name": model_name,
                            "mse": float(((output.y_hat - batch.target_y) ** 2).mean().detach().cpu()),
                            "primitive_entropy": float(output.diagnostics["primitive_entropy"].detach().cpu()),
                            "memory_swap_delta": memory_swap_delta,
                            "batch_hash": hash_model_inputs(batch),
                            "context_hash": hash_context(batch),
                            "target_hash": hash_target(batch),
                            "wall_time_seconds": round(time.time() - start, 3),
                            "device": config.device,
                            "seed": config.seed,
                            "eval_seed": eval_seed,
                            "config_hash": config.config_hash(),
                            "parameter_count": params,
                            "oracle_upper_bound": model_name == "oracle_metadata_upper_bound",
                            "checkpoint_loaded": checkpoint_loaded,
                            "checkpoint_path": str(ckpt) if checkpoint_loaded else None,
                            "checkpoint_train_steps": checkpoint_payload.get("train_steps") if checkpoint_payload else None,
                        }
                        row.update(oracle_metrics)
                        row.update(summarize_relative_l2(rel))
                        rows.append(row)
                        model_rows.append(row)
                        model_eval_index += 1
                        if _should_log_progress(model_eval_index, total_eval_rows, progress_interval):
                            _print_eval_progress(row, model_eval_index, total_eval_rows, start)
                        if primitive_names:
                            diagnostic = _diagnostic_row(
                                split=split,
                                family=hidden.family,
                                model_name=model_name,
                                seed=config.seed,
                                eval_seed=eval_seed,
                                checkpoint_loaded=checkpoint_loaded,
                                checkpoint_path=str(ckpt) if checkpoint_loaded else None,
                                output=output,
                                primitive_names=primitive_names,
                                episode_id=episode_id,
                                oracle_metrics=oracle_metrics,
                                memory_swap_delta=memory_swap_delta,
                            )
                            diagnostics.append(diagnostic)
                            model_diagnostics.append(diagnostic)

        model_metrics_path = eval_metrics_path(output_dir, model_name, config.seed)
        model_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        model_metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in model_rows))
        model_diagnostics_path = diagnostics_path(output_dir, model_name, config.seed)
        model_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        model_diagnostics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in model_diagnostics))
        print(f"[eval:done] seed={config.seed} model={model_name} metrics={model_metrics_path}", flush=True)

    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    legacy_diagnostics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostics))
    return metrics_path


def _progress_interval(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def _should_log_progress(current: int, total: int, interval: int) -> bool:
    return current == 1 or current == total or (interval > 0 and current % interval == 0)


def _print_eval_progress(row: dict[str, object], current: int, total: int, start: float) -> None:
    elapsed = time.time() - start
    rows_per_second = current / max(elapsed, 1e-6)
    remaining = (total - current) / max(rows_per_second, 1e-6)
    percent = 100.0 * current / max(total, 1)
    router_mae = row.get("router_true_weight_mae")
    router_text = "" if router_mae is None else f" router_mae={float(router_mae):.6f}"
    print(
        "[eval] "
        f"seed={row['seed']} model={row['model_name']} row={current}/{total} ({percent:.1f}%) "
        f"split={row['split']} family={row['family']} episode={row['episode_id']} "
        f"relL2={float(row['relative_l2']):.6f} swap={float(row['memory_swap_delta']):.6f}"
        f"{router_text} elapsed={elapsed:.1f}s eta={remaining:.1f}s speed={rows_per_second:.2f} row/s",
        flush=True,
    )


def _primitive_names(model: Any) -> tuple[str, ...]:
    primitive_names = getattr(model, "primitive_names", None)
    if primitive_names is None and isinstance(model, OVHAMetaOperator):
        primitive_names = model.primitive_names
    if primitive_names is None and hasattr(model, "core"):
        primitive_names = model.core.primitive_names
    return tuple(primitive_names or ())


def _memory_swap_delta(model: Any, batch: MetaOperatorBatch, output: Any) -> float:
    swapped_batch = _shuffled_context_batch(batch)
    swapped_output = model(swapped_batch)
    numerator = (output.y_hat - swapped_output.y_hat).abs().mean()
    denominator = output.y_hat.abs().mean().clamp_min(1e-8)
    return float((numerator / denominator).detach().cpu())


def _shuffled_context_batch(batch: MetaOperatorBatch) -> MetaOperatorBatch:
    torch = require_torch()
    if batch.context_u.shape[0] > 1:
        index = torch.roll(torch.arange(batch.context_u.shape[0], device=batch.context_u.device), shifts=1)
        return MetaOperatorBatch(
            context_u=batch.context_u.index_select(0, index),
            context_q=batch.context_q.index_select(0, index),
            context_y=batch.context_y.index_select(0, index),
            target_u=batch.target_u,
            target_q=batch.target_q,
            target_y=batch.target_y,
            support_grid=batch.support_grid,
            context_mask=batch.context_mask.index_select(0, index) if batch.context_mask is not None else None,
            target_mask=batch.target_mask,
        )
    return MetaOperatorBatch(
        context_u=batch.context_u.flip(1),
        context_q=batch.context_q.flip(1),
        context_y=batch.context_y.flip(1),
        target_u=batch.target_u,
        target_q=batch.target_q,
        target_y=batch.target_y,
        support_grid=batch.support_grid,
        context_mask=batch.context_mask.flip(1) if batch.context_mask is not None else None,
        target_mask=batch.target_mask,
    )


def _diagnostic_row(
    *,
    split: str,
    family: str,
    model_name: str,
    seed: int,
    eval_seed: int,
    checkpoint_loaded: bool,
    checkpoint_path: str | None,
    output: Any,
    primitive_names: tuple[str, ...],
    episode_id: int | None = None,
    oracle_metrics: dict[str, float | None] | None = None,
    memory_swap_delta: float | None = None,
) -> dict[str, object]:
    row = {
        "episode_id": episode_id,
        "split": split,
        "family": family,
        "model": model_name,
        "model_name": model_name,
        "seed": seed,
        "eval_seed": eval_seed,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": checkpoint_path,
        "primitive_load": primitive_load(output.primitive_weights, primitive_names),
        "primitive_entropy": _to_float(output.diagnostics["primitive_entropy"]),
        "memory_norm": _to_float(output.diagnostics["memory_norms"]),
        "router_context_prior_entropy": _to_float(output.diagnostics.get("router_context_prior_entropy", 0.0)),
        "router_query_residual_norm": _to_float(output.diagnostics.get("router_query_residual_norm", 0.0)),
        "memory_swap_delta": memory_swap_delta,
        "adapter_norms": {
            name: _to_float(value)
            for name, value in output.diagnostics.get("adapter_norms", {}).items()
        },
        "adapter_stats": {
            name: {stat_name: _to_float(stat_value) for stat_name, stat_value in stats.items()}
            for name, stats in output.diagnostics.get("adapter_stats", {}).items()
        },
    }
    row.update(oracle_metrics or {})
    return row


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return round(float(value), 6)
