from __future__ import annotations

import json
import os
import time
from pathlib import Path

from moat_ovha_torch.config import Phase15Config
from moat_ovha_torch.data.episodes import hash_context, hash_model_inputs, hash_target
from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
from moat_ovha_torch.models.baselines import build_model
from moat_ovha_torch.runtime import require_torch, write_environment
from moat_ovha_torch.train.checkpoints import parameter_count, save_training_checkpoint, train_metrics_path
from moat_ovha_torch.train.losses import prediction_loss
from moat_ovha_torch.train.metrics import relative_l2, summarize_relative_l2


def run_training(config: Phase15Config) -> Path:
    run_training_many(config, list(config.training_model_names()))
    return _write_legacy_train_metrics(config.output_dir, config.training_model_names(), config.seed)


def run_training_many(config: Phase15Config, model_names: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for model_name in model_names:
        checkpoints[model_name] = run_training_for_model(config, model_name)
    _write_legacy_train_metrics(config.output_dir, tuple(model_names), config.seed)
    return checkpoints


def run_training_for_model(config: Phase15Config, model_name: str) -> Path:
    torch = require_torch()
    torch.manual_seed(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir, config.device, config.config_hash())
    zoo = MetadataFreeOperatorZoo(seed=config.seed)
    model = build_model(model_name, d_model=config.d_model, memory_tokens=config.memory_tokens, top_k=config.top_k).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    metrics_path = train_metrics_path(output_dir, model_name, config.seed)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()
    params = parameter_count(model)
    progress_interval = _progress_interval("OVHA_PROGRESS_INTERVAL", 25)
    print(
        f"[train:start] seed={config.seed} model={model_name} steps={config.steps} "
        f"families={','.join(config.families)} device={config.device} progress_interval={progress_interval}",
        flush=True,
    )

    for step in range(1, config.steps + 1):
        family = config.families[(step - 1) % len(config.families)]
        episode_id = config.train_episode_base + step - 1
        batch, hidden = zoo.sample_batch(
            batch_size=config.batch_size,
            num_demos=config.num_demos,
            context_points=config.context_points,
            support_points=config.support_points,
            query_points=config.query_points,
            family=family,
            split=config.train_split,
            mode=config.mode,
            device=config.device,
            episode_id=episode_id,
            controlled_generator_variant=config.controlled_generator_variant,
        )
        output = model(batch)
        pred_loss = prediction_loss(output.y_hat, batch.target_y, batch.target_mask)
        router_aux_loss = _router_auxiliary_loss(output, hidden, _primitive_names(model), config.device)
        if router_aux_loss is not None and config.router_auxiliary_loss_weight > 0.0:
            loss = pred_loss + config.router_auxiliary_loss_weight * router_aux_loss
        else:
            loss = pred_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rel = relative_l2(output.y_hat.detach(), batch.target_y)
        row = {
            "step": step,
            "episode_id": episode_id,
            "split": config.train_split,
            "family": hidden.family,
            "model": model_name,
            "loss": float(loss.detach().cpu()),
            "prediction_loss": float(pred_loss.detach().cpu()),
            "router_auxiliary_loss": None if router_aux_loss is None else float(router_aux_loss.detach().cpu()),
            "router_auxiliary_loss_weight": config.router_auxiliary_loss_weight,
            "batch_hash": hash_model_inputs(batch),
            "context_hash": hash_context(batch),
            "target_hash": hash_target(batch),
            "primitive_entropy": float(output.diagnostics["primitive_entropy"].detach().cpu()),
            "wall_time_seconds": round(time.time() - start, 3),
            "device": config.device,
            "seed": config.seed,
            "config_hash": config.config_hash(),
            "parameter_count": params,
            "train_steps": config.steps,
        }
        row.update(summarize_relative_l2(rel))
        rows.append(row)
        if _should_log_progress(step, config.steps, progress_interval):
            _print_train_progress(row, start)

    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    ckpt = save_training_checkpoint(model, config, model_name)
    _write_training_report(output_dir, metrics_path, rows)
    print(f"[train:done] seed={config.seed} model={model_name} checkpoint={ckpt}", flush=True)
    return ckpt


def _progress_interval(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def _should_log_progress(current: int, total: int, interval: int) -> bool:
    return current == 1 or current == total or (interval > 0 and current % interval == 0)


def _print_train_progress(row: dict[str, object], start: float) -> None:
    step = int(row["step"])
    total = int(row["train_steps"])
    elapsed = time.time() - start
    steps_per_second = step / max(elapsed, 1e-6)
    remaining = (total - step) / max(steps_per_second, 1e-6)
    percent = 100.0 * step / max(total, 1)
    print(
        "[train] "
        f"seed={row['seed']} model={row['model']} step={step}/{total} ({percent:.1f}%) "
        f"episode={row['episode_id']} family={row['family']} "
        f"loss={float(row['loss']):.6f} relL2={float(row['relative_l2']):.6f} "
        f"elapsed={elapsed:.1f}s eta={remaining:.1f}s speed={steps_per_second:.2f} step/s",
        flush=True,
    )


def _write_legacy_train_metrics(output_dir: Path, model_names: tuple[str, ...], seed: int) -> Path:
    legacy_path = output_dir / "train_metrics.jsonl"
    lines: list[str] = []
    for model_name in model_names:
        path = train_metrics_path(output_dir, model_name, seed)
        if path.exists():
            lines.extend(path.read_text().splitlines())
    legacy_path.write_text("\n".join(line for line in lines if line) + ("\n" if lines else ""))
    return legacy_path


def _write_training_report(output_dir: Path, metrics_path: Path, rows: list[dict[str, object]]) -> None:
    final = rows[-1] if rows else {}
    (output_dir / "phase1_5_report.md").write_text(
        "\n".join(
            [
                "# Phase 1.5 Report",
                "",
                "## CPU Smoke Training",
                "",
                f"- metrics: `{metrics_path.name}`",
                f"- final_relative_l2: {final.get('relative_l2')}",
                f"- final_loss: {final.get('loss')}",
                f"- steps: {len(rows)}",
                "",
                "Run `eval_torch_meta_operator.py` and `scripts/summarize_phase1_5.py` for the full evaluation report.",
            ]
        )
        + "\n"
    )


def _primitive_names(model: object) -> tuple[str, ...]:
    primitive_names = getattr(model, "primitive_names", None)
    if primitive_names is None and hasattr(model, "core"):
        primitive_names = getattr(model.core, "primitive_names", None)
    return tuple(primitive_names or ())


def _router_auxiliary_loss(output: object, hidden: object, primitive_names: tuple[str, ...], device: str):
    hints = getattr(hidden, "oracle_hints", None) or {}
    true_weights = hints.get("true_component_weight_by_q")
    true_order = tuple(hints.get("primitive_order") or ())
    if true_weights is None or not true_order or not primitive_names:
        return None
    reorder = []
    for name in primitive_names:
        if name not in true_order:
            return None
        reorder.append(true_order.index(name))
    predicted = output.primitive_weights
    if predicted.shape[-1] != len(reorder):
        return None
    torch = require_torch()
    target = true_weights.to(device) if hasattr(true_weights, "to") else torch.as_tensor(true_weights, device=device)
    target = target.index_select(-1, torch.tensor(reorder, device=target.device))
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(target * predicted.clamp_min(1e-8).log()).sum(dim=-1).mean()
