from __future__ import annotations

import json
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
        loss = prediction_loss(output.y_hat, batch.target_y, batch.target_mask)
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

    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    ckpt = save_training_checkpoint(model, config, model_name)
    _write_training_report(output_dir, metrics_path, rows)
    return ckpt


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
