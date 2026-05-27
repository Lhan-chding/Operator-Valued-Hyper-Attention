from __future__ import annotations

import json
import time
from pathlib import Path

from moat_ovha_torch.config import Phase15Config
from moat_ovha_torch.data.operator_zoo_torch import MetadataFreeOperatorZoo
from moat_ovha_torch.eval.diagnostics import primitive_load
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
    per_model_rows: dict[str, list[dict[str, object]]] = {}
    per_model_diagnostics: dict[str, list[dict[str, object]]] = {}

    for model_name in config.evaluation_model_names():
        model = build_model(model_name, d_model=config.d_model, memory_tokens=config.memory_tokens, top_k=config.top_k)
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
        for split in config.eval_splits:
            resolution_multiplier = 2 if split == "resolution_transfer" else 1
            for family in config.families:
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
                    )
                    output = model(batch)
                    rel = relative_l2(output.y_hat, batch.target_y)
                    row = {
                        "split": split,
                        "family": hidden.family,
                        "model": model_name,
                        "model_name": model_name,
                        "mse": float(((output.y_hat - batch.target_y) ** 2).mean().detach().cpu()),
                        "primitive_entropy": float(output.diagnostics["primitive_entropy"].detach().cpu()),
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
                    row.update(summarize_relative_l2(rel))
                    rows.append(row)
                    model_rows.append(row)
                    primitive_names = getattr(model, "primitive_names", None)
                    if primitive_names is None and isinstance(model, OVHAMetaOperator):
                        primitive_names = model.primitive_names
                    if primitive_names is None and hasattr(model, "core"):
                        primitive_names = model.core.primitive_names
                    if primitive_names:
                        diagnostics.append(
                            {
                                "split": split,
                                "family": hidden.family,
                                "model": model_name,
                                "model_name": model_name,
                                "seed": config.seed,
                                "eval_seed": eval_seed,
                                "checkpoint_loaded": checkpoint_loaded,
                                "checkpoint_path": str(ckpt) if checkpoint_loaded else None,
                                "primitive_load": primitive_load(output.primitive_weights, primitive_names),
                            }
                        )
                        model_diagnostics.append(diagnostics[-1])
        per_model_rows[model_name] = model_rows
        per_model_diagnostics[model_name] = model_diagnostics
        model_metrics_path = eval_metrics_path(output_dir, model_name, config.seed)
        model_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        model_metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in model_rows))
        model_diagnostics_path = diagnostics_path(output_dir, model_name, config.seed)
        model_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        model_diagnostics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in model_diagnostics))

    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    legacy_diagnostics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostics))
    return metrics_path
