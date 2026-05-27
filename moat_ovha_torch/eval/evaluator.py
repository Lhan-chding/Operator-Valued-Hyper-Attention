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
from moat_ovha_torch.train.metrics import relative_l2, summarize_relative_l2


def run_evaluation(config: Phase15Config) -> Path:
    torch = require_torch()
    torch.manual_seed(config.seed + 1000)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir, config.device, config.config_hash())
    zoo = MetadataFreeOperatorZoo(seed=config.seed + 1000)
    metrics_path = output_dir / "eval_metrics.jsonl"
    diagnostics_path = output_dir / "diagnostics.jsonl"
    rows = []
    diagnostics = []
    start = time.time()

    for split in config.eval_splits:
        resolution_multiplier = 2 if split == "resolution_transfer" else 1
        for family in config.families:
            for model_name in config.models:
                model = build_model(model_name, d_model=config.d_model, memory_tokens=config.memory_tokens, top_k=config.top_k).to(config.device)
                model.eval()
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
                        "mse": float(((output.y_hat - batch.target_y) ** 2).mean().detach().cpu()),
                        "primitive_entropy": float(output.diagnostics["primitive_entropy"].detach().cpu()),
                        "wall_time_seconds": round(time.time() - start, 3),
                        "device": config.device,
                        "seed": config.seed,
                        "config_hash": config.config_hash(),
                        "parameter_count": sum(param.numel() for param in model.parameters()),
                        "oracle_upper_bound": model_name == "oracle_metadata_upper_bound",
                    }
                    row.update(summarize_relative_l2(rel))
                    rows.append(row)
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
                                "primitive_load": primitive_load(output.primitive_weights, primitive_names),
                            }
                        )

    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    diagnostics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostics))
    return metrics_path
