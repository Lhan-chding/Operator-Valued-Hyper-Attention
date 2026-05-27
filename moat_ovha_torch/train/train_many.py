from __future__ import annotations

from pathlib import Path

from moat_ovha_torch.config import Phase15Config
from moat_ovha_torch.train.trainer import run_training_many


def run_training_for_config_seeds(config: Phase15Config) -> dict[int, dict[str, Path]]:
    results: dict[int, dict[str, Path]] = {}
    for seed in config.seed_sequence():
        seed_config = config.with_seed(seed)
        results[seed] = run_training_many(seed_config, list(seed_config.training_model_names()))
    return results
