from __future__ import annotations

import argparse
import json
from pathlib import Path

from moat_ovha.baselines import make_predictor
from moat_ovha.config import Phase1Config
from moat_ovha.data.operator_zoo import OperatorZoo
from moat_ovha.metrics import mse, relative_l2


def run_training(config: Phase1Config) -> Path:
    """Run the deterministic phase-1 train split sweep and write JSONL metrics."""

    return _run_split(config=config, split="train", filename="train_metrics.jsonl", parameter_holdout=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase-1 OVHA deterministic training sweep.")
    parser.add_argument("--config", required=True, help="Path to a JSON phase-1 config file.")
    args = parser.parse_args()
    config = Phase1Config.from_file(args.config)
    print(run_training(config))


def _run_split(config: Phase1Config, split: str, filename: str, parameter_holdout: bool) -> Path:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    zoo = OperatorZoo(seed=config.seed)
    model_names = ("ovha_full", "ovha_sparse", *config.baselines, *config.ablations)

    rows = []
    for task in zoo.make_suite(config.families, split, config.context_sizes, config.resolution, parameter_holdout):
        for model_name in model_names:
            predictor = make_predictor(model_name)
            prediction = predictor.predict(task)
            row = {
                "split": split,
                "family": task.family,
                "context_size": len(task.context),
                "model": model_name,
                "mse": mse(prediction.predictions, task.targets),
                "relative_l2": relative_l2(prediction.predictions, task.targets),
                "parameter_count": prediction.parameter_count,
                "expert_entropy": float(prediction.diagnostics.get("entropy", 0.0)),
                "primitive_names": prediction.diagnostics.get("primitive_names", []),
                "parameter_holdout": parameter_holdout,
            }
            rows.append(row)

    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


if __name__ == "__main__":
    main()
