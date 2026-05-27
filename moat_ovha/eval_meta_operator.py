from __future__ import annotations

import argparse
from pathlib import Path

from moat_ovha.config import Phase1Config
from moat_ovha.train_meta_operator import _run_split


def run_evaluation(config: Phase1Config) -> Path:
    """Run the deterministic phase-1 test split with parameter holdout."""

    return _run_split(config=config, split="test", filename="eval_metrics.jsonl", parameter_holdout=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase-1 OVHA deterministic evaluation sweep.")
    parser.add_argument("--config", required=True, help="Path to a JSON phase-1 config file.")
    args = parser.parse_args()
    config = Phase1Config.from_file(args.config)
    print(run_evaluation(config))


if __name__ == "__main__":
    main()
