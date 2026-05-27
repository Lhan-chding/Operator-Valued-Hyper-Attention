from __future__ import annotations

import argparse
from pathlib import Path

from moat_ovha_torch.config import Phase15Config
from moat_ovha_torch.runtime import torch_available, write_environment, write_torch_skip_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 1.5 metadata-free OVHA torch model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = Phase15Config.from_file(args.config).with_overrides(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        device=args.device,
    )
    if not torch_available():
        write_environment(config.output_dir, config.device, config.config_hash())
        print(write_torch_skip_report(config.output_dir, "train_torch_meta_operator.py"))
        return

    from moat_ovha_torch.train.train_many import run_training_for_config_seeds

    print(run_training_for_config_seeds(config))


if __name__ == "__main__":
    main()
