#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
from moat_ovha_torch.models.multimodal.baselines import assert_same_feature_baseline_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and launch a multimodal public smoke run.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()

    config = MultimodalExperimentConfig.from_file(args.config)
    if args.cache_root is not None:
        config = _replace_cache_root(config, args.cache_root)
    assert_same_feature_baseline_policy(config)
    layout = MultimodalCacheLayout(config.cache_root, config.dataset_name, config.cache_version)
    report = validate_cache_layout(layout, splits=config.eval_splits)
    if not report.ok:
        print(
            json.dumps(
                {
                    "ok": False,
                    "policy": "fail-fast: no silent drop of missing cache artifacts or failed samples",
                    "config": config.name,
                    "errors": report.errors,
                    "warnings": report.warnings,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "policy": "cache validated; training launch intentionally requires explicit GPU runner",
                "config": config.name,
                "baselines": config.baseline_names,
                "seeds": config.seeds,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _replace_cache_root(config: MultimodalExperimentConfig, cache_root: Path) -> MultimodalExperimentConfig:
    return MultimodalExperimentConfig(
        name=config.name,
        dataset_name=config.dataset_name,
        task_type=config.task_type,
        cache_version=config.cache_version,
        cache_root=cache_root,
        output_dir=config.output_dir,
        seeds=config.seeds,
        training_stages=config.training_stages,
        candidate_names=config.candidate_names,
        baseline_names=config.baseline_names,
        eval_splits=config.eval_splits,
        eval_episode_count=config.eval_episode_count,
        enforce_same_features_for_baselines=config.enforce_same_features_for_baselines,
        fail_on_missing_cache_artifact=config.fail_on_missing_cache_artifact,
        allow_hidden_losses=config.allow_hidden_losses,
        require_public_alignment_labels=config.require_public_alignment_labels,
        robustness_corruptions=config.robustness_corruptions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
