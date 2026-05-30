#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import (
    CONTROLLED_MULTIMODAL_FAMILIES,
    ControlledSyntheticMultimodalAdapter,
)
from moat_ovha_torch.eval.multimodal_oracle import evaluate_oracle_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled multimodal oracle smoke without training.")
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-count", type=int, default=5)
    args = parser.parse_args()
    adapter = ControlledSyntheticMultimodalAdapter(output_dim=args.output_dim)
    summary = {}
    for family in CONTROLLED_MULTIMODAL_FAMILIES:
        batch = adapter.sample_batch(family=family, batch_size=args.batch_size, query_count=args.query_count)
        report = evaluate_oracle_matrix(batch)
        summary[family] = {"true_true_mse": float(report["true_true"]["mse"])}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
