#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import ControlledSyntheticMultimodalAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a controlled multimodal OVHA episode.")
    parser.add_argument("--family", default="mixed_relation_operator")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-count", type=int, default=4)
    args = parser.parse_args()
    batch = ControlledSyntheticMultimodalAdapter().sample_batch(
        family=args.family,
        batch_size=args.batch_size,
        query_count=args.query_count,
    )
    payload = {
        "fields": {name: list(field.x.shape) for name, field in batch.fields.items()},
        "query": list(batch.query.x.shape),
        "target_y": list(batch.target_y.shape),
        "hidden_keys": sorted(batch.hidden or {}),
        "model_input_keys": sorted(batch.model_inputs()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
