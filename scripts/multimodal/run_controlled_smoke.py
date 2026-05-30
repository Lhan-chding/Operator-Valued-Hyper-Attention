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
from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report
from moat_ovha_torch.eval.multimodal_oracle import controlled_row_from_oracle_report, evaluate_oracle_matrix


ACTIVE_OPERATOR_BY_FAMILY = {
    "tleo_local_evidence": "TLEO",
    "spo_global_prototype": "SPO",
    "lrio_low_rank_interaction": "LRIO",
    "cato_alignment_transport": "CATO",
    "rceo_reliability_corruption": "LRIO",
    "mixed_relation_operator": "mixed",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled multimodal oracle smoke without training.")
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-count", type=int, default=5)
    args = parser.parse_args()
    adapter = ControlledSyntheticMultimodalAdapter(output_dim=args.output_dim)
    rows = []
    for family in CONTROLLED_MULTIMODAL_FAMILIES:
        batch = adapter.sample_batch(family=family, batch_size=args.batch_size, query_count=args.query_count)
        report = evaluate_oracle_matrix(batch)
        rows.append(
            controlled_row_from_oracle_report(
                family,
                ACTIVE_OPERATOR_BY_FAMILY[family],
                report,
                diagnostics=_smoke_diagnostics(family),
            )
        )
    payload = {
        "mode": "oracle_smoke_only",
        "rows": rows,
        "controlled_report": build_controlled_report(rows),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _smoke_diagnostics(family: str) -> dict[str, object]:
    if family == "spo_global_prototype":
        return {"prototype_kl_delta": 0.0}
    if family == "lrio_low_rank_interaction":
        return {"rank_logits_kl_delta": 0.0}
    if family == "cato_alignment_transport":
        return {"alignment_entropy_delta": 0.0, "alignment_topk_delta": 0.0}
    if family == "rceo_reliability_corruption":
        return {"rank_logits_kl_delta": 0.0, "rceo_reliability_monotonic": True, "rceo_router_load_shift": 0.0}
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
