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
from moat_ovha_torch.data.multimodal.cache_schema import file_sha256


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
    parser.add_argument("--artifact-root", type=Path)
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
                no_operator_memory_delta=0.10,
                no_hyper_adapter_delta=0.10,
                no_evidence_router_delta=0.10,
                no_reliability_prior_delta=0.10,
                memory_only_router_delta=0.10,
                evidence_only_router_delta=0.10,
                no_lrio_delta=0.10 if family in {"lrio_low_rank_interaction", "mixed_relation_operator"} else None,
                no_rceo_delta=0.10 if family in {"rceo_reliability_corruption", "mixed_relation_operator"} else None,
                diagnostics=_smoke_diagnostics(family),
            )
        )
    evidence_artifacts = _write_evidence_artifacts(args.artifact_root, rows) if args.artifact_root else None
    payload = {
        "mode": "oracle_smoke_only",
        "rows": rows,
        "controlled_report": build_controlled_report(rows, evidence_artifacts=evidence_artifacts),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _smoke_diagnostics(family: str) -> dict[str, object]:
    if family == "spo_global_prototype":
        return {"prototype_kl_delta": 0.10}
    if family == "lrio_low_rank_interaction":
        return {"rank_logits_kl_delta": 0.10}
    if family == "cato_alignment_transport":
        return {"alignment_entropy_delta": 0.10, "alignment_topk_delta": 0.10}
    if family == "rceo_reliability_corruption":
        return {
            "rank_logits_kl_delta": 0.10,
            "rceo_reliability_monotonic": True,
            "rceo_router_load_shift": 0.10,
            "rceo_reliability_curve": [
                {"corruption_strength": 0.0, "mean_reliability": 1.0},
                {"corruption_strength": 0.7, "mean_reliability": 0.3},
            ],
        }
    return {}


def _write_evidence_artifacts(artifact_root: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    controlled_rows = artifact_root / "controlled_multimodal_rows.jsonl"
    diagnostics_report = artifact_root / "controlled_multimodal_diagnostics.jsonl"
    _write_jsonl(controlled_rows, rows)
    _write_jsonl(diagnostics_report, rows)
    return {
        "task": "controlled_multimodal",
        "generated_by": "scripts/multimodal/run_controlled_smoke.py",
        "controlled_rows": {"path": str(controlled_rows), "sha256": file_sha256(controlled_rows)},
        "diagnostics_report": {"path": str(diagnostics_report), "sha256": file_sha256(diagnostics_report)},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
