#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_EVIDENCE_TYPES = {"external_reference", "external_reproduction"}
REQUIRED_FIELDS = (
    "name",
    "dataset",
    "task_type",
    "evidence_type",
    "source_url",
    "official_repo",
    "reproduction_status",
    "integration_role",
    "notes",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a separate external SOTA reference/reproduction runbook. "
            "This does not download weights and does not add external models to same-feature OVHA baselines."
        )
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("configs/multimodal_external_sota_references.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        payload, exit_code = build_external_sota_runbook(args.references, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "ok": False,
            "mode": "external_sota_runbook",
            "policy": "fail-fast: external SOTA references must be explicit and separately scoped",
            "errors": [str(exc)],
            "warnings": [],
        }
        exit_code = 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def build_external_sota_runbook(references_path: Path, output_dir: Path) -> tuple[dict[str, Any], int]:
    payload = json.loads(references_path.read_text())
    references = _validate_reference_payload(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    runbook_path = output_dir / "external_sota_runbook.json"
    markdown_path = output_dir / "external_sota_runbook.md"
    grouped = _group_references(references)
    runbook = {
        "ok": True,
        "mode": "external_sota_runbook",
        "policy": (
            "external SOTA entries are separate reference/reproduction evidence; "
            "they must not be mixed into same-feature public-main baseline coverage"
        ),
        "source_config": str(references_path),
        "reference_count": len(references),
        "datasets": sorted(grouped),
        "references_by_dataset": grouped,
        "manual_actions": _manual_actions(references),
        "refcoco_comparison_tracks": _refcoco_comparison_tracks(),
    }
    runbook_path.write_text(json.dumps(runbook, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown(runbook) + "\n")
    result = {
        **runbook,
        "generated_files": {
            "runbook": str(runbook_path),
            "markdown": str(markdown_path),
        },
        "next": [
            f"review {markdown_path}",
            "download/reproduce external checkpoints only after manual licensing and storage approval",
        ],
    }
    return result, 0


def _validate_reference_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    if payload.get("schema_version") != "external-sota-references-v0.1":
        raise ValueError("schema_version must be external-sota-references-v0.1")
    references = payload.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("references must be a non-empty list")
    validated: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(references):
        if not isinstance(row, dict):
            raise ValueError(f"references[{index}] must be an object")
        missing = [field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(f"references[{index}] missing required fields: {', '.join(missing)}")
        evidence_type = str(row["evidence_type"])
        if evidence_type not in ALLOWED_EVIDENCE_TYPES:
            raise ValueError(f"references[{index}] evidence_type must be one of {sorted(ALLOWED_EVIDENCE_TYPES)}")
        source_url = str(row["source_url"])
        official_repo = str(row["official_repo"])
        if not source_url.startswith(("https://", "http://")):
            raise ValueError(f"references[{index}] source_url must be an HTTP(S) URL")
        if not official_repo.startswith(("https://", "http://")):
            raise ValueError(f"references[{index}] official_repo must be an HTTP(S) URL")
        key = (str(row["dataset"]), str(row["name"]))
        if key in seen:
            raise ValueError(f"duplicate external SOTA reference: {key[0]}/{key[1]}")
        seen.add(key)
        validated.append({field: str(row[field]) for field in REQUIRED_FIELDS})
    return validated


def _group_references(references: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in references:
        grouped.setdefault(row["dataset"], []).append(row)
    return {dataset: sorted(rows, key=lambda row: row["name"]) for dataset, rows in sorted(grouped.items())}


def _manual_actions(references: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "name": row["name"],
            "dataset": row["dataset"],
            "action": "record published metric table or run official repo in an isolated environment",
            "do_not_do": "do not import this as a same-feature linear probe baseline",
            "source_url": row["source_url"],
            "official_repo": row["official_repo"],
        }
        for row in references
    ]


def _refcoco_comparison_tracks() -> list[dict[str, str]]:
    return [
        {
            "track": "A_external_open_box_reference",
            "input": "raw image plus referring expression",
            "output": "free-form predicted boxes",
            "comparison_scope": "external detector/reference table only",
            "required_metrics": "Acc@0.5, mean IoU, split-specific val/testA/testB where available",
        },
        {
            "track": "B_same_candidate_scorer",
            "input": "OVHA candidate boxes plus external detector predictions",
            "output": "candidate score_i=max_j IoU(candidate_i, box_j) * score_j",
            "comparison_scope": "same-candidate table",
            "required_metrics": "R@1, R@5, Acc@0.5, mean IoU, MRR",
        },
        {
            "track": "C_proposal_generator_plus_reranker",
            "input": "shared GroundingDINO top-K proposals for every reranker",
            "output": "proposal upper bound plus reranker metrics",
            "comparison_scope": "proposal-conditioned reranker table",
            "required_metrics": "proposal_recall@K first, then reranker R@1/R@5/Acc@0.5/mean IoU/MRR",
        },
    ]


def _markdown(runbook: dict[str, Any]) -> str:
    lines = [
        "# External SOTA Reference Runbook",
        "",
        runbook["policy"],
        "",
    ]
    for dataset, rows in runbook["references_by_dataset"].items():
        lines.extend([f"## {dataset}", "", "| Model | Evidence | Status | Source | Repo |", "|---|---|---|---|---|"])
        for row in rows:
            lines.append(
                "| {name} | {evidence_type} | {reproduction_status} | [paper/project]({source_url}) | [repo]({official_repo}) |".format(
                    **row
                )
            )
        lines.append("")
    lines.extend(
        [
            "## RefCOCO Comparison Tracks",
            "",
        ]
    )
    for row in runbook.get("refcoco_comparison_tracks", []):
        lines.extend(
            [
                f"### {row['track']}",
                "",
                f"- Input: {row['input']}",
                f"- Output: {row['output']}",
                f"- Scope: {row['comparison_scope']}",
                f"- Required metrics: {row['required_metrics']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Rules",
            "",
            "- These entries are external references or external reproductions.",
            "- They are not trained by `run_public_main.py`.",
            "- They must not be listed in `baseline_names` for same-feature public-main configs.",
            "- Any reproduced result must record checkpoint, commit, environment, split, metric, and data license.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
