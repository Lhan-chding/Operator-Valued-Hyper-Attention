#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize controlled multimodal oracle/gate JSONL rows.")
    parser.add_argument("controlled_jsonl", type=Path)
    args = parser.parse_args()
    rows = []
    with args.controlled_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    print(
        json.dumps(
            build_controlled_report(
                rows,
                evidence_artifacts=_evidence_artifacts(args.controlled_jsonl),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _evidence_artifacts(path: Path) -> dict[str, object]:
    descriptor = _artifact_descriptor(path)
    return {
        "task": "controlled_multimodal",
        "generated_by": str(Path(__file__).relative_to(ROOT)),
        "controlled_rows": descriptor,
        "diagnostics_report": descriptor,
    }


def _artifact_descriptor(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.exists() else "",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
