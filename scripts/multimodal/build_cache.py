#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.data.multimodal.adapters.cmu_mosei import CMUMOSEIAdapter
from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import ControlledSyntheticMultimodalAdapter
from moat_ovha_torch.data.multimodal.adapters.refcoco import RefCOCOAdapter
from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, default_data_card


ADAPTERS = {
    "controlled_multimodal": ControlledSyntheticMultimodalAdapter,
    "refcoco": RefCOCOAdapter,
    "cmu_mosei": CMUMOSEIAdapter,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or initialize a multimodal OVHA cache.")
    parser.add_argument("dataset_name", choices=sorted(ADAPTERS))
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--version", default="v0.1")
    args = parser.parse_args()
    adapter = ADAPTERS[args.dataset_name]()
    manifest = adapter.discover_raw(args.raw_root)
    layout = MultimodalCacheLayout(args.cache_root, adapter.name, args.version)
    layout.root.mkdir(parents=True, exist_ok=True)
    card = default_data_card(adapter.name, args.version, _modalities_for(adapter.name), _tasks_for(adapter.name))
    (layout.root / "data_card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    (layout.root / "raw_manifest.json").write_text(
        json.dumps({"dataset_name": manifest.dataset_name, "raw_root": str(manifest.raw_root), "files": {k: str(v) for k, v in manifest.files.items()}}, indent=2)
        + "\n"
    )
    return 0


def _modalities_for(name: str) -> list[str]:
    if name == "cmu_mosei":
        return ["text", "audio", "vision"]
    if name == "refcoco":
        return ["text", "region"]
    return ["text", "region", "audio"]


def _tasks_for(name: str) -> list[str]:
    if name == "cmu_mosei":
        return ["sentiment_regression", "emotion_classification"]
    if name == "refcoco":
        return ["phrase_region_grounding"]
    return ["controlled_relation_operator"]


if __name__ == "__main__":
    raise SystemExit(main())
