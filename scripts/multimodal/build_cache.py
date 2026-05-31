#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moat_ovha_torch.data.multimodal.adapters import (
    CMUMOSEIAdapter,
    CMUMOSIAdapter,
    ControlledSyntheticMultimodalAdapter,
    Flickr30kEntitiesAdapter,
    IEMOCAPAdapter,
    MELDAdapter,
    MissingMultimodalDataError,
    RefCOCOAdapter,
    VisualGenomeAdapter,
)
from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout


ADAPTERS = {
    "controlled_multimodal": ControlledSyntheticMultimodalAdapter,
    "refcoco": RefCOCOAdapter,
    "flickr30k_entities": Flickr30kEntitiesAdapter,
    "visual_genome": VisualGenomeAdapter,
    "cmu_mosei": CMUMOSEIAdapter,
    "cmu_mosi": CMUMOSIAdapter,
    "meld": MELDAdapter,
    "iemocap": IEMOCAPAdapter,
}
REGION_TEXT_DATASETS = {"refcoco", "flickr30k_entities", "visual_genome"}
SENTIMENT_DATASETS = {"cmu_mosei", "cmu_mosi", "meld", "iemocap"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or initialize a multimodal OVHA cache.")
    parser.add_argument("dataset_name", choices=sorted(ADAPTERS))
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--version", default="v0.1")
    args = parser.parse_args()
    adapter = ADAPTERS[args.dataset_name]()
    try:
        manifest = adapter.discover_raw(args.raw_root)
    except MissingMultimodalDataError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "policy": "fail-fast: raw data manifest must be complete before cache initialization",
                    "dataset_name": args.dataset_name,
                    "raw_root": str(args.raw_root),
                    "errors": [str(exc)],
                    "warnings": [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    layout = MultimodalCacheLayout(args.cache_root, adapter.name, args.version)
    try:
        for split in _cache_splits_for(adapter.name):
            adapter.write_cache(manifest, args.cache_root, split, args.version)
    except (NotImplementedError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "policy": "fail-fast: cache writer must create a complete validated cache; partial cache initialization is forbidden",
                    "dataset_name": args.dataset_name,
                    "raw_root": str(args.raw_root),
                    "cache_root": str(layout.root),
                    "errors": [str(exc)],
                    "warnings": [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    validation = adapter.validate_cache(args.cache_root, args.version)
    if not validation.ok:
        print(
            json.dumps(
                {
                    "ok": False,
                    "policy": "fail-fast: cache writer output must pass validate_cache_layout before use",
                    "dataset_name": args.dataset_name,
                    "raw_root": str(args.raw_root),
                    "cache_root": str(layout.root),
                    "errors": validation.errors,
                    "warnings": validation.warnings,
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
                "policy": "cache built and validated",
                "dataset_name": manifest.dataset_name,
                "raw_root": str(manifest.raw_root),
                "cache_root": str(layout.root),
                "warnings": validation.warnings,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _modalities_for(name: str) -> list[str]:
    if name in SENTIMENT_DATASETS:
        return ["text", "audio", "vision"]
    if name in REGION_TEXT_DATASETS:
        return ["text", "region"]
    return ["text", "region", "audio"]


def _tasks_for(name: str) -> list[str]:
    if name in SENTIMENT_DATASETS:
        return ["sentiment_regression", "emotion_classification"]
    if name in REGION_TEXT_DATASETS:
        return ["phrase_region_grounding"]
    return ["controlled_relation_operator"]


def _cache_splits_for(name: str) -> tuple[str, ...]:
    if name == "controlled_multimodal":
        return ("train",)
    return ("train", "val", "test")


if __name__ == "__main__":
    raise SystemExit(main())
