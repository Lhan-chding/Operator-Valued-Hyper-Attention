from __future__ import annotations

from pathlib import Path

from moat_ovha_torch.data.multimodal.cache_schema import CacheValidationReport, MultimodalCacheLayout, validate_cache_layout


def validate_multimodal_cache(cache_root: Path | str, dataset_name: str, version: str, splits: tuple[str, ...]) -> CacheValidationReport:
    return validate_cache_layout(MultimodalCacheLayout(cache_root, dataset_name, version), splits=splits)
