from __future__ import annotations

from dataclasses import replace

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch
from moat_ovha_torch.data.multimodal.transforms._metadata import batch_metadata_column


def record_temporal_shift_metadata(batch: MultimodalEpisodeBatch, temporal_shift_sec: float) -> MultimodalEpisodeBatch:
    metadata = dict(batch.supervision.corruption_metadata or {})
    metadata["temporal_shift_sec"] = batch_metadata_column(batch, float(temporal_shift_sec))
    supervision = replace(batch.supervision, corruption_metadata=metadata)
    return replace(batch, supervision=supervision)
