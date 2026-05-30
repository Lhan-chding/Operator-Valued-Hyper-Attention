from __future__ import annotations

from dataclasses import replace

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch


def record_temporal_shift_metadata(batch: MultimodalEpisodeBatch, temporal_shift_sec: float) -> MultimodalEpisodeBatch:
    metadata = dict(batch.supervision.corruption_metadata or {})
    metadata["temporal_shift_sec"] = temporal_shift_sec
    supervision = replace(batch.supervision, corruption_metadata=metadata)
    return replace(batch, supervision=supervision)
