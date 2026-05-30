from __future__ import annotations

from dataclasses import replace

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch


def record_hard_negative_metadata(batch: MultimodalEpisodeBatch, mismatch_source_id: str) -> MultimodalEpisodeBatch:
    metadata = dict(batch.supervision.corruption_metadata or {})
    metadata["mismatch_source_id"] = mismatch_source_id
    supervision = replace(batch.supervision, corruption_metadata=metadata)
    return replace(batch, supervision=supervision)
