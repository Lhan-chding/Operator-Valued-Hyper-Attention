from __future__ import annotations

from dataclasses import replace

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch
from moat_ovha_torch.data.multimodal.transforms._metadata import batch_metadata_column


def record_hard_negative_metadata(batch: MultimodalEpisodeBatch, mismatch_source_id: str) -> MultimodalEpisodeBatch:
    metadata = dict(batch.supervision.corruption_metadata or {})
    metadata["hard_negative_mismatch"] = batch_metadata_column(batch, True)
    supervision = replace(batch.supervision, corruption_metadata=metadata)
    hidden = {**(batch.hidden or {}), "mismatch_source_id": mismatch_source_id}
    return replace(batch, supervision=supervision, hidden=hidden)
