from __future__ import annotations

from dataclasses import replace

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch, TokenField


def apply_modality_dropout(batch: MultimodalEpisodeBatch, modality: str) -> MultimodalEpisodeBatch:
    if modality not in batch.fields:
        raise ValueError(f"cannot drop missing modality: {modality}")
    field = batch.fields[modality]
    dropped = TokenField(field.modality, field.x * 0, field.pos, field.mask & False, quality=field.quality, attrs=field.attrs)
    fields = {**batch.fields, modality: dropped}
    metadata = dict(batch.supervision.corruption_metadata or {})
    metadata["missing_modalities"] = tuple(sorted(set(metadata.get("missing_modalities", ())) | {modality}))
    supervision = replace(batch.supervision, corruption_metadata=metadata)
    return replace(batch, fields=fields, supervision=supervision)
