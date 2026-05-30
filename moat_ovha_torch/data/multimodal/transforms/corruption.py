from __future__ import annotations

from dataclasses import replace

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch, TokenField


def add_gaussian_corruption(batch: MultimodalEpisodeBatch, modality: str, noise) -> MultimodalEpisodeBatch:
    if modality not in batch.fields:
        raise ValueError(f"cannot corrupt missing modality: {modality}")
    field = batch.fields[modality]
    corrupted = TokenField(field.modality, field.x + noise, field.pos, field.mask, quality=field.quality, attrs=field.attrs)
    fields = {**batch.fields, modality: corrupted}
    metadata = dict(batch.supervision.corruption_metadata or {})
    metadata["corruption_type"] = "gaussian_noise"
    metadata["corruption_strength"] = noise.detach().abs().mean() if hasattr(noise, "detach") else noise
    supervision = replace(batch.supervision, corruption_metadata=metadata)
    return replace(batch, fields=fields, supervision=supervision)
