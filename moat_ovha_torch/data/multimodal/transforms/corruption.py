from __future__ import annotations

from dataclasses import replace

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch, TokenField
from moat_ovha_torch.data.multimodal.transforms._metadata import batch_strength_from_noise


def add_gaussian_corruption(batch: MultimodalEpisodeBatch, modality: str, noise) -> MultimodalEpisodeBatch:
    if modality not in batch.fields:
        raise ValueError(f"cannot corrupt missing modality: {modality}")
    field = batch.fields[modality]
    strength = batch_strength_from_noise(noise)
    corrupted = TokenField(
        field.modality,
        field.x + noise,
        field.pos,
        field.mask,
        quality=_decayed_quality(field, strength),
        attrs=field.attrs,
    )
    fields = {**batch.fields, modality: corrupted}
    metadata = dict(batch.supervision.corruption_metadata or {})
    metadata["gaussian_noise_strength"] = strength
    supervision = replace(batch.supervision, corruption_metadata=metadata)
    return replace(batch, fields=fields, supervision=supervision)


def _decayed_quality(field: TokenField, strength):
    if not hasattr(strength, "clamp"):
        return field.quality
    decay = (1.0 - strength).clamp(0.0, 1.0)
    if field.quality is not None:
        quality = field.quality
    elif hasattr(field.mask, "to"):
        quality = field.mask.to(dtype=field.x.dtype, device=field.x.device).unsqueeze(-1)
    else:
        return None
    while decay.ndim < quality.ndim:
        decay = decay.unsqueeze(-1)
    return quality * decay
