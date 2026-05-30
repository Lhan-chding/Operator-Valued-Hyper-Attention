from __future__ import annotations

from typing import Any


def batch_metadata_column(batch, value: float | bool = 1.0):
    batch_size = int(batch.target_y.shape[0])
    if hasattr(batch.target_y, "new_full"):
        return batch.target_y.new_full((batch_size, 1), float(value))
    raise TypeError("batch.target_y must support new_full for transform metadata")


def batch_strength_from_noise(noise: Any):
    if not hasattr(noise, "detach"):
        raise TypeError("noise must be a tensor-like value with detach")
    detached = noise.detach().abs()
    if len(detached.shape) == 0:
        raise ValueError("noise must include a batch dimension")
    reduce_dims = tuple(range(1, len(detached.shape)))
    if reduce_dims:
        return detached.mean(dim=reduce_dims, keepdim=False).view(detached.shape[0], 1)
    return detached.view(detached.shape[0], 1)


def missing_mask_for(batch, modality: str):
    ordered_modalities = tuple(sorted(batch.fields))
    if modality not in ordered_modalities:
        raise ValueError(f"cannot mark missing modality outside batch fields: {modality}")
    batch_size = int(batch.target_mask.shape[0])
    current = batch.supervision.modality_missing_mask
    if current is None:
        if not hasattr(batch.target_mask, "new_zeros"):
            raise TypeError("batch.target_mask must support new_zeros for missing-modality metadata")
        mask = batch.target_mask.new_zeros((batch_size, len(ordered_modalities))).bool()
    elif tuple(int(dim) for dim in current.shape) != (batch_size, len(ordered_modalities)):
        raise ValueError(
            "existing modality_missing_mask shape must match sorted batch fields: "
            f"expected {(batch_size, len(ordered_modalities))}, got {tuple(int(dim) for dim in current.shape)}"
        )
    elif hasattr(current, "clone"):
        mask = current.clone()
    else:
        raise TypeError("existing modality_missing_mask must support clone")
    mask[:, ordered_modalities.index(modality)] = True
    return mask
