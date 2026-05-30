from __future__ import annotations

from dataclasses import replace

from moat_ovha_torch.data.multimodal.typed_batch import MultimodalEpisodeBatch


def attach_pseudo_label_metadata(batch: MultimodalEpisodeBatch, label_name: str, label_value, source: str, confidence) -> MultimodalEpisodeBatch:
    weak_labels = dict(batch.supervision.weak_labels or {})
    weak_confidence = dict(batch.supervision.weak_label_confidence or {})
    pseudo_sources = dict(batch.supervision.pseudo_label_source or {})
    weak_labels[label_name] = label_value
    weak_confidence[label_name] = confidence
    pseudo_sources[label_name] = source
    supervision = replace(
        batch.supervision,
        weak_labels=weak_labels,
        weak_label_confidence=weak_confidence,
        pseudo_label_source=pseudo_sources,
    )
    return replace(batch, supervision=supervision)
