from __future__ import annotations

from typing import Sequence

from moat_ovha_torch.data.episodes import EpisodeHiddenInfo, MetaOperatorBatch
from moat_ovha_torch.runtime import require_torch


def collate_episode_batches(items: Sequence[tuple[MetaOperatorBatch, EpisodeHiddenInfo]]) -> tuple[MetaOperatorBatch, list[EpisodeHiddenInfo]]:
    torch = require_torch()
    batches, hidden = zip(*items)
    batch = MetaOperatorBatch(
        context_u=torch.cat([item.context_u for item in batches], dim=0),
        context_q=torch.cat([item.context_q for item in batches], dim=0),
        context_y=torch.cat([item.context_y for item in batches], dim=0),
        target_u=torch.cat([item.target_u for item in batches], dim=0),
        target_q=torch.cat([item.target_q for item in batches], dim=0),
        target_y=torch.cat([item.target_y for item in batches], dim=0),
        support_grid=batches[0].support_grid,
        context_mask=torch.cat([item.context_mask for item in batches], dim=0) if batches[0].context_mask is not None else None,
        target_mask=torch.cat([item.target_mask for item in batches], dim=0) if batches[0].target_mask is not None else None,
    )
    return batch, list(hidden)
