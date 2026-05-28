from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from moat_ovha_torch.data.episodes import MetaOperatorBatch
from moat_ovha_torch.runtime import require_torch


SamplingMode = Literal["uniform", "stratified", "boundary_focused", "random"]
EpisodeMode = Literal["same_sample", "operator_transfer", "few_shot_operator"]


@dataclass(frozen=True)
class EpisodeSamplingRecord:
    context_indices: Any
    query_indices: Any
    demo_indices: Any
    target_indices: Any
    mode: str
    context_sampling: str


class FieldToEpisodeAdapter:
    """Convert full-field benchmark samples into metadata-free OVHA episodes."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.last_record: EpisodeSamplingRecord | None = None

    def sample_episode(
        self,
        sample_batch: dict[str, Any],
        num_demos: int,
        context_points: int,
        query_points: int,
        mode: EpisodeMode,
        context_sampling: SamplingMode,
        episode_id: int = 0,
    ) -> MetaOperatorBatch:
        torch = require_torch()
        if mode not in {"same_sample", "operator_transfer", "few_shot_operator"}:
            raise ValueError(f"unknown episode mode: {mode}")
        if context_sampling not in {"uniform", "stratified", "boundary_focused", "random"}:
            raise ValueError(f"unknown context sampling mode: {context_sampling}")

        input_field = _require_tensor(sample_batch, "input_field")
        output_field = _require_tensor(sample_batch, "output_field")
        coordinates = _coordinates(sample_batch, input_field, torch)
        if input_field.shape[:2] != output_field.shape[:2]:
            raise ValueError("input_field and output_field must share [batch, points] dimensions")
        if coordinates.shape[-2] != input_field.shape[1]:
            raise ValueError("coordinates length must match field point count")

        batch_size, total_points = input_field.shape[0], input_field.shape[1]
        device = input_field.device
        operator_group_id = sample_batch.get("operator_group_id")
        if operator_group_id is not None:
            operator_group_id = operator_group_id.to(device)
        context_indices = _sample_indices(torch, total_points, context_points, context_sampling, self.seed + episode_id, device)
        query_indices = _sample_indices(torch, total_points, query_points, "stratified", self.seed + 1009 + episode_id, device)
        target_source_indices = _target_indices(torch, batch_size, mode, self.seed + episode_id, device)
        demo_indices = _demo_indices(torch, batch_size, num_demos, mode, self.seed + episode_id, device, target_source_indices, operator_group_id)

        context_u = input_field.index_select(0, demo_indices.reshape(-1)).view(batch_size, num_demos, total_points, -1)
        context_q = _expand_coordinates(coordinates, batch_size, num_demos, context_indices)
        context_y_full = output_field.index_select(0, demo_indices.reshape(-1)).view(batch_size, num_demos, total_points, -1)
        context_y = context_y_full.index_select(2, context_indices)

        target_u = input_field.index_select(0, target_source_indices)
        target_q = _expand_target_coordinates(coordinates, batch_size, query_indices)
        target_y = output_field.index_select(0, target_source_indices).index_select(1, query_indices)

        support_grid = coordinates.unsqueeze(0) if coordinates.dim() == 2 else coordinates
        context_mask = torch.ones(batch_size, num_demos, context_points, dtype=torch.bool, device=device)
        target_mask = torch.ones(batch_size, query_points, dtype=torch.bool, device=device)
        self.last_record = EpisodeSamplingRecord(
            context_indices=context_indices.detach().cpu(),
            query_indices=query_indices.detach().cpu(),
            demo_indices=demo_indices.detach().cpu(),
            target_indices=target_source_indices.detach().cpu(),
            mode=mode,
            context_sampling=context_sampling,
        )
        return MetaOperatorBatch(
            context_u=context_u,
            context_q=context_q,
            context_y=context_y,
            target_u=target_u,
            target_q=target_q,
            target_y=target_y,
            support_grid=support_grid,
            context_mask=context_mask,
            target_mask=target_mask,
        )


def _require_tensor(sample_batch: dict[str, Any], key: str):
    if key not in sample_batch:
        raise KeyError(f"sample_batch missing required key: {key}")
    return sample_batch[key]


def _coordinates(sample_batch: dict[str, Any], input_field: Any, torch: Any):
    if "coordinates" in sample_batch:
        coordinates = sample_batch["coordinates"].to(input_field.device)
    else:
        coordinates = torch.linspace(0.0, 1.0, input_field.shape[1], device=input_field.device).view(input_field.shape[1], 1)
    return coordinates


def _sample_indices(torch: Any, total_points: int, count: int, mode: str, seed: int, device: Any):
    if count > total_points:
        raise ValueError(f"cannot sample {count} points from field with {total_points} points")
    if mode in {"uniform", "stratified"}:
        return torch.linspace(0, total_points - 1, count, device=device).round().long().unique(sorted=True)[:count]
    if mode == "boundary_focused":
        edge_count = max(2, count // 2)
        left = torch.arange(0, min(edge_count // 2, total_points), device=device)
        right = torch.arange(max(0, total_points - (edge_count - len(left))), total_points, device=device)
        middle_count = count - len(left) - len(right)
        middle = torch.linspace(0, total_points - 1, max(middle_count, 0), device=device).round().long() if middle_count else torch.empty(0, dtype=torch.long, device=device)
        return torch.cat([left.long(), middle.long(), right.long()]).unique(sorted=True)[:count]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(total_points, generator=generator, device="cpu")[:count].sort().values.to(device)


def _target_indices(torch: Any, batch_size: int, mode: str, seed: int, device: Any):
    if mode in {"same_sample", "operator_transfer"}:
        return torch.arange(batch_size, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 13)
    return torch.randint(0, batch_size, (batch_size,), generator=generator, device="cpu").to(device)


def _demo_indices(
    torch: Any,
    batch_size: int,
    num_demos: int,
    mode: str,
    seed: int,
    device: Any,
    target_indices: Any,
    operator_group_id: Any | None,
):
    if mode == "same_sample":
        return target_indices.view(batch_size, 1).repeat(1, num_demos)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    if mode == "operator_transfer" and operator_group_id is not None:
        return _grouped_demo_indices(torch, batch_size, num_demos, target_indices, operator_group_id, generator, device)
    if mode == "operator_transfer":
        offsets = torch.arange(1, num_demos + 1, device=device).view(1, num_demos)
        return (target_indices.view(batch_size, 1) + offsets) % batch_size
    if operator_group_id is not None:
        return _grouped_demo_indices(torch, batch_size, num_demos, target_indices, operator_group_id, generator, device)
    return torch.randint(0, batch_size, (batch_size, num_demos), generator=generator, device="cpu").to(device)


def _grouped_demo_indices(torch: Any, batch_size: int, num_demos: int, target_indices: Any, operator_group_id: Any, generator: Any, device: Any):
    rows = []
    for row in range(batch_size):
        target_index = int(target_indices[row].detach().cpu())
        group = operator_group_id[target_index]
        candidates = torch.nonzero(operator_group_id == group, as_tuple=False).flatten()
        candidates = candidates[candidates != target_indices[row]]
        if candidates.numel() == 0:
            raise ValueError(f"operator group {group.item()} has no non-target demo candidate")
        choice_positions = torch.randint(0, int(candidates.numel()), (num_demos,), generator=generator, device="cpu").to(device)
        rows.append(candidates.index_select(0, choice_positions))
    return torch.stack(rows, dim=0)


def _expand_coordinates(coordinates: Any, batch_size: int, num_demos: int, indices: Any):
    if coordinates.dim() == 2:
        selected = coordinates.index_select(0, indices)
        return selected.view(1, 1, selected.shape[0], selected.shape[1]).repeat(batch_size, num_demos, 1, 1)
    selected = coordinates.index_select(1, indices)
    return selected.unsqueeze(1).repeat(1, num_demos, 1, 1)


def _expand_target_coordinates(coordinates: Any, batch_size: int, indices: Any):
    if coordinates.dim() == 2:
        selected = coordinates.index_select(0, indices)
        return selected.view(1, selected.shape[0], selected.shape[1]).repeat(batch_size, 1, 1)
    return coordinates.index_select(1, indices)
