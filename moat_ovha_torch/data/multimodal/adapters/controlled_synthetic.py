from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from moat_ovha_torch.data.multimodal.adapters.base import RawDatasetManifest, SupervisionShard, TokenFieldShard, ValidationReport
from moat_ovha_torch.data.multimodal.cache_schema import (
    MultimodalCacheLayout,
    default_data_card,
    file_sha256,
    validate_cache_layout,
)
from moat_ovha_torch.data.multimodal.typed_batch import (
    MultimodalEpisodeBatch,
    ProvenanceBank,
    QueryField,
    SupervisionBank,
    TokenField,
)
from moat_ovha_torch.models.multimodal.baselines import baseline_names_for_task


CONTROLLED_MULTIMODAL_FAMILIES = (
    "tleo_local_evidence",
    "spo_global_prototype",
    "lrio_low_rank_interaction",
    "cato_alignment_transport",
    "rceo_reliability_corruption",
    "mixed_relation_operator",
)

CONTROLLED_OPERATOR_ORDER = ("TLEO", "SPO", "LRIO", "CATO")
CONTROLLED_FAMILY_ACTIVE_OPERATOR = {
    "tleo_local_evidence": "TLEO",
    "spo_global_prototype": "SPO",
    "lrio_low_rank_interaction": "LRIO",
    "cato_alignment_transport": "CATO",
    "rceo_reliability_corruption": "LRIO",
    "mixed_relation_operator": "mixed",
}
CONTROLLED_TRUE_ADAPTER_PARAM_KEYS = {
    "TLEO": ("lengthscale", "local_temperature", "scale", "bias"),
    "SPO": ("prototype_temperature", "prototype_logits_shift", "scale", "bias"),
    "LRIO": ("rank_logits", "interaction_temperature", "scale", "bias"),
    "CATO": ("alignment_temperature", "transport_scale", "scale", "bias"),
}


def required_true_adapter_param_keys_for_family(family: str) -> dict[str, tuple[str, ...]]:
    if family not in CONTROLLED_FAMILY_ACTIVE_OPERATOR:
        raise ValueError(f"unknown controlled multimodal family: {family}")
    active_operator = CONTROLLED_FAMILY_ACTIVE_OPERATOR[family]
    if active_operator == "mixed":
        return dict(CONTROLLED_TRUE_ADAPTER_PARAM_KEYS)
    return {active_operator: CONTROLLED_TRUE_ADAPTER_PARAM_KEYS[active_operator]}


@dataclass(frozen=True)
class ControlledSyntheticMultimodalAdapter:
    name: str = "controlled_multimodal"
    version: str = "v0.1"
    seed: int = 0
    output_dim: int = 2
    field_dim: int = 4
    token_count: int = 8

    def discover_raw(self, raw_root: Path) -> RawDatasetManifest:
        return RawDatasetManifest(dataset_name=self.name, raw_root=raw_root, files={})

    def build_index(self, manifest: RawDatasetManifest) -> list[dict[str, str]]:
        return [{"dataset": manifest.dataset_name, "split": "synthetic"}]

    def extract_token_fields(self, rows, split: str) -> dict[str, TokenFieldShard]:
        root = Path(rows["cache_root"]) if isinstance(rows, dict) and "cache_root" in rows else Path(".")
        return {
            modality: TokenFieldShard(
                modality,
                split,
                root / "token_fields" / f"{modality}_{split}.npy",
                root / "positions" / f"{modality}_pos_{split}.npy",
                root / "masks" / f"{modality}_mask_{split}.npy",
            )
            for modality in ("text", "region", "audio")
        }

    def extract_supervision(self, rows, split: str) -> SupervisionShard:
        root = Path(rows["cache_root"]) if isinstance(rows, dict) and "cache_root" in rows else Path(".")
        return SupervisionShard(
            split=split,
            task_label_path=root / "supervision" / f"task_labels_{split}.npy",
            alignment_pairs_path=root / "supervision" / f"alignment_pairs_{split}.npy",
            corruption_metadata_path=root / "supervision" / f"corruption_{split}.parquet",
        )

    def write_cache(
        self,
        manifest: RawDatasetManifest,
        cache_root: Path,
        split: str,
        cache_version: str,
    ) -> None:
        if split != "train":
            raise ValueError("controlled synthetic formal cache currently supports the train split")
        import numpy as np
        import torch

        layout = MultimodalCacheLayout(cache_root, self.name, cache_version)
        root = layout.root
        _ensure_controlled_cache_dirs(root)
        batches = [
            self.sample_batch(family=family, batch_size=2, query_count=4, device="cpu")
            for family in CONTROLLED_MULTIMODAL_FAMILIES
        ]
        source_ids = [source_id for batch in batches for source_id in batch.provenance.source_id]
        split_ids = {split: source_ids}
        _write_controlled_data_card(root, cache_version)
        (root / "splits.json").write_text(json.dumps(split_ids, sort_keys=True) + "\n")
        _write_controlled_feature_versions(root)
        _write_controlled_provenance(root, manifest, batches, split)

        fields_by_modality = _concatenate_controlled_fields(torch, batches)
        manifest_payload: dict[str, dict[str, str]] = {}
        for modality, tensors in fields_by_modality.items():
            x_path = root / "token_fields" / f"{modality}_{split}.npy"
            pos_path = root / "positions" / f"{modality}_pos_{split}.npy"
            mask_path = root / "masks" / f"{modality}_mask_{split}.npy"
            np.save(x_path, _to_numpy(tensors["x"]))
            np.save(pos_path, _to_numpy(tensors["pos"]))
            np.save(mask_path, _to_numpy(tensors["mask"]))
            manifest_payload[modality] = {
                "x": str(x_path.relative_to(root)),
                "pos": str(pos_path.relative_to(root)),
                "mask": str(mask_path.relative_to(root)),
            }
        (root / "token_fields" / f"manifest_{split}.json").write_text(
            json.dumps(manifest_payload, sort_keys=True) + "\n"
        )

        query = _concatenate_controlled_query(torch, batches)
        np.save(root / "query" / f"query_x_{split}.npy", _to_numpy(query["x"]))
        np.save(root / "query" / f"query_pos_{split}.npy", _to_numpy(query["pos"]))
        np.save(root / "query" / f"query_type_{split}.npy", _to_numpy(query["query_type"]))
        np.save(root / "masks" / f"target_mask_{split}.npy", _to_numpy(torch.cat([batch.target_mask for batch in batches], dim=0)))
        np.save(root / "supervision" / f"target_y_{split}.npy", _to_numpy(torch.cat([batch.target_y for batch in batches], dim=0)))

        true_active = torch.cat([batch.hidden["true_active_operator"] for batch in batches], dim=0)
        true_router = torch.cat([batch.hidden["true_router_weights"] for batch in batches], dim=0)
        true_candidates = torch.cat([batch.hidden["true_candidate_values"] for batch in batches], dim=0)
        true_alignment = torch.cat([batch.hidden["true_alignment_pairs"] for batch in batches], dim=0)
        true_reliability = torch.cat([batch.hidden["true_reliability"] for batch in batches], dim=0)
        true_corruption = torch.cat([batch.hidden["true_corruption_level"] for batch in batches], dim=0)
        np.save(root / "supervision" / f"task_labels_{split}.npy", _to_numpy(true_active))
        np.save(root / "supervision" / f"alignment_pairs_{split}.npy", _to_numpy(true_alignment))
        (root / "supervision" / f"corruption_{split}.parquet").write_text(
            "\n".join(
                json.dumps(
                    {
                        "source_id": source_id,
                        "split": split,
                        "controlled_family": family,
                        "corruption_strength": float(corruption),
                    },
                    sort_keys=True,
                )
                for source_id, family, corruption in zip(
                    source_ids,
                    _family_names_for_batches(batches),
                    _to_numpy(true_corruption).reshape(-1),
                )
            )
            + "\n"
        )
        np.save(root / "controlled_hidden" / f"true_active_operator_{split}.npy", _to_numpy(true_active))
        np.save(root / "controlled_hidden" / f"true_router_weights_{split}.npy", _to_numpy(true_router))
        np.save(root / "controlled_hidden" / f"true_candidate_values_{split}.npy", _to_numpy(true_candidates))
        np.save(root / "controlled_hidden" / f"true_alignment_pairs_{split}.npy", _to_numpy(true_alignment))
        np.save(root / "controlled_hidden" / f"true_reliability_{split}.npy", _to_numpy(true_reliability))
        np.save(root / "controlled_hidden" / f"true_corruption_level_{split}.npy", _to_numpy(true_corruption))
        np.savez(
            root / "controlled_hidden" / f"true_adapter_params_{split}.npz",
            **_adapter_param_arrays(np, batches),
        )
        _write_controlled_checksums(root)

    def validate_cache(self, cache_root: Path, cache_version: str | None = None) -> ValidationReport:
        version = cache_version or self.version
        report = validate_cache_layout(MultimodalCacheLayout(cache_root, self.name, version), splits=("train",))
        return ValidationReport(report.ok, report.errors, report.warnings)

    def sample_batch(
        self,
        family: str,
        batch_size: int = 4,
        query_count: int = 16,
        token_count: int | None = None,
        device: str = "cpu",
    ) -> MultimodalEpisodeBatch:
        if family not in CONTROLLED_MULTIMODAL_FAMILIES:
            raise ValueError(f"unknown controlled multimodal family: {family}")
        import torch

        token_count = token_count or self.token_count
        generator = torch.Generator()
        generator.manual_seed(self.seed + CONTROLLED_MULTIMODAL_FAMILIES.index(family) * 997)
        text_x = torch.randn(batch_size, token_count, self.field_dim, generator=generator).to(device)
        region_x = torch.randn(batch_size, token_count, self.field_dim, generator=generator).to(device)
        audio_x = torch.randn(batch_size, token_count, self.field_dim, generator=generator).to(device)
        text_pos = torch.linspace(0.0, 1.0, token_count).view(1, token_count, 1).repeat(batch_size, 1, 2).to(device)
        region_pos = torch.flip(text_pos, dims=(1,))
        audio_pos = text_pos.clone()
        query_x = torch.randn(batch_size, query_count, self.field_dim, generator=generator).to(device)
        query_pos = torch.linspace(0.0, 1.0, query_count).view(1, query_count, 1).repeat(batch_size, 1, 2).to(device)
        true_alignment_pairs = _alignment_pairs(torch, batch_size, query_count, token_count, device)
        if family == "cato_alignment_transport":
            query_x = _aligned_region_tokens(torch, region_x, true_alignment_pairs)

        router_weights, active = _router_truth(torch, family, batch_size, query_count, device)
        if family == "mixed_relation_operator":
            query_x = _inject_relation_query_code(torch, query_x, active)
        fields = {
            "text": TokenField("text", text_x, text_pos, torch.ones(batch_size, token_count, dtype=torch.bool, device=device)),
            "region": TokenField(
                "region",
                region_x,
                region_pos,
                torch.ones(batch_size, token_count, dtype=torch.bool, device=device),
                quality=torch.ones(batch_size, token_count, 1, device=device),
            ),
            "audio": TokenField(
                "audio",
                audio_x,
                audio_pos,
                torch.ones(batch_size, token_count, dtype=torch.bool, device=device),
                quality=_quality_for_family(torch, family, batch_size, token_count, device),
            ),
        }
        candidate_values = _candidate_values(
            torch,
            text_x,
            region_x,
            audio_x,
            query_x,
            self.output_dim,
            true_alignment_pairs=true_alignment_pairs,
        )
        target_y = (router_weights.unsqueeze(-1) * candidate_values).sum(dim=-2)
        true_lengthscale = torch.full((batch_size, query_count, 1), 0.16, device=device)
        true_rank_logits = _structured_adapter_logits(torch, batch_size, query_count, 4, device, offset=1, scale=1.5)
        true_prototype_logits = _structured_adapter_logits(torch, batch_size, query_count, 4, device, offset=0, scale=1.5)
        true_adapter_params = _true_adapter_params(
            torch,
            family,
            batch_size,
            query_count,
            device,
            true_lengthscale=true_lengthscale,
            true_rank_logits=true_rank_logits,
            true_prototype_logits=true_prototype_logits,
        )
        hidden = {
            "true_active_operator": active,
            "true_router_weights": router_weights,
            "true_candidate_values": candidate_values,
            "true_adapter_params": true_adapter_params,
            "true_alignment_pairs": true_alignment_pairs,
            "true_rank_logits": true_rank_logits,
            "true_prototype_logits": true_prototype_logits,
            "true_lengthscale": true_lengthscale,
            "true_reliability": fields["audio"].quality.mean(dim=1),
            "true_corruption_level": torch.full((batch_size, 1), 0.7 if family == "rceo_reliability_corruption" else 0.0, device=device),
        }
        return MultimodalEpisodeBatch(
            fields=fields,
            query=QueryField(
                x=query_x,
                pos=query_pos,
                query_type=torch.full((batch_size, query_count), -1, dtype=torch.long, device=device),
                mask=torch.ones(batch_size, query_count, dtype=torch.bool, device=device),
            ),
            target_y=target_y,
            target_mask=torch.ones(batch_size, query_count, dtype=torch.bool, device=device),
            task_type=family,
            split="train",
            source_dataset=self.name,
            supervision=SupervisionBank(
                task_label=None,
                alignment_pairs=true_alignment_pairs if family == "cato_alignment_transport" else None,
                alignment_weights=None,
                bbox_targets=None,
                region_targets=None,
                timestamp_targets=None,
                modality_missing_mask=None,
                corruption_metadata={"synthetic_corruption": hidden["true_corruption_level"]},
                weak_labels=None,
                weak_label_confidence=None,
                pseudo_label_source=None,
            ),
            provenance=ProvenanceBank(
                source_id=[f"controlled-{family}-{idx}" for idx in range(batch_size)],
                original_split=["train"] * batch_size,
                raw_ref=["generated"] * batch_size,
                license_tag=["synthetic"] * batch_size,
                preprocessing_version=self.version,
                feature_extractor_version={"text": "synthetic", "region": "synthetic", "audio": "synthetic"},
                pseudo_label_version={},
            ),
            hidden=hidden,
        )


def _ensure_controlled_cache_dirs(root: Path) -> None:
    for folder in ("controlled_hidden", "masks", "positions", "provenance", "query", "supervision", "token_fields"):
        (root / folder).mkdir(parents=True, exist_ok=True)


def _write_controlled_data_card(root: Path, cache_version: str) -> None:
    data_card = default_data_card(
        "controlled_multimodal",
        cache_version,
        ["text", "region", "audio"],
        ["controlled_relation_operator"],
    )
    data_card["controlled_families"] = list(CONTROLLED_MULTIMODAL_FAMILIES)
    data_card["hidden_truth_policy"] = "stored under controlled_hidden/ for oracle evaluation; never model input"
    (root / "data_card.json").write_text(json.dumps(data_card, sort_keys=True) + "\n")


def _write_controlled_feature_versions(root: Path) -> None:
    reference = {"text": "synthetic-v0.1", "region": "synthetic-v0.1", "audio": "synthetic-v0.1"}
    baselines = {"ovha_full": dict(reference)}
    for baseline in baseline_names_for_task("controlled_multimodal"):
        baselines[baseline] = dict(reference)
    payload = {**reference, "baselines": baselines}
    (root / "provenance" / "feature_versions.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
    (root / "provenance" / "pseudo_label_versions.json").write_text(
        json.dumps({"generated_from_splits": [], "version": "none"}, sort_keys=True) + "\n"
    )


def _write_controlled_provenance(
    root: Path,
    manifest: RawDatasetManifest,
    batches: list[MultimodalEpisodeBatch],
    split: str,
) -> None:
    records: list[dict[str, Any]] = []
    source_ids: list[str] = []
    for family, batch in zip(CONTROLLED_MULTIMODAL_FAMILIES, batches):
        source_ids.extend(batch.provenance.source_id)
        for source_id in batch.provenance.source_id:
            records.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "original_split": split,
                    "raw_ref": f"generated://{source_id}",
                    "license_tag": "synthetic-controlled",
                    "preprocessing_version": batch.provenance.preprocessing_version,
                    "controlled_family": family,
                    "hidden_truth_ref": f"controlled_hidden/true_active_operator_{split}.npy",
                }
            )
    (root / "provenance" / f"source_ids_{split}.txt").write_text("\n".join(source_ids) + "\n")
    (root / "provenance" / f"sample_records_{split}.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    )
    (root / "provenance" / f"failed_samples_{split}.jsonl").write_text("")
    (root / "raw_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": manifest.dataset_name,
                "raw_root": str(manifest.raw_root),
                "files": {name: str(path) for name, path in sorted(manifest.files.items())},
                "generated": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (root / "samples.parquet").write_text(
        "\n".join(
            json.dumps(
                {
                    "source_id": record["source_id"],
                    "split": split,
                    "controlled_family": record["controlled_family"],
                },
                sort_keys=True,
            )
            for record in records
        )
        + "\n"
    )


def _concatenate_controlled_fields(torch, batches: list[MultimodalEpisodeBatch]) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for modality in ("text", "region", "audio"):
        fields[modality] = {
            "x": torch.cat([batch.fields[modality].x for batch in batches], dim=0),
            "pos": torch.cat([batch.fields[modality].pos for batch in batches], dim=0),
            "mask": torch.cat([batch.fields[modality].mask for batch in batches], dim=0),
        }
    return fields


def _concatenate_controlled_query(torch, batches: list[MultimodalEpisodeBatch]) -> dict[str, Any]:
    return {
        "x": torch.cat([batch.query.x for batch in batches], dim=0),
        "pos": torch.cat([batch.query.pos for batch in batches], dim=0),
        "query_type": torch.cat([batch.query.query_type for batch in batches], dim=0),
    }


def _family_names_for_batches(batches: list[MultimodalEpisodeBatch]) -> list[str]:
    families: list[str] = []
    for family, batch in zip(CONTROLLED_MULTIMODAL_FAMILIES, batches):
        families.extend([family] * len(batch.provenance.source_id))
    return families


def _adapter_param_arrays(np, batches: list[MultimodalEpisodeBatch]) -> dict[str, Any]:
    total_rows = sum(len(batch.provenance.source_id) for batch in batches)
    query_count = int(batches[0].target_y.shape[1])
    arrays: dict[str, Any] = {}
    mixed_params = batches[-1].hidden["true_adapter_params"]["params_by_operator"]
    for operator, keys in CONTROLLED_TRUE_ADAPTER_PARAM_KEYS.items():
        for key in keys:
            width = int(mixed_params[operator][key].shape[-1])
            arrays[f"{operator}__{key}"] = np.full((total_rows, query_count, width), np.nan, dtype=np.float32)

    offset = 0
    for batch in batches:
        row_count = len(batch.provenance.source_id)
        params_by_operator = batch.hidden["true_adapter_params"]["params_by_operator"]
        for operator, params in params_by_operator.items():
            for key, value in params.items():
                arrays[f"{operator}__{key}"][offset : offset + row_count] = _to_numpy(value).astype(np.float32)
        offset += row_count
    return arrays


def _to_numpy(value):
    return value.detach().cpu().numpy()


def _write_controlled_checksums(root: Path) -> None:
    checksums = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    (root / "checksums.json").write_text(json.dumps(checksums, sort_keys=True) + "\n")


def _candidate_values(torch, text_x, region_x, audio_x, query_x, output_dim: int, *, true_alignment_pairs=None):
    batch_size, query_count, _ = query_x.shape
    text_local = text_x[:, :query_count].mean(dim=-1, keepdim=True)
    if text_local.shape[1] < query_count:
        text_local = text_x.mean(dim=1, keepdim=True).expand(-1, query_count, -1).mean(dim=-1, keepdim=True)
    spo = text_x.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True).expand(batch_size, query_count, 1)
    lrio = (text_x.mean(dim=1) * audio_x.mean(dim=1)).mean(dim=-1, keepdim=True).unsqueeze(1).expand(batch_size, query_count, 1)
    if true_alignment_pairs is None:
        alignment = torch.matmul(query_x, region_x.transpose(1, 2)).softmax(dim=-1)
        cato = torch.matmul(alignment, region_x).mean(dim=-1, keepdim=True)
    else:
        cato = _aligned_region_tokens(torch, region_x, true_alignment_pairs).mean(dim=-1, keepdim=True)
    base = torch.cat([text_local, spo, lrio, cato], dim=-1).unsqueeze(-1)
    scales = torch.linspace(0.5, 1.5, output_dim, device=query_x.device).view(1, 1, 1, output_dim)
    return base * scales


def _aligned_region_tokens(torch, region_x, alignment_pairs):
    region_indices = alignment_pairs[..., 1].unsqueeze(-1).expand(-1, -1, region_x.shape[-1])
    return torch.gather(region_x, dim=1, index=region_indices)


def _router_truth(torch, family: str, batch_size: int, query_count: int, device: str):
    if family == "mixed_relation_operator":
        active = torch.arange(query_count, device=device).view(1, query_count).repeat(batch_size, 1) % len(CONTROLLED_OPERATOR_ORDER)
    else:
        active_operator = CONTROLLED_FAMILY_ACTIVE_OPERATOR[family]
        active = torch.full(
            (batch_size, query_count),
            CONTROLLED_OPERATOR_ORDER.index(active_operator),
            dtype=torch.long,
            device=device,
        )
    weights = torch.nn.functional.one_hot(active, num_classes=len(CONTROLLED_OPERATOR_ORDER)).float()
    if family == "rceo_reliability_corruption":
        weights = 0.85 * weights + 0.15 / len(CONTROLLED_OPERATOR_ORDER)
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, active


def _inject_relation_query_code(torch, query_x, active):
    width = min(int(query_x.shape[-1]), len(CONTROLLED_OPERATOR_ORDER))
    if width <= 0:
        return query_x
    relation_code = torch.nn.functional.one_hot(active, num_classes=len(CONTROLLED_OPERATOR_ORDER)).to(
        dtype=query_x.dtype,
        device=query_x.device,
    )
    coded = query_x.clone()
    coded[..., :width] = relation_code[..., :width]
    return coded


def _structured_adapter_logits(torch, batch_size: int, query_count: int, width: int, device: str, *, offset: int, scale: float):
    indices = (torch.arange(query_count, device=device).view(1, query_count).repeat(batch_size, 1) + offset) % width
    return torch.nn.functional.one_hot(indices, num_classes=width).to(dtype=torch.float32) * scale


def _quality_for_family(torch, family: str, batch_size: int, token_count: int, device: str):
    value = 0.3 if family == "rceo_reliability_corruption" else 1.0
    return torch.full((batch_size, token_count, 1), value, device=device)


def _true_adapter_params(
    torch,
    family: str,
    batch_size: int,
    query_count: int,
    device: str,
    *,
    true_lengthscale,
    true_rank_logits,
    true_prototype_logits,
):
    keys_by_operator = required_true_adapter_param_keys_for_family(family)
    scale = torch.ones(batch_size, query_count, 1, device=device)
    bias = torch.zeros(batch_size, query_count, 1, device=device)
    params_by_operator = {
        "TLEO": {
            "lengthscale": true_lengthscale,
            "local_temperature": torch.ones(batch_size, query_count, 1, device=device),
            "scale": scale,
            "bias": bias,
        },
        "SPO": {
            "prototype_temperature": torch.ones(batch_size, query_count, 1, device=device),
            "prototype_logits_shift": true_prototype_logits,
            "scale": scale,
            "bias": bias,
        },
        "LRIO": {
            "rank_logits": true_rank_logits,
            "interaction_temperature": torch.ones(batch_size, query_count, 1, device=device),
            "scale": scale,
            "bias": bias,
        },
        "CATO": {
            "alignment_temperature": torch.ones(batch_size, query_count, 1, device=device),
            "transport_scale": torch.ones(batch_size, query_count, 1, device=device),
            "scale": scale,
            "bias": bias,
        },
    }
    return {
        "family": family,
        "active_operator": CONTROLLED_FAMILY_ACTIVE_OPERATOR[family],
        "params_by_operator": {
            operator: {key: params_by_operator[operator][key] for key in keys}
            for operator, keys in keys_by_operator.items()
        },
    }


def _alignment_pairs(torch, batch_size: int, query_count: int, token_count: int, device: str):
    pairs = torch.zeros(batch_size, query_count, 2, dtype=torch.long, device=device)
    pairs[..., 0] = torch.arange(query_count, device=device).view(1, query_count) % token_count
    pairs[..., 1] = pairs[..., 0]
    return pairs
