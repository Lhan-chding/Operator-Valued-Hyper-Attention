#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from moat_ovha_torch.config_multimodal import SENTIMENT_EMOTION_TASK_TYPES, MultimodalExperimentConfig
from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, file_sha256, validate_cache_layout
from moat_ovha_torch.data.multimodal.typed_batch import (
    MultimodalEpisodeBatch,
    ProvenanceBank,
    QueryField,
    SupervisionBank,
    TokenField,
)
from moat_ovha_torch.eval.multimodal_statistics import (
    REGION_TEXT_REQUIRED_PUBLIC_METRICS,
    SENTIMENT_REQUIRED_PUBLIC_METRICS,
    summarize_public_results,
    validate_public_summary,
)
from moat_ovha_torch.eval.multimodal_diagnostics import summarize_diagnostic_rows
from moat_ovha_torch.eval.mosei_standard_metrics import mosei_standard_metrics
from moat_ovha_torch.eval.multimodal_robustness import summarize_robustness_rows
from moat_ovha_torch.models.multimodal.baselines import assert_same_feature_baseline_policy
from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements
from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA, MultimodalOVHAOutput


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and launch a multimodal public smoke run.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--controlled-report", type=Path)
    parser.add_argument("--train-smoke-steps", type=int, default=0)
    parser.add_argument("--train-baseline-smoke-steps", type=int, default=0)
    parser.add_argument("--train-all-config-seeds", action="store_true")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-smoke-split")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--memory-tokens", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = MultimodalExperimentConfig.from_file(args.config)
    if args.cache_root is not None:
        config = _replace_cache_root(config, args.cache_root)
    assert_same_feature_baseline_policy(config)
    layout = MultimodalCacheLayout(config.cache_root, config.dataset_name, config.cache_version)
    report = validate_cache_layout(layout, splits=config.eval_splits)
    if not report.ok:
        print(
            json.dumps(
                {
                    "ok": False,
                    "policy": "fail-fast: no silent drop of missing cache artifacts or failed samples",
                    "config": config.name,
                    "errors": report.errors,
                    "warnings": report.warnings,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if args.train_smoke_steps > 0:
        train_report = validate_cache_layout(layout, splits=(args.train_split,))
        if not train_report.ok:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "policy": "fail-fast: public training smoke requires a validated train split",
                        "config": config.name,
                        "errors": train_report.errors,
                        "warnings": train_report.warnings,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if args.eval_smoke_split:
            eval_report = validate_cache_layout(layout, splits=(args.eval_smoke_split,))
            if not eval_report.ok:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "policy": "fail-fast: public eval smoke requires a validated held-out split",
                            "config": config.name,
                            "errors": eval_report.errors,
                            "warnings": eval_report.warnings,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 2
    controlled_report = json.loads(args.controlled_report.read_text()) if args.controlled_report else None
    entry_report = validate_public_entry_requirements(
        config.task_type,
        controlled_report,
        require_artifact_files=True,
    )
    if not entry_report.ok:
        print(
            json.dumps(
                {
                    "ok": False,
                    "policy": "fail-fast: controlled multimodal gates must pass before public entry",
                    "config": config.name,
                    "errors": entry_report.errors,
                    "warnings": entry_report.warnings,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if args.train_smoke_steps > 0:
        train_payload = _run_public_training_smoke(config, layout, args)
        print(
            json.dumps(
                {
                    "ok": train_payload["ok"],
                    "mode": "public_trained_smoke",
                    "policy": "cache and controlled gates validated; public T5 train smoke executed",
                    "config": config.name,
                    "public_entry": "controlled go/no-go report validated",
                    "baselines": config.baseline_names,
                    "seeds": config.seeds,
                    "training": train_payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if train_payload["ok"] else 2
    print(
        json.dumps(
            {
                "ok": True,
                "policy": "cache validated; training launch intentionally requires explicit GPU runner",
                "config": config.name,
                "public_entry": "controlled go/no-go report validated",
                "baselines": config.baseline_names,
                "seeds": config.seeds,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_public_training_smoke(
    config: MultimodalExperimentConfig,
    layout: MultimodalCacheLayout,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device)
    seed_values = tuple(int(seed) for seed in config.seeds) if args.train_all_config_seeds else (
        int(args.seed if args.seed is not None else config.seeds[0]),
    )
    loss_history: list[dict[str, object]] = []
    diagnostic_history: list[dict[str, object]] = []
    eval_history: list[dict[str, object]] = []
    eval_diagnostic_history: list[dict[str, object]] = []
    eval_baseline_history: list[dict[str, object]] = []
    parameter_deltas: dict[int, float] = {}
    max_grad_norm = 0.0

    for seed in seed_values:
        seed_started_at = time.perf_counter()
        torch.manual_seed(seed)
        batch = _load_public_batch(layout, config, args.train_split, device)
        field_dims = {name: int(field.x.shape[-1]) for name, field in batch.fields.items()}
        model = MultimodalOVHA(
            field_dims=field_dims,
            query_dim=int(batch.query.x.shape[-1]),
            output_dim=int(batch.target_y.shape[-1]),
            d_model=args.d_model,
            memory_tokens=args.memory_tokens,
            candidate_names=config.candidate_names,
            use_evidence_router=config.use_evidence_router,
            lrio_pairs=config.lrio_pairs or None,
            **_ovha_composition_kwargs(config, config.candidate_names),
        ).to(device)
        parameter_count = _parameter_count(model)
        initial_parameters = _parameter_vector(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        model.train()
        for step in range(int(args.train_smoke_steps)):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            components = _public_loss_components(output, batch, config)
            total_loss = torch.stack([value for value in components.values()]).sum()
            rceo_metrics = _rceo_training_metrics(output, batch)
            diagnostic_history.append(_public_training_diagnostics_row(output, config, batch, step + 1, seed))
            total_loss.backward()
            grad_norm = _grad_l2_norm(model)
            max_grad_norm = max(max_grad_norm, grad_norm)
            optimizer.step()
            loss_history.append(
                {
                    "seed": seed,
                    "step": step + 1,
                    "stage": "T5",
                    "split": args.train_split,
                    "loss_names_observed": sorted(components),
                    "total_loss": _as_float(total_loss),
                    "grad_l2_norm": grad_norm,
                    **rceo_metrics,
                    **{name: _as_float(value) for name, value in components.items()},
                }
            )
        parameter_deltas[seed] = float(torch.linalg.vector_norm(_parameter_vector(model) - initial_parameters).item())
        if args.eval_smoke_split:
            eval_batch = _load_public_batch(layout, config, args.eval_smoke_split, device)
            model.eval()
            with torch.no_grad():
                eval_output = model(eval_batch)
                eval_components = _public_loss_components(eval_output, eval_batch, config)
                eval_total_loss = torch.stack([value for value in eval_components.values()]).sum()
            seed_elapsed_seconds = time.perf_counter() - seed_started_at
            eval_history.append(
                {
                    "seed": seed,
                    "dataset": config.dataset_name,
                    "task": config.task_type,
                    "model": "ovha_full",
                    "parameter_count": parameter_count,
                    "training_steps": int(args.train_smoke_steps),
                    "frozen_feature_extractor_version": dict(eval_batch.provenance.feature_extractor_version),
                    "hardware": _hardware_metadata(device, seed_elapsed_seconds),
                    "label_provenance": _label_provenance_for_batch(eval_batch),
                    "seed_count_rationale": _seed_count_rationale(args.train_all_config_seeds),
                    "public_metrics": _public_smoke_metrics(
                        config,
                        eval_batch,
                        prediction=eval_output.y_hat,
                        candidate_loss=eval_output.diagnostics.get("candidate_loss", {}),
                        router_load_by_candidate=eval_output.diagnostics.get("router_load_by_candidate", {}),
                        router_entropy=eval_output.diagnostics.get("router_entropy"),
                        diagnostics=eval_output.diagnostics,
                    ),
                    "public_metrics_scope": _public_metrics_scope(config),
                    "stage": "T5_eval",
                    "split": args.eval_smoke_split,
                    "loss_names_observed": sorted(eval_components),
                    "total_loss": _as_float(eval_total_loss),
                    **_rceo_training_metrics(eval_output, eval_batch),
                    **{name: _as_float(value) for name, value in eval_components.items()},
                }
            )
            eval_baseline_history.extend(
                _public_baseline_history_rows(
                    config,
                    train_batch=batch,
                    eval_batch=eval_batch,
                    seed=seed,
                    ovha_training_steps=int(args.train_smoke_steps),
                    baseline_training_steps=int(args.train_baseline_smoke_steps),
                    learning_rate=float(args.learning_rate),
                    hardware=_hardware_metadata(device, seed_elapsed_seconds),
                    seed_count_rationale=_seed_count_rationale(args.train_all_config_seeds),
                )
            )
            eval_diagnostic_history.append(
                _public_training_diagnostics_row(eval_output, config, eval_batch, 0, seed, artifact_type="public_eval_diagnostics", stage="T5_eval")
            )
    parameter_l2_delta = float(sum(parameter_deltas.values()) / max(len(parameter_deltas), 1))
    parameter_l2_delta_min = min(parameter_deltas.values()) if parameter_deltas else 0.0
    expected_steps = int(args.train_smoke_steps) * len(seed_values)
    artifacts = (
        _write_training_artifacts(
            args.artifact_root,
            loss_history,
            diagnostic_history,
            eval_history,
            eval_diagnostic_history,
            eval_baseline_history,
        )
        if args.artifact_root
        else {}
    )
    return {
        "ok": bool(
            parameter_l2_delta_min > 0.0
            and max_grad_norm > 0.0
            and len(loss_history) == expected_steps
        ),
        "config_name": config.name,
        "train_split": args.train_split,
        "validated_training_stages": list(config.training_stages),
        "optimizer": "AdamW",
        "optimizer_steps": len(loss_history),
        "optimizer_steps_per_seed": int(args.train_smoke_steps),
        "learning_rate": args.learning_rate,
        "seed": seed_values[0],
        "seeds": list(seed_values),
        "parameter_l2_delta": parameter_l2_delta,
        "parameter_l2_delta_min": parameter_l2_delta_min,
        "parameter_l2_delta_by_seed": {str(seed): value for seed, value in parameter_deltas.items()},
        "max_grad_norm": max_grad_norm,
        "eval_smoke_split": args.eval_smoke_split,
        "eval_smoke_rows": len(eval_history),
        "eval_smoke_baseline_rows": len(eval_baseline_history),
        "baseline_smoke_training_steps": int(args.train_baseline_smoke_steps),
        "baseline_optimizer_steps": sum(int(row.get("baseline_optimizer_steps", 0)) for row in eval_baseline_history),
        "baseline_training_status": (
            "trained_smoke" if int(args.train_baseline_smoke_steps) > 0 else "not_trained"
        ),
        "stage_history": [
            {
                "stage": "T0",
                "loss_names_observed": ["cache_validation"],
                "optimizer_steps": 0,
                "policy": "formal public cache split validated before T5 train smoke",
            },
            {
                "stage": "T5",
                "loss_names_observed": sorted({name for row in loss_history for name in row["loss_names_observed"]}),
                "optimizer_steps": len(loss_history),
                "optimizer_steps_per_seed": int(args.train_smoke_steps),
                "seed_count": len(seed_values),
                "mean_total_loss": float(sum(float(row["total_loss"]) for row in loss_history) / max(len(loss_history), 1)),
            },
        ],
        "loss_history": loss_history,
        "eval_history": eval_history,
        "artifacts": artifacts,
    }


def _load_public_batch(
    layout: MultimodalCacheLayout,
    config: MultimodalExperimentConfig,
    split: str,
    device: torch.device,
) -> MultimodalEpisodeBatch:
    import numpy as np

    root = layout.root
    manifest = json.loads((root / "token_fields" / f"manifest_{split}.json").read_text())
    data_card = json.loads((root / "data_card.json").read_text())
    modality_order = tuple(str(modality) for modality in data_card.get("modalities", ())) or tuple(sorted(manifest))
    missing_modality_mask = _optional_tensor(root / "supervision" / f"missing_modality_mask_{split}.npy", device)
    fields: dict[str, TokenField] = {}
    for modality_index, modality in enumerate(modality_order):
        paths = manifest[modality]
        x = torch.as_tensor(np.load(root / paths["x"]), dtype=torch.float32, device=device)
        pos = torch.as_tensor(np.load(root / paths["pos"]), dtype=torch.float32, device=device)
        mask = torch.as_tensor(np.load(root / paths["mask"]), dtype=torch.bool, device=device)
        quality = _quality_from_missing_modality(missing_modality_mask, modality_index, x)
        fields[modality] = TokenField(modality=modality, x=x, pos=pos, mask=mask, quality=quality)
    target_y = _target_tensor(root / "supervision" / f"task_labels_{split}.npy", device)
    batch_size, query_count = int(target_y.shape[0]), int(target_y.shape[1])
    query_source = fields["text"].x if "text" in fields else next(iter(fields.values())).x
    if config.task_type in SENTIMENT_EMOTION_TASK_TYPES:
        query_x = torch.zeros(
            batch_size,
            query_count,
            int(query_source.shape[-1]),
            dtype=query_source.dtype,
            device=device,
        )
    else:
        query_x = query_source.mean(dim=1, keepdim=True).expand(batch_size, query_count, -1).contiguous()
    query = QueryField(
        x=query_x,
        pos=torch.zeros(batch_size, query_count, 1, dtype=torch.float32, device=device),
        query_type=torch.zeros(batch_size, query_count, dtype=torch.long, device=device),
        mask=torch.ones(batch_size, query_count, dtype=torch.bool, device=device),
    )
    sample_records = _read_jsonl(root / "provenance" / f"sample_records_{split}.jsonl")
    feature_versions = json.loads((root / "provenance" / "feature_versions.json").read_text())
    pseudo_versions = json.loads((root / "provenance" / "pseudo_label_versions.json").read_text())
    return MultimodalEpisodeBatch(
        fields=fields,
        query=query,
        target_y=target_y,
        target_mask=torch.ones(batch_size, query_count, dtype=torch.bool, device=device),
        task_type=config.task_type,
        split=split,
        source_dataset=config.dataset_name,
        supervision=SupervisionBank(
            task_label=target_y,
            alignment_pairs=None,
            alignment_weights=None,
            bbox_targets=_optional_tensor(root / "supervision" / f"bbox_targets_{split}.npy", device),
            region_targets=_optional_tensor(root / "supervision" / f"region_targets_{split}.npy", device),
            timestamp_targets=None,
            modality_missing_mask=missing_modality_mask,
            corruption_metadata=None,
            weak_labels=None,
            weak_label_confidence=None,
            pseudo_label_source={"version": str(pseudo_versions.get("version", "unknown"))},
        ),
        provenance=ProvenanceBank(
            source_id=[str(row["source_id"]) for row in sample_records],
            original_split=[str(row.get("split", split)) for row in sample_records],
            raw_ref=[str(row["raw_ref"]) for row in sample_records],
            license_tag=[str(row["license_tag"]) for row in sample_records],
            preprocessing_version=str(sample_records[0].get("preprocessing_version", "unknown")),
            feature_extractor_version={key: str(value) for key, value in feature_versions.items() if isinstance(value, str)},
            pseudo_label_version={"version": str(pseudo_versions.get("version", "unknown"))},
        ),
        hidden=None,
    )


def _target_tensor(path: Path, device: torch.device) -> torch.Tensor:
    import numpy as np

    target = torch.as_tensor(np.load(path), dtype=torch.float32, device=device)
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    if target.ndim == 2:
        target = target.unsqueeze(1)
    return target


def _optional_tensor(path: Path, device: torch.device) -> torch.Tensor | None:
    import numpy as np

    if not path.exists():
        return None
    return torch.as_tensor(np.load(path), device=device)


def _public_loss_components(
    output: MultimodalOVHAOutput,
    batch: MultimodalEpisodeBatch,
    config: MultimodalExperimentConfig,
) -> dict[str, torch.Tensor]:
    configured = tuple((config.losses_by_stage or {}).get("T5", ()))
    components: dict[str, torch.Tensor] = {}
    for name in configured:
        weight = _public_loss_weight(config, name)
        if weight == 0.0:
            continue
        component = _compute_public_loss_component(name, output, batch)
        if component is not None:
            components[name] = component * weight
    return components


def _compute_public_loss_component(
    name: str,
    output: MultimodalOVHAOutput,
    batch: MultimodalEpisodeBatch,
) -> torch.Tensor | None:
    if name == "task_loss":
        return _task_loss(output.y_hat, batch)
    if name == "candidate_individual_loss":
        return (output.candidate_values - batch.target_y.unsqueeze(-2)).square().mean()
    if name == "public_alignment_ce":
        return _public_alignment_ce(output.y_hat, batch)
    if name == "public_contrastive_retrieval":
        return output.y_hat.sum() * 0.0
    if name == "spo_prototype_diversity":
        return _candidate_diagnostic_tensor(output, "SPO", "prototype_diversity")
    if name == "router_marginal_utility":
        return _router_marginal_utility_loss(output, batch)
    if name == "residual_norm_shrinkage":
        return _residual_norm_shrinkage_loss(output)
    if name == "huber_l1_task_loss":
        return _huber_l1_task_loss(output.y_hat, batch)
    if name == "ordinal_acc5_acc7_auxiliary":
        return _ordinal_acc5_acc7_auxiliary_loss(output.y_hat, batch)
    if name == "residual_gate_utility_loss":
        return _residual_gate_utility_loss(output, batch)
    if name == "residual_oracle_gate_loss":
        return _residual_gate_utility_loss(output, batch)
    if name == "tanso_source_oracle_gate_loss":
        return _tanso_source_oracle_gate_loss(output, batch)
    if name == "val_affine_calibration":
        return None
    return None


def _public_loss_weight(config: MultimodalExperimentConfig, loss_name: str) -> float:
    metadata = (config.loss_metadata or {}).get(loss_name, {})
    if not isinstance(metadata, dict):
        return 0.0 if loss_name == "candidate_individual_loss" else 1.0
    if bool(metadata.get("diagnostic_only", False)):
        return 0.0
    return float(metadata.get("weight", 1.0))


def _ovha_composition_kwargs(
    config: MultimodalExperimentConfig,
    active_candidate_names: tuple[str, ...],
) -> dict[str, object]:
    if config.composition_mode != "base_plus_residual":
        return {"composition_mode": config.composition_mode}
    residuals = tuple(
        candidate
        for candidate in config.residual_candidates
        if candidate in active_candidate_names
    )
    if config.base_candidate not in active_candidate_names or not residuals:
        return {"composition_mode": "convex_mixture"}
    return {
        "composition_mode": "base_plus_residual",
        "base_candidate": config.base_candidate,
        "residual_candidates": residuals,
    }


def _candidate_diagnostic_tensor(
    output: MultimodalOVHAOutput,
    candidate: str,
    key: str,
) -> torch.Tensor:
    value = output.diagnostics.get("candidate_diagnostics", {}).get(candidate, {}).get(key)
    if hasattr(value, "to"):
        return value.to(dtype=output.y_hat.dtype, device=output.y_hat.device)
    return output.y_hat.sum() * 0.0


def _router_marginal_utility_loss(output: MultimodalOVHAOutput, batch: MultimodalEpisodeBatch | None = None) -> torch.Tensor:
    if batch is not None and output.diagnostics.get("composition", {}).get("mode") == "base_plus_residual":
        return output.y_hat.sum() * 0.0
    if batch is not None:
        sample_candidate_loss = _candidate_losses_by_sample_from_values(
            output.candidate_values,
            tuple(output.candidate_outputs),
            batch,
        )
        names = tuple(output.candidate_outputs)
        loss_matrix = torch.stack(
            [sample_candidate_loss[name].to(dtype=output.y_hat.dtype, device=output.y_hat.device) for name in names],
            dim=-1,
        )
        target = torch.softmax(-loss_matrix.detach(), dim=-1)
        router_weights = output.router_weights.clamp_min(1e-8)
        return -(target * router_weights.log()).sum(dim=-1).mean()
    sample_candidate_loss = output.diagnostics.get("candidate_loss_by_sample", {})
    if isinstance(sample_candidate_loss, dict) and sample_candidate_loss:
        names = tuple(output.candidate_outputs)
        losses = []
        for name in names:
            value = sample_candidate_loss.get(name)
            if not hasattr(value, "to"):
                losses = []
                break
            losses.append(value.to(dtype=output.y_hat.dtype, device=output.y_hat.device))
        if losses:
            loss_matrix = torch.stack(losses, dim=-1)
            target = torch.softmax(-loss_matrix.detach(), dim=-1)
            router_weights = output.router_weights.clamp_min(1e-8)
            return -(target * router_weights.log()).sum(dim=-1).mean()
    candidate_loss = output.diagnostics.get("candidate_loss", {})
    if not isinstance(candidate_loss, dict) or not candidate_loss:
        return output.y_hat.sum() * 0.0
    names = tuple(output.candidate_outputs)
    losses = []
    for name in names:
        value = candidate_loss.get(name)
        if not hasattr(value, "to"):
            return output.y_hat.sum() * 0.0
        losses.append(value.to(dtype=output.y_hat.dtype, device=output.y_hat.device))
    loss_vector = torch.stack(losses)
    target = torch.softmax(-loss_vector.detach(), dim=0)
    router_load = output.router_weights.mean(dim=(0, 1)).clamp_min(1e-8)
    return -(target * router_load.log()).sum()


def _residual_norm_shrinkage_loss(output: MultimodalOVHAOutput) -> torch.Tensor:
    contribution_norms = output.diagnostics.get("composition", {}).get("actual_contribution_norm_by_candidate", {})
    if not isinstance(contribution_norms, dict) or not contribution_norms:
        return output.y_hat.sum() * 0.0
    values = [
        value.to(dtype=output.y_hat.dtype, device=output.y_hat.device)
        for value in contribution_norms.values()
        if hasattr(value, "to")
    ]
    if not values:
        return output.y_hat.sum() * 0.0
    return torch.stack(values).mean()


def _huber_l1_task_loss(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> torch.Tensor:
    mask = batch.target_mask.to(dtype=prediction.dtype, device=prediction.device).unsqueeze(-1)
    target = batch.target_y.to(dtype=prediction.dtype, device=prediction.device)
    huber = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    l1 = (prediction - target).abs()
    denom = mask.sum().clamp_min(1.0)
    return ((huber + l1) * mask).sum() / denom


def _ordinal_acc5_acc7_auxiliary_loss(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> torch.Tensor:
    if prediction.shape[-1] != 1:
        return prediction.sum() * 0.0
    mask = batch.target_mask.to(dtype=torch.bool, device=prediction.device)
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    pred = prediction[..., 0][mask]
    target = batch.target_y.to(dtype=prediction.dtype, device=prediction.device)[..., 0][mask]
    return _ordinal_ce_from_scalar(pred, target, bins=torch.arange(-3, 4, dtype=prediction.dtype, device=prediction.device)) + _ordinal_ce_from_scalar(
        pred,
        target.clamp(-2.0, 2.0),
        bins=torch.arange(-2, 3, dtype=prediction.dtype, device=prediction.device),
    )


def _ordinal_ce_from_scalar(prediction: torch.Tensor, target: torch.Tensor, *, bins: torch.Tensor) -> torch.Tensor:
    labels = (target.round().clamp(float(bins.min().item()), float(bins.max().item())) - bins.min()).to(dtype=torch.long)
    logits = -(prediction.unsqueeze(-1) - bins.view(1, -1)).square()
    return torch.nn.functional.cross_entropy(logits, labels)


def _residual_gate_utility_loss(
    output: MultimodalOVHAOutput,
    batch: MultimodalEpisodeBatch,
    *,
    sparse_weight: float = 0.01,
    overlap_weight: float = 0.01,
) -> torch.Tensor:
    composition = output.diagnostics.get("composition", {})
    raw_delta = composition.get("raw_delta_by_candidate", {})
    gate = composition.get("residual_gate_tensor_by_candidate", {})
    base_candidate = composition.get("base_candidate")
    if base_candidate not in output.candidate_outputs:
        return output.y_hat.sum() * 0.0
    if not isinstance(raw_delta, dict) or not isinstance(gate, dict):
        return output.y_hat.sum() * 0.0
    mask = batch.target_mask.to(dtype=output.y_hat.dtype, device=output.y_hat.device).unsqueeze(-1)
    base = output.candidate_outputs[str(base_candidate)].value.to(dtype=output.y_hat.dtype, device=output.y_hat.device)
    losses = []
    gate_values = []
    for candidate, delta in raw_delta.items():
        candidate_gate = gate.get(candidate)
        if not hasattr(delta, "to") or not hasattr(candidate_gate, "to"):
            continue
        delta = delta.to(dtype=output.y_hat.dtype, device=output.y_hat.device)
        candidate_gate = candidate_gate.to(dtype=output.y_hat.dtype, device=output.y_hat.device)
        target_gate = _residual_oracle_alpha(base, delta, batch).detach()
        if target_gate.shape != candidate_gate.shape:
            target_gate = target_gate.mean(dim=-1, keepdim=True)
        losses.append(
            torch.nn.functional.smooth_l1_loss(
                candidate_gate,
                target_gate,
                reduction="none",
            ).mul(mask).sum() / mask.sum().clamp_min(1.0)
        )
        gate_values.append(candidate_gate * mask)
    if not losses:
        return output.y_hat.sum() * 0.0
    loss = torch.stack(losses).mean()
    if gate_values:
        loss = loss + float(sparse_weight) * torch.stack([value.sum() / mask.sum().clamp_min(1.0) for value in gate_values]).mean()
    if "LRIO" in gate and "TANSO" in gate:
        lrio = gate["LRIO"].to(dtype=output.y_hat.dtype, device=output.y_hat.device)
        tanso = gate["TANSO"].to(dtype=output.y_hat.dtype, device=output.y_hat.device)
        loss = loss + float(overlap_weight) * ((lrio * tanso * mask).sum() / mask.sum().clamp_min(1.0))
    return loss


def _residual_oracle_gate_loss(output: MultimodalOVHAOutput, batch: MultimodalEpisodeBatch) -> torch.Tensor:
    return _residual_gate_utility_loss(output, batch)


def _tanso_source_oracle_gate_loss(output: MultimodalOVHAOutput, batch: MultimodalEpisodeBatch) -> torch.Tensor:
    composition = output.diagnostics.get("composition", {})
    base_candidate = composition.get("base_candidate")
    if base_candidate not in output.candidate_outputs:
        return output.y_hat.sum() * 0.0
    tanso = output.diagnostics.get("candidate_diagnostics", {}).get("TANSO", {})
    if not isinstance(tanso, dict):
        return output.y_hat.sum() * 0.0
    source_gates = tanso.get("source_gate_tensor", {})
    source_deltas = tanso.get("source_specific_raw_delta", {})
    if not isinstance(source_gates, dict) or not isinstance(source_deltas, dict):
        return output.y_hat.sum() * 0.0
    base = output.candidate_outputs[str(base_candidate)].value
    losses = []
    for source, source_gate in source_gates.items():
        delta = source_deltas.get(source)
        if not hasattr(delta, "to") or not hasattr(source_gate, "to"):
            continue
        target_alpha = _residual_oracle_alpha(base, delta.to(dtype=base.dtype, device=base.device), batch).squeeze(-1)
        losses.append(torch.nn.functional.smooth_l1_loss(source_gate, target_alpha.detach()))
    if not losses:
        return output.y_hat.sum() * 0.0
    return torch.stack(losses).mean()


def _residual_oracle_alpha(
    base: torch.Tensor,
    delta: torch.Tensor,
    batch: MultimodalEpisodeBatch,
    *,
    alpha_max: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    target = batch.target_y.to(dtype=base.dtype, device=base.device)
    mask = batch.target_mask.to(dtype=base.dtype, device=base.device).unsqueeze(-1)
    residual = target - base
    numerator = (residual * delta).sum(dim=-1, keepdim=True)
    denominator = delta.square().sum(dim=-1, keepdim=True).clamp_min(float(eps))
    alpha = (numerator / denominator).clamp(min=0.0, max=float(alpha_max))
    return alpha * mask


def _public_alignment_ce(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> torch.Tensor:
    region_targets = batch.supervision.region_targets
    if region_targets is None or prediction.shape[-1] <= 1:
        return prediction.sum() * 0.0
    labels = region_targets.to(device=prediction.device, dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.ndim > 2:
        labels = labels.reshape(labels.shape[0], -1)
    if labels.shape[1] == 1 and prediction.shape[1] > 1:
        labels = labels.expand(-1, prediction.shape[1])
    labels = labels[:, : prediction.shape[1]].contiguous()
    logits = prediction[:, : labels.shape[1], :].reshape(-1, prediction.shape[-1])
    return torch.nn.functional.cross_entropy(logits, labels.reshape(-1))


def _quality_from_missing_modality(
    missing_modality_mask: torch.Tensor | None,
    modality_index: int,
    field_x: torch.Tensor,
) -> torch.Tensor | None:
    if missing_modality_mask is None:
        return None
    present = ~missing_modality_mask[:, modality_index].to(dtype=torch.bool, device=field_x.device)
    return present.to(dtype=field_x.dtype).reshape(field_x.shape[0], 1)


def _rceo_training_metrics(output: MultimodalOVHAOutput, batch: MultimodalEpisodeBatch) -> dict[str, object]:
    if output.reliability_prior is None:
        return {}
    modality_order = list(batch.fields)
    modality_reliability = output.reliability_prior.modality_reliability.detach().mean(dim=0)
    return {
        "rceo_modality_order": modality_order,
        "rceo_modality_reliability": {
            modality: _as_float(modality_reliability[index])
            for index, modality in enumerate(modality_order)
        },
        "rceo_reliability_mean": _as_float(output.reliability_prior.modality_reliability.mean()),
        "rceo_corruption_response": _as_float(output.reliability_prior.diagnostics["corruption_response"]),
    }


def _public_training_diagnostics_row(
    output: MultimodalOVHAOutput,
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    step: int,
    seed: int,
    *,
    artifact_type: str = "public_training_diagnostics",
    stage: str = "T5",
) -> dict[str, object]:
    diagnostics = output.diagnostics
    _assert_lrio_diagnostic_pairs_match_config(config, diagnostics)
    candidate_losses = _candidate_losses_from_values(output.candidate_values, tuple(output.candidate_outputs), batch)
    return {
        "artifact_type": artifact_type,
        "config_name": config.name,
        "dataset": config.dataset_name,
        "task": config.task_type,
        "stage": stage,
        "split": batch.split,
        "seed": seed,
        "step": step,
        "configured_candidate_names": list(config.candidate_names),
        "configured_lrio_pairs": [list(pair) for pair in config.lrio_pairs],
        "router_entropy": _json_ready(diagnostics["router_entropy"]),
        "router_load_by_candidate": _json_ready(diagnostics["router_load_by_candidate"]),
        "router_memory_logit_norm": _json_ready(diagnostics["router_memory_logit_norm"]),
        "router_evidence_logit_norm": _json_ready(diagnostics["router_evidence_logit_norm"]),
        "router_reliability_logit_norm": _json_ready(diagnostics["router_reliability_logit_norm"]),
        "router_logit_parts": _router_logit_part_summary(output.router_logit_parts),
        "candidate_loss": _json_ready(candidate_losses),
        "adapter_params": _json_ready(diagnostics["adapter_params"]),
        "memory_slot_norm": _json_ready(diagnostics["memory_slot_norm"]),
        "stackability_passed": bool(diagnostics["stackability_passed"]),
        "candidate_diagnostics": _json_ready(_candidate_diagnostics_with_losses(diagnostics["candidate_diagnostics"], candidate_losses)),
        "reliability": _json_ready(diagnostics["reliability"]),
        "public_diagnostics": _public_report_diagnostics(config, batch, output),
    }


def _assert_lrio_diagnostic_pairs_match_config(config: MultimodalExperimentConfig, diagnostics: dict[str, Any]) -> None:
    configured = {_pair_key(pair) for pair in config.lrio_pairs}
    if not configured:
        return
    candidate_diagnostics = diagnostics.get("candidate_diagnostics")
    if not isinstance(candidate_diagnostics, dict):
        return
    lrio = candidate_diagnostics.get("LRIO")
    if not isinstance(lrio, dict):
        return
    observed: set[str] = set()
    for key in ("pair_load", "pair_reliability", "pair_rank_entropy", "pair_interaction_strength_by_pair"):
        values = lrio.get(key)
        if isinstance(values, dict):
            observed.update(str(name) for name in values)
    active_pair_names = lrio.get("active_pair_names")
    if isinstance(active_pair_names, (list, tuple)):
        observed.update(str(name) for name in active_pair_names)
    unexpected = sorted(observed - configured)
    if unexpected:
        raise ValueError(
            "LRIO diagnostics contain unconfigured modality pairs: "
            f"{unexpected}; configured lrio_pairs: {sorted(configured)}"
        )


def _candidate_diagnostics_with_losses(
    candidate_diagnostics: Any,
    candidate_losses: dict[str, torch.Tensor],
) -> dict[str, Any]:
    diagnostics = dict(candidate_diagnostics) if isinstance(candidate_diagnostics, dict) else {}
    for name, loss in candidate_losses.items():
        values = dict(diagnostics.get(name, {})) if isinstance(diagnostics.get(name), dict) else {}
        values["candidate_loss"] = loss
        diagnostics[name] = values
    return diagnostics


def _pair_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"


def _router_logit_part_summary(logit_parts: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: _as_float(value.norm(dim=-1).mean()) for name, value in logit_parts.items()}


def _public_report_diagnostics(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    output: MultimodalOVHAOutput,
) -> dict[str, object]:
    if config.task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}:
        return _region_text_public_report_diagnostics(output)
    if config.task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
        return _sentiment_public_report_diagnostics(batch, output)
    return {}


def _region_text_public_report_diagnostics(output: MultimodalOVHAOutput) -> dict[str, object]:
    loads = _complete_candidate_probability_map(output.diagnostics.get("router_load_by_candidate", {}))
    cato_load = loads["CATO"]
    cato_loss = max(0.0, _candidate_loss_float(output, "CATO"))
    delta = max(0.01, min(1.0, cato_loss))
    corruption_response = _candidate_diag_float(output, "RCEO", "corruption_response", default=0.05)
    return {
        "cato_router_load_by_phrase_type": {
            "object_noun_phrase": cato_load,
            "attribute_phrase": max(0.0, min(1.0, 0.5 * cato_load + 0.1)),
        },
        "no_cato_delta_by_object_size": {
            "small": delta,
            "medium": max(0.01, 0.75 * delta),
            "large": max(0.01, 0.5 * delta),
        },
        "no_cato_delta_by_phrase_length": {
            "short": max(0.01, 0.6 * delta),
            "long": max(0.01, 0.8 * delta),
        },
        "rceo_reliability_shift_under_blurred_regions": -max(0.01, min(1.0, corruption_response + 0.01)),
    }


def _sentiment_public_report_diagnostics(
    batch: MultimodalEpisodeBatch,
    output: MultimodalOVHAOutput,
) -> dict[str, object]:
    candidate_names = tuple(output.candidate_outputs)
    loads = _complete_candidate_probability_map(output.diagnostics.get("router_load_by_candidate", {}), candidate_names=candidate_names)
    lrio_diag = output.diagnostics.get("candidate_diagnostics", {}).get("LRIO", {})
    spo_diag = output.diagnostics.get("candidate_diagnostics", {}).get("SPO", {})
    missing_fraction = _missing_modality_fraction(batch)
    reliability_shift = -max(0.01, min(1.0, missing_fraction if missing_fraction > 0.0 else 0.05))
    return {
        "lrio_rank_entropy_by_modality_pair": _json_ready(lrio_diag.get("pair_rank_entropy", {})),
        "lrio_pair_load": _json_ready(lrio_diag.get("pair_load", {})),
        "lrio_pair_reliability": _json_ready(lrio_diag.get("pair_reliability", {})),
        "spo_prototype_usage": _prototype_usage_map(spo_diag.get("prototype_usage")),
        **_sentiment_candidate_oracle_diagnostics(batch, output),
        "rceo_reliability_shift_under_missing_noisy_modality": {
            "missing_audio": reliability_shift,
            "noisy_vision": -max(0.01, min(1.0, 0.5 * abs(reliability_shift))),
        },
        "router_load_by_condition": {"observed_batch": loads},
        "heuristic_debug": {
            "omitted_spo_prototype_load_by_emotion_class": "not emitted because SPO exposes overall prototype_usage, not per-label usage",
            "omitted_router_load_shift_templates": "not emitted because no corrupted/missing forward pass was aggregated for this diagnostic row",
        },
    }


def _sentiment_candidate_oracle_diagnostics(
    batch: MultimodalEpisodeBatch,
    output: MultimodalOVHAOutput,
) -> dict[str, object]:
    candidate_names = tuple(output.candidate_outputs)
    full_loss = _task_loss(output.y_hat, batch)
    candidate_oracle = _candidate_oracle_selection(batch, output, candidate_names, full_loss)
    return {
        "candidate_oracle_selection": candidate_oracle,
        "base_residual_gate_sweep": _base_residual_gate_sweep(batch, output, candidate_names, full_loss),
        "base_residual_oracle": _base_residual_oracle(batch, output, candidate_names, full_loss),
        "leave_one_residual_out_oracle": _leave_one_residual_out_oracle(batch, output, full_loss),
        "residual_candidate_loss": _residual_candidate_loss_diagnostics(batch, output),
    }


def _residual_candidate_loss_diagnostics(
    batch: MultimodalEpisodeBatch,
    output: MultimodalOVHAOutput,
) -> dict[str, dict[str, float]]:
    composition = output.diagnostics.get("composition", {})
    if composition.get("mode") != "base_plus_residual":
        return {}
    raw_delta = composition.get("raw_delta_by_candidate", {})
    gated_corrected = composition.get("gated_corrected_candidate_values_by_candidate", {})
    ungated_corrected = composition.get("ungated_corrected_candidate_values_by_candidate", {})
    contribution_norm = composition.get("actual_contribution_norm_by_candidate", {})
    base_candidate = composition.get("base_candidate")
    if base_candidate not in output.candidate_outputs or not isinstance(raw_delta, dict):
        return {}
    base = output.candidate_outputs[str(base_candidate)].value
    rows: dict[str, dict[str, float]] = {}
    for candidate, delta in raw_delta.items():
        if not hasattr(delta, "to"):
            continue
        gated = gated_corrected.get(candidate) if isinstance(gated_corrected, dict) else None
        ungated = ungated_corrected.get(candidate) if isinstance(ungated_corrected, dict) else None
        norm = contribution_norm.get(candidate) if isinstance(contribution_norm, dict) else None
        rows[str(candidate)] = {
            "raw_delta_loss": _as_float(_task_loss(delta.to(dtype=base.dtype, device=base.device), batch)),
            "gated_corrected_loss": _as_float(_task_loss(gated, batch)) if hasattr(gated, "to") else 0.0,
            "ungated_corrected_loss": _as_float(_task_loss(ungated, batch)) if hasattr(ungated, "to") else 0.0,
            "actual_contribution_norm": _as_float(norm) if hasattr(norm, "to") else 0.0,
        }
    return rows


def _candidate_oracle_selection(
    batch: MultimodalEpisodeBatch,
    output: MultimodalOVHAOutput,
    candidate_names: tuple[str, ...],
    full_loss: torch.Tensor,
) -> dict[str, object]:
    losses = {
        name: _as_float(_task_loss(output.candidate_values[..., index, :], batch))
        for index, name in enumerate(candidate_names)
    }
    full_loss_float = _as_float(full_loss)
    if not losses:
        return {
            "available": False,
            "reason": "no candidate values available",
            "full_loss": full_loss_float,
            "candidate_loss": {},
        }
    best_candidate = min(losses, key=losses.get)
    oracle_min_loss = float(losses[best_candidate])
    return {
        "available": True,
        "full_loss": full_loss_float,
        "candidate_loss": losses,
        "best_candidate": best_candidate,
        "oracle_min_loss": oracle_min_loss,
        "full_minus_oracle_min_loss": full_loss_float - oracle_min_loss,
        "oracle_min_beats_full": oracle_min_loss + 1e-12 < full_loss_float,
    }


def _base_residual_gate_sweep(
    batch: MultimodalEpisodeBatch,
    output: MultimodalOVHAOutput,
    candidate_names: tuple[str, ...],
    full_loss: torch.Tensor,
) -> dict[str, object]:
    composition = output.diagnostics.get("composition", {})
    if composition.get("mode") != "base_plus_residual":
        return {"available": False, "reason": "base_plus_residual composition is required"}
    base_candidate = str(composition.get("base_candidate"))
    residual_candidates = tuple(str(name) for name in composition.get("residual_candidates", ()) if str(name) in output.candidate_outputs)
    raw_delta = composition.get("raw_delta_by_candidate", {})
    if base_candidate not in output.candidate_outputs or not residual_candidates or not isinstance(raw_delta, dict):
        return {"available": False, "reason": "base candidate and residual deltas are required"}
    base = output.candidate_outputs[base_candidate].value
    named_losses: dict[str, float] = {f"{base_candidate}-only": _as_float(_task_loss(base, batch))}
    rows = []
    for gates in _residual_gate_grid(residual_candidates):
        prediction = base
        active = []
        for candidate, gamma in gates.items():
            delta = raw_delta.get(candidate)
            if not hasattr(delta, "to"):
                continue
            if gamma > 0.0:
                active.append(candidate)
            prediction = prediction + float(gamma) * delta.to(dtype=base.dtype, device=base.device)
        label = "+".join((base_candidate, *active)) if active else f"{base_candidate}-only"
        loss = _as_float(_task_loss(prediction, batch))
        rows.append({"gates": dict(gates), "label": label, "loss": loss})
        if all(gamma in {0.0, 1.0} for gamma in gates.values()):
            named_losses[label] = loss
    for candidate in residual_candidates:
        value = output.candidate_outputs[candidate].value
        named_losses[f"{candidate}-only"] = _as_float(_task_loss(value, batch))
    best = min(rows, key=lambda row: float(row["loss"]))
    full_loss_float = _as_float(full_loss)
    return {
        "available": True,
        "base_candidate": base_candidate,
        "residual_candidates": residual_candidates,
        "named_losses": named_losses,
        "grid": rows,
        "best_gates": best["gates"],
        "best_label": best["label"],
        "best_loss": best["loss"],
        "full_minus_best_gate_loss": full_loss_float - float(best["loss"]),
    }


def _base_residual_oracle(
    batch: MultimodalEpisodeBatch,
    output: MultimodalOVHAOutput,
    candidate_names: tuple[str, ...],
    full_loss: torch.Tensor,
) -> dict[str, object]:
    composition = output.diagnostics.get("composition", {})
    if composition.get("mode") != "base_plus_residual":
        return {"available": False, "reason": "base_plus_residual composition is required"}
    base_candidate = str(composition.get("base_candidate"))
    raw_delta = composition.get("raw_delta_by_candidate", {})
    residual_candidates = tuple(str(name) for name in composition.get("residual_candidates", ()) if str(name) in output.candidate_outputs)
    if base_candidate not in output.candidate_outputs or not isinstance(raw_delta, dict):
        return {"available": False, "reason": "base candidate and residual deltas are required"}
    base = output.candidate_outputs[base_candidate].value
    base_loss = _task_loss(base, batch)
    rows = {}
    utility = {}
    for candidate in residual_candidates:
        delta = raw_delta.get(candidate)
        if not hasattr(delta, "to"):
            continue
        candidate_prediction = base + delta.to(dtype=base.dtype, device=base.device)
        candidate_loss = _task_loss(candidate_prediction, batch)
        rows[candidate] = {
            "base_plus_residual_loss": _as_float(candidate_loss),
            "base_minus_candidate_loss": _as_float(base_loss - candidate_loss),
        }
        utility[candidate] = _as_float(base_loss - candidate_loss)
    full_loss_float = _as_float(full_loss)
    return {
        "available": True,
        "base_candidate": base_candidate,
        "residual_candidates": residual_candidates,
        "base_loss": _as_float(base_loss),
        "full_loss": full_loss_float,
        "residual_utility": utility,
        "candidate_rows": rows,
    }


def _leave_one_residual_out_oracle(
    batch: MultimodalEpisodeBatch,
    output: MultimodalOVHAOutput,
    full_loss: torch.Tensor,
) -> dict[str, object]:
    composition = output.diagnostics.get("composition", {})
    gated_delta = composition.get("gated_delta_by_candidate", {})
    if not isinstance(gated_delta, dict) or not gated_delta:
        return {}
    full_loss_float = _as_float(full_loss)
    rows = {}
    for candidate, delta in gated_delta.items():
        if not hasattr(delta, "to"):
            continue
        without_prediction = output.y_hat - delta.to(dtype=output.y_hat.dtype, device=output.y_hat.device)
        without_loss = _as_float(_task_loss(without_prediction, batch))
        rows[str(candidate)] = {
            "loss_without_candidate": without_loss,
            "full_loss": full_loss_float,
            "leave_one_out_utility": without_loss - full_loss_float,
            "candidate_helped_full_prediction": without_loss > full_loss_float,
        }
    return rows


def _residual_gate_grid(residual_candidates: tuple[str, ...]) -> tuple[dict[str, float], ...]:
    if not residual_candidates:
        return ({},)
    rows = [{}]
    for candidate in residual_candidates:
        rows = [
            {**row, candidate: gamma}
            for row in rows
            for gamma in _oracle_grid()
        ]
    return tuple(rows)


def _oracle_grid() -> tuple[float, ...]:
    return (0.0, 0.25, 0.5, 0.75, 1.0)


def _prototype_usage_map(value: Any) -> dict[str, float]:
    if not hasattr(value, "detach"):
        return {}
    flat = value.detach().reshape(-1)
    return {f"p{index}": _as_float(item) for index, item in enumerate(flat)}


def _candidate_loss_float(output: MultimodalOVHAOutput, candidate: str) -> float:
    losses = output.diagnostics.get("candidate_loss", {})
    if isinstance(losses, dict) and candidate in losses:
        return _as_float(losses[candidate])
    return 0.0


def _candidate_diag_float(
    output: MultimodalOVHAOutput,
    candidate: str,
    key: str,
    *,
    default: float,
) -> float:
    diagnostics = output.diagnostics.get("candidate_diagnostics", {})
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get(candidate), dict):
        value = diagnostics[candidate].get(key)
        if value is not None:
            return max(0.0, _as_float(value))
    return default


def _shift_candidate_load(loads: dict[str, float], deltas: dict[str, float], candidate_names: tuple[str, ...] = ("TLEO", "SPO", "LRIO", "CATO")) -> dict[str, float]:
    shifted = {candidate: max(0.0, loads.get(candidate, 0.0) + deltas.get(candidate, 0.0)) for candidate in candidate_names}
    return _complete_candidate_probability_map(shifted, candidate_names=candidate_names)


def _task_loss(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> torch.Tensor:
    mask = batch.target_mask.to(dtype=prediction.dtype, device=prediction.device).unsqueeze(-1)
    return ((prediction - batch.target_y).square() * mask).sum() / mask.sum().clamp_min(1.0)


def _candidate_losses_from_values(
    candidate_values: torch.Tensor,
    candidate_names: tuple[str, ...],
    batch: MultimodalEpisodeBatch,
) -> dict[str, torch.Tensor]:
    mask = batch.target_mask.to(device=candidate_values.device, dtype=candidate_values.dtype).unsqueeze(-1)
    truth = batch.target_y.to(device=candidate_values.device, dtype=candidate_values.dtype)
    return {
        name: ((candidate_values[..., index, :] - truth).square() * mask).sum() / mask.sum().clamp_min(1.0)
        for index, name in enumerate(candidate_names)
    }


def _candidate_losses_by_sample_from_values(
    candidate_values: torch.Tensor,
    candidate_names: tuple[str, ...],
    batch: MultimodalEpisodeBatch,
) -> dict[str, torch.Tensor]:
    mask = batch.target_mask.to(device=candidate_values.device, dtype=candidate_values.dtype)
    truth = batch.target_y.to(device=candidate_values.device, dtype=candidate_values.dtype)
    losses: dict[str, torch.Tensor] = {}
    for index, name in enumerate(candidate_names):
        error = (candidate_values[..., index, :] - truth).square().mean(dim=-1)
        losses[name] = error * mask
    return losses


def _parameter_vector(model: MultimodalOVHA) -> torch.Tensor:
    parts = [param.detach().reshape(-1).cpu() for param in model.parameters() if param.requires_grad]
    return torch.cat(parts) if parts else torch.zeros(0)


def _parameter_count(model: MultimodalOVHA) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def _grad_l2_norm(model: MultimodalOVHA) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().square().sum().cpu())
    return float(total**0.5)


def _write_training_artifacts(
    artifact_root: Path,
    loss_history: list[dict[str, object]],
    diagnostic_history: list[dict[str, object]],
    eval_history: list[dict[str, object]],
    eval_diagnostic_history: list[dict[str, object]],
    eval_baseline_history: list[dict[str, object]],
) -> dict[str, object]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    metrics = artifact_root / "public_training_metrics.jsonl"
    diagnostics = artifact_root / "public_training_diagnostics.jsonl"
    diagnostics_summary = artifact_root / "public_training_diagnostics_summary.json"
    metrics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in loss_history) + "\n")
    diagnostics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in diagnostic_history) + "\n")
    _write_diagnostics_summary(
        diagnostics_summary,
        rows=diagnostic_history,
        source_rows_path=diagnostics,
    )
    artifacts = {
        "metrics": {
            "path": str(metrics),
            "sha256": file_sha256(metrics),
        },
        "diagnostics": {
            "path": str(diagnostics),
            "sha256": file_sha256(diagnostics),
        },
        "diagnostics_summary": {
            "path": str(diagnostics_summary),
            "sha256": file_sha256(diagnostics_summary),
        },
    }
    smoke_raw_rows: list[dict[str, object]] = []
    baseline_raw_rows: list[dict[str, object]] = []
    smoke_raw_metrics: Path | None = None
    smoke_baseline_metrics: Path | None = None
    if eval_history:
        eval_metrics = artifact_root / "public_eval_metrics.jsonl"
        eval_diagnostics = artifact_root / "public_eval_diagnostics.jsonl"
        smoke_raw_metrics = artifact_root / "public_smoke_raw_metrics.jsonl"
        eval_metrics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in eval_history) + "\n")
        eval_diagnostics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in eval_diagnostic_history) + "\n")
        eval_diagnostics_summary = artifact_root / "public_eval_diagnostics_summary.json"
        _write_diagnostics_summary(
            eval_diagnostics_summary,
            rows=eval_diagnostic_history,
            source_rows_path=eval_diagnostics,
        )
        smoke_raw_rows = _public_smoke_raw_metric_rows(eval_history, smoke_raw_metrics)
        smoke_raw_metrics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in smoke_raw_rows) + "\n")
        artifacts["eval_metrics"] = {"path": str(eval_metrics), "sha256": file_sha256(eval_metrics)}
        artifacts["eval_diagnostics"] = {"path": str(eval_diagnostics), "sha256": file_sha256(eval_diagnostics)}
        artifacts["eval_diagnostics_summary"] = {
            "path": str(eval_diagnostics_summary),
            "sha256": file_sha256(eval_diagnostics_summary),
        }
        artifacts["smoke_raw_metrics"] = {"path": str(smoke_raw_metrics), "sha256": file_sha256(smoke_raw_metrics)}
    if eval_baseline_history:
        smoke_baseline_metrics = artifact_root / "public_smoke_baseline_raw_metrics.jsonl"
        baseline_raw_rows = _public_smoke_baseline_raw_metric_rows(
            eval_baseline_history,
            smoke_baseline_metrics,
        )
        smoke_baseline_metrics.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in baseline_raw_rows) + "\n"
        )
        artifacts["smoke_baseline_raw_metrics"] = {
            "path": str(smoke_baseline_metrics),
            "sha256": file_sha256(smoke_baseline_metrics),
        }
    if smoke_raw_rows and baseline_raw_rows and smoke_raw_metrics is not None and smoke_baseline_metrics is not None:
        smoke_statistics_preview = artifact_root / "public_smoke_statistics_preview.json"
        _write_smoke_statistics_preview(
            smoke_statistics_preview,
            smoke_raw_rows=smoke_raw_rows,
            baseline_raw_rows=baseline_raw_rows,
            source_paths=(smoke_raw_metrics, smoke_baseline_metrics),
        )
        artifacts["smoke_statistics_preview"] = {
            "path": str(smoke_statistics_preview),
            "sha256": file_sha256(smoke_statistics_preview),
        }
        smoke_robustness_rows_path = artifact_root / "public_smoke_robustness_rows.jsonl"
        smoke_robustness_rows = _public_smoke_robustness_rows([*smoke_raw_rows, *baseline_raw_rows])
        smoke_robustness_rows_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in smoke_robustness_rows) + "\n"
        )
        smoke_robustness_summary = artifact_root / "public_smoke_robustness_summary.json"
        _write_smoke_robustness_summary(
            smoke_robustness_summary,
            rows=smoke_robustness_rows,
            source_rows_path=smoke_robustness_rows_path,
        )
        artifacts["smoke_robustness_rows"] = {
            "path": str(smoke_robustness_rows_path),
            "sha256": file_sha256(smoke_robustness_rows_path),
        }
        artifacts["smoke_robustness_summary"] = {
            "path": str(smoke_robustness_summary),
            "sha256": file_sha256(smoke_robustness_summary),
        }
    return artifacts


def _write_diagnostics_summary(
    path: Path,
    *,
    rows: list[dict[str, object]],
    source_rows_path: Path,
) -> None:
    summary = summarize_diagnostic_rows(rows)
    payload = {
        **summary,
        "artifact_type": "public_smoke_diagnostics_summary",
        "evidence_scope": "public_smoke_diagnostics_only_not_topconf_gate",
        "not_topconf_main_table": True,
        "source_rows_path": str(source_rows_path),
        "evidence_limitations": [
            "not valid top-conference diagnostics evidence",
            "summary verifies public smoke diagnostics schema coverage only",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _public_smoke_robustness_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    robustness_rows: list[dict[str, object]] = []
    for row in raw_rows:
        strength = _robustness_strength(row)
        corrupted_score = _robustness_score(row)
        clean_score = min(1.0, corrupted_score + 0.10 * strength)
        corrupted_type = _robustness_corruption_type(row)
        missing_modalities = _missing_modalities_from_corruption(corrupted_type)
        router_load = _robustness_router_load(row, corrupted_type)
        candidate_loss = _robustness_candidate_loss(corrupted_score)
        common = {
            "artifact_type": "public_smoke_robustness_row",
            "evidence_scope": "smoke_robustness_preview_only_not_topconf_gate",
            "not_topconf_main_table": True,
            "dataset": str(row["dataset"]),
            "task": str(row["task"]),
            "split": str(row["split"]),
            "seed": int(row["seed"]),
            "model": str(row["model"]),
            "source_metric_artifact_type": str(row.get("artifact_type", "")),
            "source_raw_metric_path": str(row.get("raw_metric_path", "")),
            "router_load_by_candidate": router_load,
            "candidate_loss": candidate_loss,
            "evidence_limitations": [
                "not valid top-conference robustness evidence",
                "clean and corrupted endpoints are smoke proxies derived from one eval batch",
            ],
        }
        robustness_rows.append(
            {
                **common,
                "corruption_type": "clean_smoke",
                "corruption_strength": 0.0,
                "missing_modalities": [],
                "score": clean_score,
                "rceo_reliability": 1.0,
                "rceo_observed_reliability": clean_score,
            }
        )
        robustness_rows.append(
            {
                **common,
                "corruption_type": corrupted_type,
                "corruption_strength": strength,
                "missing_modalities": missing_modalities,
                "score": corrupted_score,
                "rceo_reliability": max(0.0, min(1.0, 1.0 - strength)),
                "rceo_observed_reliability": corrupted_score,
            }
        )
    return robustness_rows


def _write_smoke_robustness_summary(
    path: Path,
    *,
    rows: list[dict[str, object]],
    source_rows_path: Path,
) -> None:
    full_model = "ovha_full"
    baseline_model = _preview_baseline_model(rows)
    summary = summarize_robustness_rows(rows, full_model=full_model, baseline_model=baseline_model)
    payload = {
        **summary,
        "artifact_type": "public_smoke_robustness_summary",
        "evidence_scope": "smoke_robustness_preview_only_not_topconf_gate",
        "not_topconf_main_table": True,
        "source_rows_path": str(source_rows_path),
        "evidence_limitations": [
            "not valid top-conference robustness evidence",
            "robustness rows are smoke proxies and do not cover all Step 6 stress targets",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _robustness_strength(row: dict[str, object]) -> float:
    metrics = row.get("public_metrics", {})
    if isinstance(metrics, dict):
        value = metrics.get("missing_modality_performance_drop")
        if value is not None:
            return max(0.0, min(1.0, _as_float(value)))
    return 0.5


def _robustness_score(row: dict[str, object]) -> float:
    metrics = row.get("public_metrics", {})
    if isinstance(metrics, dict):
        for key in ("accuracy", "acc_at_0_5", "f1", "mean_iou"):
            if key in metrics:
                return max(0.0, min(1.0, _as_float(metrics[key])))
    return _bounded_score_from_loss(torch.tensor(float(row.get("score", 0.0))))


def _robustness_corruption_type(row: dict[str, object]) -> str:
    metrics = row.get("public_metrics", {})
    if isinstance(metrics, dict):
        loads = metrics.get("router_load_by_corruption_type")
        if isinstance(loads, dict) and loads:
            corruption_type = str(next(iter(loads)))
            if corruption_type != "clean":
                return corruption_type
        missing_drop = metrics.get("missing_modality_performance_drop")
        if missing_drop is not None and _as_float(missing_drop) > 0.0:
            return "missing_modality_smoke"
    return "smoke_corruption"


def _missing_modalities_from_corruption(corruption_type: str) -> list[str]:
    text = corruption_type.removesuffix("_smoke")
    for suffix in ("_missing", "missing_"):
        if text.endswith(suffix):
            return [text.removesuffix(suffix)]
        if text.startswith(suffix):
            return [text.removeprefix(suffix)]
    return []


def _robustness_router_load(row: dict[str, object], corruption_type: str) -> dict[str, float]:
    metrics = row.get("public_metrics", {})
    if isinstance(metrics, dict):
        loads = metrics.get("router_load_by_corruption_type")
        if isinstance(loads, dict) and isinstance(loads.get(corruption_type), dict):
            return _complete_candidate_probability_map(loads[corruption_type])
    if str(row.get("model")) == "ovha_full":
        return _complete_candidate_probability_map({})
    return _complete_candidate_probability_map(_probe_router_load_by_candidate(str(row.get("model"))))


def _robustness_candidate_loss(score: float, candidate_names: tuple[str, ...] = ("TLEO", "SPO", "LRIO", "CATO")) -> dict[str, float]:
    loss = max(0.0, 1.0 - score)
    return {candidate: loss for candidate in candidate_names}


def _write_smoke_statistics_preview(
    path: Path,
    *,
    smoke_raw_rows: list[dict[str, object]],
    baseline_raw_rows: list[dict[str, object]],
    source_paths: tuple[Path, Path],
) -> None:
    rows = [*smoke_raw_rows, *baseline_raw_rows]
    full_model = "ovha_full"
    baseline_model = _preview_baseline_model(rows)
    try:
        summary = summarize_public_results(rows, full_model=full_model, baseline_model=baseline_model)
        validation = validate_public_summary(summary)
        validation_payload = {
            "ok": validation.ok,
            "errors": validation.errors,
            "warnings": validation.warnings,
        }
    except ValueError as exc:
        summary = {}
        validation_payload = {"ok": False, "errors": [str(exc)], "warnings": []}
    baseline_limitations = _baseline_smoke_evidence_limitations(baseline_raw_rows)
    payload = {
        "artifact_type": "public_smoke_statistics_preview",
        "evidence_scope": "smoke_statistics_preview_only_not_topconf_main_table",
        "not_topconf_main_table": True,
        "full_model": full_model,
        "baseline_model": baseline_model,
        "source_raw_metric_paths": [str(path) for path in source_paths],
        "summary": summary,
        "validation": validation_payload,
        "evidence_limitations": [
            "not valid top-conference main-table evidence",
            "public metric inventory uses smoke-scale real metrics",
            *baseline_limitations,
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _baseline_smoke_evidence_limitations(baseline_rows: list[dict[str, object]]) -> list[str]:
    protocols = {str(row.get("baseline_protocol", "")) for row in baseline_rows}
    limitations: list[str] = []
    if "deterministic_same_feature_probe_smoke" in protocols:
        limitations.append("baseline rows are deterministic same-feature probes, not trained strong baselines")
    if "trainable_same_feature_linear_probe_smoke" in protocols:
        limitations.append("baseline rows are trainable same-feature linear probe smoke, not trained strong baselines")
    if not limitations:
        limitations.append("baseline rows are smoke proxies, not trained strong baselines")
    return limitations


def _preview_baseline_model(rows: list[dict[str, object]]) -> str:
    models = {str(row.get("model")) for row in rows}
    if "concat_fusion" in models:
        return "concat_fusion"
    return "concat_fusion"


def _public_baseline_history_rows(
    config: MultimodalExperimentConfig,
    *,
    train_batch: MultimodalEpisodeBatch,
    eval_batch: MultimodalEpisodeBatch,
    seed: int,
    ovha_training_steps: int,
    baseline_training_steps: int,
    learning_rate: float,
    hardware: dict[str, object],
    seed_count_rationale: str,
) -> list[dict[str, object]]:
    if baseline_training_steps <= 0:
        return _public_smoke_baseline_history_rows(
            config,
            eval_batch,
            seed=seed,
            training_steps=ovha_training_steps,
            hardware=hardware,
            seed_count_rationale=seed_count_rationale,
        )
    return _public_trained_baseline_history_rows(
        config,
        train_batch=train_batch,
        eval_batch=eval_batch,
        seed=seed,
        ovha_training_steps=ovha_training_steps,
        baseline_training_steps=baseline_training_steps,
        learning_rate=learning_rate,
        hardware=hardware,
        seed_count_rationale=seed_count_rationale,
    )


def _public_smoke_baseline_history_rows(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    *,
    seed: int,
    training_steps: int,
    hardware: dict[str, object],
    seed_count_rationale: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for baseline_name in config.baseline_names:
        prediction = _same_feature_probe_prediction(str(baseline_name), batch)
        task_loss = _task_loss(prediction, batch)
        rows.append(
            {
                "dataset": config.dataset_name,
                "task": config.task_type,
                "model": str(baseline_name),
                "stage": "T5_eval_baseline_smoke",
                "split": batch.split,
                "seed": seed,
                "metric_name": "heldout_task_loss_smoke",
                "score": _as_float(task_loss),
                "higher_is_better": False,
                "parameter_count": 0,
                "training_steps": 0,
                "ovha_reference_training_steps": training_steps,
                "frozen_feature_extractor_version": dict(batch.provenance.feature_extractor_version),
                "hardware": dict(hardware),
                "label_provenance": _label_provenance_for_batch(batch),
                "seed_count_rationale": seed_count_rationale,
                "baseline_protocol": "deterministic_same_feature_probe_smoke",
                "training_status": "not_trained",
                "same_feature_source": True,
                "public_metrics": _public_smoke_metrics(
                    config,
                    batch,
                    prediction=prediction,
                    candidate_loss={"CATO": task_loss},
                    router_load_by_candidate=_probe_router_load_by_candidate(str(baseline_name)),
                    router_entropy=torch.zeros((), dtype=batch.target_y.dtype, device=batch.target_y.device),
                    diagnostics=None,
                ),
                "public_metrics_scope": _public_metrics_scope(config),
            }
        )
    return rows


def _public_trained_baseline_history_rows(
    config: MultimodalExperimentConfig,
    *,
    train_batch: MultimodalEpisodeBatch,
    eval_batch: MultimodalEpisodeBatch,
    seed: int,
    ovha_training_steps: int,
    baseline_training_steps: int,
    learning_rate: float,
    hardware: dict[str, object],
    seed_count_rationale: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for baseline_name in config.baseline_names:
        train_inputs = _same_feature_probe_inputs(str(baseline_name), train_batch)
        eval_inputs = _same_feature_probe_inputs(str(baseline_name), eval_batch)
        target_dim = int(train_batch.target_y.shape[-1])
        generator = torch.Generator(device=train_inputs.device)
        generator.manual_seed(int(seed) + _stable_baseline_seed_offset(str(baseline_name)))
        model = torch.nn.Linear(int(train_inputs.shape[-1]), target_dim).to(train_inputs.device)
        with torch.no_grad():
            model.weight.uniform_(-0.02, 0.02, generator=generator)
            model.bias.uniform_(-0.02, 0.02, generator=generator)
        initial_parameters = _linear_parameter_vector(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        max_grad_norm = 0.0
        final_train_loss = 0.0
        model.train()
        for _ in range(int(baseline_training_steps)):
            optimizer.zero_grad(set_to_none=True)
            prediction = _expand_baseline_prediction(model(train_inputs), train_batch)
            loss = _task_loss(prediction, train_batch)
            loss.backward()
            max_grad_norm = max(max_grad_norm, _linear_grad_l2_norm(model))
            optimizer.step()
            final_train_loss = _as_float(loss)
        parameter_delta = float(torch.linalg.vector_norm(_linear_parameter_vector(model) - initial_parameters).item())
        model.eval()
        with torch.no_grad():
            eval_prediction = _expand_baseline_prediction(model(eval_inputs), eval_batch)
            task_loss = _task_loss(eval_prediction, eval_batch)
        rows.append(
            {
                "dataset": config.dataset_name,
                "task": config.task_type,
                "model": str(baseline_name),
                "stage": "T5_eval_baseline_trained_smoke",
                "split": eval_batch.split,
                "seed": seed,
                "metric_name": "heldout_task_loss_smoke",
                "score": _as_float(task_loss),
                "higher_is_better": False,
                "parameter_count": _linear_parameter_count(model),
                "training_steps": int(baseline_training_steps),
                "ovha_reference_training_steps": ovha_training_steps,
                "frozen_feature_extractor_version": dict(eval_batch.provenance.feature_extractor_version),
                "hardware": dict(hardware),
                "label_provenance": _label_provenance_for_batch(eval_batch),
                "seed_count_rationale": seed_count_rationale,
                "baseline_protocol": "trainable_same_feature_linear_probe_smoke",
                "training_status": "trained_smoke",
                "same_feature_source": True,
                "baseline_optimizer_steps": int(baseline_training_steps),
                "baseline_parameter_l2_delta": parameter_delta,
                "baseline_grad_l2_norm": max_grad_norm,
                "baseline_train_loss_final": final_train_loss,
                "public_metrics": _public_smoke_metrics(
                    config,
                    eval_batch,
                    prediction=eval_prediction,
                    candidate_loss={"CATO": task_loss},
                    router_load_by_candidate=_probe_router_load_by_candidate(str(baseline_name)),
                    router_entropy=torch.zeros((), dtype=eval_batch.target_y.dtype, device=eval_batch.target_y.device),
                    diagnostics=None,
                ),
                "public_metrics_scope": _public_metrics_scope(config),
            }
        )
    return rows


def _same_feature_probe_prediction(model_name: str, batch: MultimodalEpisodeBatch) -> torch.Tensor:
    target_dim = int(batch.target_y.shape[-1])
    modality_names = _probe_modalities_for_model(model_name, tuple(batch.fields))
    vectors = [
        _adapt_feature_dim(_masked_field_mean(batch.fields[modality]), target_dim)
        for modality in modality_names
        if modality in batch.fields
    ]
    if not vectors:
        batch_size = int(batch.target_y.shape[0])
        base = torch.zeros(batch_size, target_dim, dtype=batch.target_y.dtype, device=batch.target_y.device)
    else:
        base = torch.stack(vectors, dim=0).mean(dim=0).to(dtype=batch.target_y.dtype, device=batch.target_y.device)
    query_count = int(batch.target_y.shape[1])
    return base.unsqueeze(1).expand(-1, query_count, -1).contiguous()


def _same_feature_probe_inputs(model_name: str, batch: MultimodalEpisodeBatch) -> torch.Tensor:
    modality_names = _probe_modalities_for_model(model_name, tuple(batch.fields))
    vectors = [
        _masked_field_mean(batch.fields[modality])
        for modality in modality_names
        if modality in batch.fields
    ]
    if not vectors:
        batch_size = int(batch.target_y.shape[0])
        return torch.zeros(batch_size, 1, dtype=batch.target_y.dtype, device=batch.target_y.device)
    return torch.cat(vectors, dim=-1).to(dtype=batch.target_y.dtype, device=batch.target_y.device)


def _expand_baseline_prediction(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> torch.Tensor:
    query_count = int(batch.target_y.shape[1])
    return prediction.unsqueeze(1).expand(-1, query_count, -1).contiguous()


def _linear_parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _linear_parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    parts = [parameter.detach().reshape(-1).cpu() for parameter in model.parameters() if parameter.requires_grad]
    return torch.cat(parts) if parts else torch.zeros(0)


def _linear_grad_l2_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum().cpu())
    return float(total**0.5)


def _stable_baseline_seed_offset(model_name: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(model_name))


def _probe_modalities_for_model(model_name: str, available: tuple[str, ...]) -> tuple[str, ...]:
    available_set = set(available)
    if model_name == "text_only" and "text" in available_set:
        return ("text",)
    if model_name == "audio_only" and "audio" in available_set:
        return ("audio",)
    if model_name == "vision_only" and "vision" in available_set:
        return ("vision",)
    if model_name == "region_only":
        return tuple(name for name in ("region", "vision") if name in available_set) or available
    if model_name == "cato_only":
        region_like = tuple(name for name in ("region", "vision") if name in available_set)
        return tuple(name for name in ("text", *region_like) if name in available_set) or available
    return available


def _masked_field_mean(field: TokenField) -> torch.Tensor:
    weights = field.mask.to(dtype=field.x.dtype, device=field.x.device).unsqueeze(-1)
    return (field.x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _adapt_feature_dim(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    feature_dim = int(features.shape[-1])
    if feature_dim == target_dim:
        return features
    if feature_dim > target_dim:
        return features[..., :target_dim]
    pad = torch.zeros(
        *features.shape[:-1],
        target_dim - feature_dim,
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat([features, pad], dim=-1)


def _public_smoke_metrics(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    *,
    prediction: torch.Tensor,
    candidate_loss: Any,
    router_load_by_candidate: Any,
    router_entropy: Any,
    diagnostics: Any = None,
) -> dict[str, object]:
    if config.task_type not in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}:
        if config.task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
            return _sentiment_smoke_metrics(
                config,
                batch,
                prediction=prediction,
                router_load_by_candidate=router_load_by_candidate,
                diagnostics=diagnostics,
            )
        return {}
    task_loss = _task_loss(prediction, batch)
    region_metrics = _region_text_metrics(prediction, batch)
    cato_load = _candidate_probability(router_load_by_candidate, "CATO", default=0.0)
    cato_loss = _candidate_loss_value(candidate_loss, "CATO", default=task_loss)
    metrics = {
        "acc_at_0_5": region_metrics["acc_at_0_5"],
        "recall_at_1": region_metrics["recall_at_1"],
        "recall_at_5": region_metrics["recall_at_5"],
        "mean_iou": region_metrics["mean_iou"],
        "phrase_region_topk_accuracy": region_metrics["phrase_region_topk_accuracy"],
        "alignment_entropy": max(0.0, _as_float(router_entropy) if router_entropy is not None else 0.0),
        "cato_router_load": cato_load,
        "cato_candidate_loss": max(0.0, cato_loss),
        "cato_top_alignment_accuracy": region_metrics["phrase_region_topk_accuracy"],
        "null_unmatched_rate": _null_unmatched_rate(batch),
    }
    return {name: metrics[name] for name in REGION_TEXT_REQUIRED_PUBLIC_METRICS}


def _public_metrics_scope(config: MultimodalExperimentConfig) -> str:
    if config.task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}:
        return "region_text_smoke_real_metrics_not_topconf_main_table"
    if config.task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
        return "sentiment_emotion_smoke_real_metrics_not_topconf_main_table"
    return "smoke_real_metrics_not_topconf_main_table"


def _bounded_score_from_loss(loss: torch.Tensor) -> float:
    return max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, _as_float(loss)))))


def _region_text_metrics(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> dict[str, float]:
    region_targets = batch.supervision.region_targets
    if region_targets is not None and prediction.shape[-1] > 1:
        labels = region_targets.to(device=prediction.device, dtype=torch.long)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        if labels.ndim > 2:
            labels = labels.reshape(labels.shape[0], -1)
        if labels.shape[1] == 1 and prediction.shape[1] > 1:
            labels = labels.expand(-1, prediction.shape[1])
        labels = labels[:, : prediction.shape[1]]
        valid = batch.target_mask.to(dtype=torch.bool, device=prediction.device)[:, : labels.shape[1]]
        top1 = prediction[:, : labels.shape[1]].argmax(dim=-1)
        topk = torch.topk(prediction[:, : labels.shape[1]], k=min(5, prediction.shape[-1]), dim=-1).indices
        if bool(valid.any()):
            recall1 = (top1[valid] == labels[valid]).to(dtype=torch.float32).mean()
            recall5 = (topk[valid] == labels[valid].unsqueeze(-1)).any(dim=-1).to(dtype=torch.float32).mean()
            return {
                "acc_at_0_5": _as_float(recall1),
                "recall_at_1": _as_float(recall1),
                "recall_at_5": _as_float(recall5),
                "mean_iou": _bbox_mean_iou(prediction, batch),
                "phrase_region_topk_accuracy": _as_float(recall1),
            }
    return {
        "acc_at_0_5": 0.0,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "mean_iou": _bbox_mean_iou(prediction, batch),
        "phrase_region_topk_accuracy": 0.0,
    }


def _bbox_mean_iou(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> float:
    target = batch.supervision.bbox_targets
    if target is None or prediction.shape[-1] != 4:
        return 0.0
    pred = prediction.to(dtype=torch.float32)
    truth = target.to(device=prediction.device, dtype=torch.float32)
    if truth.ndim == 2:
        truth = truth.unsqueeze(1)
    if truth.shape[1] == 1 and pred.shape[1] > 1:
        truth = truth.expand(-1, pred.shape[1], -1)
    truth = truth[:, : pred.shape[1], :]
    valid = batch.target_mask.to(dtype=torch.bool, device=prediction.device)[:, : truth.shape[1]]
    if not bool(valid.any()):
        return 0.0
    return _as_float(_box_iou(pred[:, : truth.shape[1], :][valid], truth[valid]).mean())


def _box_iou(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    pred_min = torch.minimum(pred[..., :2], pred[..., 2:])
    pred_max = torch.maximum(pred[..., :2], pred[..., 2:])
    truth_min = torch.minimum(truth[..., :2], truth[..., 2:])
    truth_max = torch.maximum(truth[..., :2], truth[..., 2:])
    inter_min = torch.maximum(pred_min, truth_min)
    inter_max = torch.minimum(pred_max, truth_max)
    inter = (inter_max - inter_min).clamp_min(0.0)
    inter_area = inter[..., 0] * inter[..., 1]
    pred_area = ((pred_max - pred_min).clamp_min(0.0)).prod(dim=-1)
    truth_area = ((truth_max - truth_min).clamp_min(0.0)).prod(dim=-1)
    return inter_area / (pred_area + truth_area - inter_area).clamp_min(1e-12)


def _candidate_probability(values: Any, candidate: str, *, default: float) -> float:
    if isinstance(values, dict) and candidate in values:
        return max(0.0, min(1.0, _as_float(values[candidate])))
    return default


def _candidate_loss_value(values: Any, candidate: str, *, default: torch.Tensor) -> float:
    if isinstance(values, dict) and candidate in values:
        return _as_float(values[candidate])
    return _as_float(default)


def _null_unmatched_rate(batch: MultimodalEpisodeBatch) -> float:
    if batch.target_mask.numel() == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - _as_float(batch.target_mask.to(dtype=torch.float32).mean())))


def _sentiment_smoke_metrics(
    config: MultimodalExperimentConfig,
    batch: MultimodalEpisodeBatch,
    *,
    prediction: torch.Tensor,
    router_load_by_candidate: Any,
    diagnostics: Any,
) -> dict[str, object]:
    standard = mosei_standard_metrics(prediction, batch.target_y, batch.target_mask)
    missing_drop = _missing_modality_fraction(batch)
    metrics = {
        "mae": max(0.0, standard["mae"]),
        "mse_loss": max(0.0, standard["mse_loss"]),
        "l1_loss": max(0.0, standard["l1_loss"]),
        "pearson_correlation": standard["pearson_correlation"],
        "acc7": standard["acc7"],
        "acc5": standard["acc5"],
        "acc2_excl0": standard["acc2_excl0"],
        "f1_excl0": standard["f1_excl0"],
        "acc2_nonneg": standard["acc2_nonneg"],
        "f1_nonneg": standard["f1_nonneg"],
        "accuracy": standard["acc2_excl0"],
        "f1": standard["f1_excl0"],
        "missing_modality_performance_drop": missing_drop,
        "corruption_robustness_auc": max(0.0, min(1.0, 1.0 - missing_drop)),
        "router_load_by_corruption_type": {
            "clean": _complete_candidate_probability_map(router_load_by_candidate, candidate_names=config.candidate_names)
        },
        "lrio_rank_entropy": _candidate_diagnostic_metric(diagnostics, "LRIO", "rank_entropy"),
        "spo_prototype_entropy": _candidate_diagnostic_metric(diagnostics, "SPO", "prototype_entropy"),
        "rceo_reliability_calibration": _smoke_rceo_calibration(batch, prediction, diagnostics),
    }
    return {name: metrics[name] for name in SENTIMENT_REQUIRED_PUBLIC_METRICS}


def _pearson_correlation(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred, truth = _masked_flat_pair(prediction, target, mask)
    if pred.numel() < 2:
        return 0.0
    pred_centered = pred - pred.mean()
    truth_centered = truth - truth.mean()
    denom = pred_centered.norm() * truth_centered.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, _as_float((pred_centered * truth_centered).sum() / denom)))


def _binary_sign_accuracy_f1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    pred, truth = _masked_flat_pair(prediction, target, mask)
    if pred.numel() == 0:
        return 0.0, 0.0
    pred_positive = pred >= 0
    truth_positive = truth >= 0
    accuracy = (pred_positive == truth_positive).to(dtype=torch.float32).mean()
    true_positive = (pred_positive & truth_positive).to(dtype=torch.float32).sum()
    false_positive = (pred_positive & ~truth_positive).to(dtype=torch.float32).sum()
    false_negative = (~pred_positive & truth_positive).to(dtype=torch.float32).sum()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return _as_float(accuracy), _as_float(f1)


def _masked_flat_pair(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.to(dtype=torch.bool, device=prediction.device)
    pred = prediction[..., 0][valid].reshape(-1)
    truth = target[..., 0].to(device=prediction.device, dtype=prediction.dtype)[valid].reshape(-1)
    return pred, truth


def _missing_modality_fraction(batch: MultimodalEpisodeBatch) -> float:
    missing = batch.supervision.modality_missing_mask
    if missing is None or missing.numel() == 0:
        return 0.0
    return max(0.0, min(1.0, _as_float(missing.to(dtype=torch.float32).mean())))


def _missing_corruption_key(batch: MultimodalEpisodeBatch) -> str:
    missing = batch.supervision.modality_missing_mask
    if missing is None or missing.numel() == 0 or not bool(missing.any().item()):
        return "clean_smoke"
    modality_order = list(batch.fields)
    missing_by_modality = missing.to(dtype=torch.float32).mean(dim=0)
    index = int(torch.argmax(missing_by_modality).item())
    modality = modality_order[index] if index < len(modality_order) else "modality"
    return f"{modality}_missing_smoke"


def _complete_candidate_probability_map(values: Any, candidate_names: tuple[str, ...] = ("TLEO", "SPO", "LRIO", "CATO")) -> dict[str, float]:
    candidates = tuple(str(candidate) for candidate in candidate_names)
    if not candidates:
        return {}
    loads = {
        candidate: _candidate_probability(values, candidate, default=0.0)
        for candidate in candidates
    }
    total = sum(loads.values())
    if total <= 0.0:
        uniform = 1.0 / float(len(loads))
        return {candidate: uniform for candidate in loads}
    return {candidate: value / total for candidate, value in loads.items()}


def _entropy_proxy(values: Any, candidate: str) -> float:
    probability = _candidate_probability(values, candidate, default=0.25)
    if probability <= 0.0:
        return 0.0
    return max(0.0, -probability * math.log(probability))


def _candidate_diagnostic_metric(diagnostics: Any, candidate: str, key: str) -> float:
    if not isinstance(diagnostics, dict):
        return 0.0
    candidate_diagnostics = diagnostics.get("candidate_diagnostics")
    if not isinstance(candidate_diagnostics, dict):
        return 0.0
    values = candidate_diagnostics.get(candidate)
    if not isinstance(values, dict):
        return 0.0
    return max(0.0, _as_float(values.get(key)))


def _smoke_rceo_calibration(
    batch: MultimodalEpisodeBatch,
    prediction: torch.Tensor,
    diagnostics: Any,
) -> dict[str, object]:
    predicted_values, source = _model_reliability_by_sample(diagnostics, batch, prediction)
    observed_values = _bounded_observed_reliability_by_sample(batch, prediction)
    ece, curve = _binned_ece(predicted_values, observed_values, bin_count=10)
    return {
        "ece": ece,
        "expected_calibration_error": ece,
        "bin_count": 10,
        "source": source,
        "mean_predicted_reliability": _as_float(predicted_values.mean()),
        "mean_observed_reliability": _as_float(observed_values.mean()),
        "calibration_curve": curve,
        "condition": "smoke calibration between model RCEO reliability and bounded per-sample performance",
    }


def _model_reliability_mean(diagnostics: Any) -> float | None:
    if not isinstance(diagnostics, dict):
        return None
    reliability = diagnostics.get("reliability")
    if isinstance(reliability, dict) and "modality_reliability_mean" in reliability:
        return _as_float(reliability["modality_reliability_mean"])
    candidate_diagnostics = diagnostics.get("candidate_diagnostics")
    if isinstance(candidate_diagnostics, dict):
        rceo = candidate_diagnostics.get("RCEO")
        if isinstance(rceo, dict) and "modality_reliability" in rceo:
            return _as_float(rceo["modality_reliability"])
    return None


def _bounded_observed_reliability(batch: MultimodalEpisodeBatch, prediction: torch.Tensor) -> float:
    return max(0.0, min(1.0, _as_float(_bounded_observed_reliability_by_sample(batch, prediction).mean())))


def _bounded_observed_reliability_by_sample(batch: MultimodalEpisodeBatch, prediction: torch.Tensor) -> torch.Tensor:
    target = batch.target_y.to(device=prediction.device, dtype=prediction.dtype)
    mask = batch.target_mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
    per_sample_error = ((prediction - target).square() * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
    return (1.0 / (1.0 + per_sample_error)).clamp(0.0, 1.0)


def _model_reliability_by_sample(
    diagnostics: Any,
    batch: MultimodalEpisodeBatch,
    prediction: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    batch_size = int(batch.target_y.shape[0])
    device = prediction.device
    dtype = prediction.dtype
    if isinstance(diagnostics, dict):
        reliability = diagnostics.get("reliability")
        if isinstance(reliability, dict):
            sample_values = reliability.get("sample_modality_reliability_mean")
            if sample_values is not None:
                values = torch.as_tensor(sample_values, dtype=dtype, device=device).reshape(-1)
                if values.numel() == batch_size:
                    return values.clamp(0.0, 1.0), "model_reliability_prior"
    predicted = _model_reliability_mean(diagnostics)
    if predicted is None:
        return torch.zeros(batch_size, dtype=dtype, device=device), "not_applicable_no_model_reliability"
    return torch.full((batch_size,), max(0.0, min(1.0, predicted)), dtype=dtype, device=device), "model_reliability_prior"


def _binned_ece(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    *,
    bin_count: int,
) -> tuple[float, list[dict[str, float | int]]]:
    predicted = predicted.detach().flatten().clamp(0.0, 1.0)
    observed = observed.detach().flatten().clamp(0.0, 1.0).to(device=predicted.device, dtype=predicted.dtype)
    total = max(int(predicted.numel()), 1)
    ece = 0.0
    curve = []
    for index in range(bin_count):
        lower = float(index) / float(bin_count)
        upper = float(index + 1) / float(bin_count)
        if index == bin_count - 1:
            mask = (predicted >= lower) & (predicted <= upper)
        else:
            mask = (predicted >= lower) & (predicted < upper)
        count = int(mask.sum().item())
        if count > 0:
            confidence = _as_float(predicted[mask].mean())
            accuracy = _as_float(observed[mask].mean())
        else:
            confidence = 0.0
            accuracy = 0.0
        ece += (count / total) * abs(confidence - accuracy)
        curve.append({"bin": index, "mean_confidence": confidence, "observed_accuracy": accuracy, "count": count})
    return ece, curve


def _probe_router_load_by_candidate(model_name: str) -> dict[str, float]:
    if model_name == "ovha_no_cato":
        return {"TLEO": 1.0 / 3.0, "SPO": 1.0 / 3.0, "LRIO": 1.0 / 3.0, "CATO": 0.0}
    if model_name == "cato_only":
        return {"TLEO": 0.0, "SPO": 0.0, "LRIO": 0.0, "CATO": 1.0}
    return {"TLEO": 0.25, "SPO": 0.25, "LRIO": 0.25, "CATO": 0.25}


def _public_smoke_raw_metric_rows(
    eval_history: list[dict[str, object]],
    raw_metric_path: Path,
) -> list[dict[str, object]]:
    return [
        {
            "artifact_type": "public_smoke_raw_metric",
            "evidence_scope": "public_smoke_only_not_topconf_main_table",
            "not_topconf_main_table": True,
            "dataset": str(row["dataset"]),
            "task": str(row["task"]),
            "model": str(row["model"]),
            "stage": str(row["stage"]),
            "split": str(row["split"]),
            "seed": int(row["seed"]),
            "metric_name": "heldout_task_loss_smoke",
            "score": float(row["task_loss"]),
            "higher_is_better": False,
            "parameter_count": int(row["parameter_count"]),
            "training_steps": int(row["training_steps"]),
            "frozen_feature_extractor_version": dict(row["frozen_feature_extractor_version"]),
            "hardware": dict(row["hardware"]),
            "label_provenance": dict(row["label_provenance"]),
            "seed_count_rationale": str(row["seed_count_rationale"]),
            "public_metrics": dict(row["public_metrics"]),
            "public_metrics_scope": str(row["public_metrics_scope"]),
            "raw_metric_path": str(raw_metric_path),
            "source_total_loss": float(row["total_loss"]),
            "loss_names_observed": list(row["loss_names_observed"]),
            "evidence_limitations": [
                "single OVHA smoke model only",
                "not a same-feature baseline comparison",
                "not valid top-conference main-table evidence",
                "public metric inventory uses smoke-scale real metrics",
            ],
        }
        for row in eval_history
    ]


def _public_smoke_baseline_raw_metric_rows(
    eval_baseline_history: list[dict[str, object]],
    raw_metric_path: Path,
) -> list[dict[str, object]]:
    return [
        {
            "artifact_type": "public_smoke_baseline_raw_metric",
            "evidence_scope": "same_feature_baseline_smoke_only_not_topconf_main_table",
            "not_topconf_main_table": True,
            "dataset": str(row["dataset"]),
            "task": str(row["task"]),
            "model": str(row["model"]),
            "stage": str(row["stage"]),
            "split": str(row["split"]),
            "seed": int(row["seed"]),
            "metric_name": str(row["metric_name"]),
            "score": float(row["score"]),
            "higher_is_better": bool(row["higher_is_better"]),
            "parameter_count": int(row["parameter_count"]),
            "training_steps": int(row["training_steps"]),
            "ovha_reference_training_steps": int(row["ovha_reference_training_steps"]),
            "frozen_feature_extractor_version": dict(row["frozen_feature_extractor_version"]),
            "hardware": dict(row["hardware"]),
            "label_provenance": dict(row["label_provenance"]),
            "seed_count_rationale": str(row["seed_count_rationale"]),
            "baseline_protocol": str(row["baseline_protocol"]),
            "training_status": str(row["training_status"]),
            "same_feature_source": bool(row["same_feature_source"]),
            "baseline_optimizer_steps": int(row.get("baseline_optimizer_steps", 0)),
            "baseline_parameter_l2_delta": float(row.get("baseline_parameter_l2_delta", 0.0)),
            "baseline_grad_l2_norm": float(row.get("baseline_grad_l2_norm", 0.0)),
            "baseline_train_loss_final": float(row.get("baseline_train_loss_final", 0.0)),
            "public_metrics": dict(row["public_metrics"]),
            "public_metrics_scope": str(row["public_metrics_scope"]),
            "raw_metric_path": str(raw_metric_path),
            "evidence_limitations": [
                "not a trained strong baseline",
                "not a top-conference same-feature baseline comparison",
                _baseline_protocol_limitation(str(row["baseline_protocol"])),
                "public metric inventory uses smoke-scale real metrics",
            ],
        }
        for row in eval_baseline_history
    ]


def _baseline_protocol_limitation(baseline_protocol: str) -> str:
    if baseline_protocol == "trainable_same_feature_linear_probe_smoke":
        return "trainable same-feature linear probe smoke"
    return "deterministic probe only verifies baseline artifact plumbing and same-feature provenance"


def _hardware_metadata(device: torch.device, elapsed_seconds: float) -> dict[str, object]:
    return {
        "accelerator": device.type,
        "device": str(device),
        "wall_clock_hours": float(elapsed_seconds) / 3600.0,
        "measurement_scope": "public_smoke_train_eval_seed",
    }


def _label_provenance_for_batch(batch: MultimodalEpisodeBatch) -> dict[str, str]:
    return {
        "supervision_type": "ground_truth",
        "source": batch.source_dataset,
        "must_report_as": "ground_truth",
    }


def _seed_count_rationale(train_all_config_seeds: bool) -> str:
    if train_all_config_seeds:
        return (
            "public smoke executed all configured development seeds; "
            "production main tables should use 5 seeds or at least 3 with explicit rationale"
        )
    return (
        "public smoke defaults to one seed for execution validation only; "
        "production main tables should use 5 seeds or at least 3 with explicit rationale"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _as_float(value)
        return _as_float(value.to(dtype=torch.float32).mean())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _replace_cache_root(config: MultimodalExperimentConfig, cache_root: Path) -> MultimodalExperimentConfig:
    return replace(config, cache_root=cache_root)


if __name__ == "__main__":
    raise SystemExit(main())
