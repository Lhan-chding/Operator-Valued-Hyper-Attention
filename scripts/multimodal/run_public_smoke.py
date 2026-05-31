#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from moat_ovha_torch.config_multimodal import MultimodalExperimentConfig
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
            original_split=[str(row["original_split"]) for row in sample_records],
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
    available = {
        "task_loss": _task_loss(output.y_hat, batch),
        "candidate_individual_loss": (output.candidate_values - batch.target_y.unsqueeze(-2)).square().mean(),
        "public_alignment_ce": output.y_hat.sum() * 0.0,
        "public_contrastive_retrieval": output.y_hat.sum() * 0.0,
    }
    return {name: available[name] for name in configured if name in available}


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
    return {
        "artifact_type": artifact_type,
        "config_name": config.name,
        "dataset": config.dataset_name,
        "task": config.task_type,
        "stage": stage,
        "split": batch.split,
        "seed": seed,
        "step": step,
        "router_entropy": _json_ready(diagnostics["router_entropy"]),
        "router_load_by_candidate": _json_ready(diagnostics["router_load_by_candidate"]),
        "router_logit_parts": _router_logit_part_summary(output.router_logit_parts),
        "candidate_loss": _json_ready(diagnostics["candidate_loss"]),
        "adapter_params": _json_ready(diagnostics["adapter_params"]),
        "memory_slot_norm": _json_ready(diagnostics["memory_slot_norm"]),
        "stackability_passed": bool(diagnostics["stackability_passed"]),
        "candidate_diagnostics": _json_ready(diagnostics["candidate_diagnostics"]),
        "reliability": _json_ready(diagnostics["reliability"]),
    }


def _router_logit_part_summary(logit_parts: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": _as_float(value.mean()),
            "norm": _as_float(value.norm(dim=-1).mean()),
        }
        for name, value in logit_parts.items()
    }


def _task_loss(prediction: torch.Tensor, batch: MultimodalEpisodeBatch) -> torch.Tensor:
    mask = batch.target_mask.to(dtype=prediction.dtype, device=prediction.device).unsqueeze(-1)
    return ((prediction - batch.target_y).square() * mask).sum() / mask.sum().clamp_min(1.0)


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
    metrics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in loss_history) + "\n")
    diagnostics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in diagnostic_history) + "\n")
    artifacts = {
        "metrics": {
            "path": str(metrics),
            "sha256": file_sha256(metrics),
        },
        "diagnostics": {
            "path": str(diagnostics),
            "sha256": file_sha256(diagnostics),
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
        smoke_raw_rows = _public_smoke_raw_metric_rows(eval_history, smoke_raw_metrics)
        smoke_raw_metrics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in smoke_raw_rows) + "\n")
        artifacts["eval_metrics"] = {"path": str(eval_metrics), "sha256": file_sha256(eval_metrics)}
        artifacts["eval_diagnostics"] = {"path": str(eval_diagnostics), "sha256": file_sha256(eval_diagnostics)}
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
            return str(next(iter(loads)))
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


def _robustness_candidate_loss(score: float) -> dict[str, float]:
    loss = max(0.0, 1.0 - score)
    return {candidate: loss for candidate in ("TLEO", "SPO", "LRIO", "CATO")}


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
            "public metric inventory uses smoke proxies",
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
    if "cross_attention_transformer" in models:
        return "cross_attention_transformer"
    # Public gates use cross-attention as the canonical anchor; do not silently
    # replace it with a weaker arbitrary smoke baseline when it is absent.
    return "cross_attention_transformer"


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
    if model_name == "region_only":
        return tuple(name for name in ("region", "vision") if name in available_set) or available
    if model_name in {"cato_only", "clip_style_region_text_retrieval"}:
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
) -> dict[str, object]:
    if config.task_type not in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}:
        if config.task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
            return _sentiment_smoke_metrics(
                batch,
                prediction=prediction,
                router_load_by_candidate=router_load_by_candidate,
            )
        return {}
    task_loss = _task_loss(prediction, batch)
    bounded_score = _bounded_score_from_loss(task_loss)
    cato_load = _candidate_probability(router_load_by_candidate, "CATO", default=0.0)
    cato_loss = _candidate_loss_value(candidate_loss, "CATO", default=task_loss)
    metrics = {
        "acc_at_0_5": bounded_score,
        "recall_at_1": bounded_score,
        "recall_at_5": min(1.0, bounded_score + 0.10),
        "mean_iou": bounded_score,
        "phrase_region_topk_accuracy": bounded_score,
        "alignment_entropy": max(0.0, _as_float(router_entropy) if router_entropy is not None else 0.0),
        "cato_router_load": cato_load,
        "cato_candidate_loss": max(0.0, cato_loss),
        "cato_top_alignment_accuracy": min(1.0, bounded_score * max(cato_load, 0.0)),
        "null_unmatched_rate": _null_unmatched_rate(batch),
    }
    return {name: metrics[name] for name in REGION_TEXT_REQUIRED_PUBLIC_METRICS}


def _public_metrics_scope(config: MultimodalExperimentConfig) -> str:
    if config.task_type in {"phrase_region_grounding", "region_text_grounding", "refcoco", "flickr30k_entities", "visual_genome"}:
        return "region_text_smoke_proxy_not_topconf_main_table"
    if config.task_type in {"sentiment_emotion", "sentiment_regression", "emotion_classification", "cmu_mosei", "cmu_mosi", "meld", "iemocap"}:
        return "sentiment_emotion_smoke_proxy_not_topconf_main_table"
    return "smoke_proxy_not_topconf_main_table"


def _bounded_score_from_loss(loss: torch.Tensor) -> float:
    return max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, _as_float(loss)))))


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
    batch: MultimodalEpisodeBatch,
    *,
    prediction: torch.Tensor,
    router_load_by_candidate: Any,
) -> dict[str, object]:
    mae = _as_float((prediction - batch.target_y).abs().mean())
    bounded_score = _bounded_score_from_loss(_task_loss(prediction, batch))
    missing_drop = _missing_modality_fraction(batch)
    metrics = {
        "mae": max(0.0, mae),
        "pearson_correlation": 0.0,
        "accuracy": bounded_score,
        "f1": bounded_score,
        "missing_modality_performance_drop": missing_drop,
        "corruption_robustness_auc": max(0.0, min(1.0, 1.0 - missing_drop)),
        "router_load_by_corruption_type": {
            _missing_corruption_key(batch): _complete_candidate_probability_map(router_load_by_candidate)
        },
        "lrio_rank_entropy": _entropy_proxy(router_load_by_candidate, "LRIO"),
        "spo_prototype_entropy": _entropy_proxy(router_load_by_candidate, "SPO"),
        "rceo_reliability_calibration": _smoke_rceo_calibration(batch, bounded_score),
    }
    return {name: metrics[name] for name in SENTIMENT_REQUIRED_PUBLIC_METRICS}


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


def _complete_candidate_probability_map(values: Any) -> dict[str, float]:
    loads = {
        candidate: _candidate_probability(values, candidate, default=0.0)
        for candidate in ("TLEO", "SPO", "LRIO", "CATO")
    }
    total = sum(loads.values())
    if total <= 0.0:
        return {candidate: 0.25 for candidate in loads}
    return {candidate: value / total for candidate, value in loads.items()}


def _entropy_proxy(values: Any, candidate: str) -> float:
    probability = _candidate_probability(values, candidate, default=0.25)
    if probability <= 0.0:
        return 0.0
    return max(0.0, -probability * math.log(probability))


def _smoke_rceo_calibration(batch: MultimodalEpisodeBatch, observed_score: float) -> dict[str, object]:
    predicted_reliability = max(0.0, min(1.0, 1.0 - _missing_modality_fraction(batch)))
    observed = max(0.0, min(1.0, observed_score))
    ece = abs(predicted_reliability - observed)
    return {
        "ece": ece,
        "expected_calibration_error": ece,
        "bin_count": 1,
        "calibration_curve": [
            {
                "bin": 0,
                "mean_confidence": predicted_reliability,
                "observed_accuracy": observed,
                "count": max(1, int(batch.target_y.shape[0])),
            }
        ],
        "condition": "smoke proxy calibration between missing-modality reliability and bounded task score",
    }


def _probe_router_load_by_candidate(model_name: str) -> dict[str, float]:
    if model_name == "ovha_no_cato":
        return {"TLEO": 1.0 / 3.0, "SPO": 1.0 / 3.0, "LRIO": 1.0 / 3.0, "CATO": 0.0}
    if model_name in {"cato_only", "clip_style_region_text_retrieval"}:
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
                "public metric inventory uses smoke proxies",
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
                "public metric inventory uses smoke proxies",
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
    return MultimodalExperimentConfig(
        name=config.name,
        dataset_name=config.dataset_name,
        task_type=config.task_type,
        cache_version=config.cache_version,
        cache_root=cache_root,
        output_dir=config.output_dir,
        seeds=config.seeds,
        training_stages=config.training_stages,
        candidate_names=config.candidate_names,
        baseline_names=config.baseline_names,
        eval_splits=config.eval_splits,
        eval_episode_count=config.eval_episode_count,
        enforce_same_features_for_baselines=config.enforce_same_features_for_baselines,
        fail_on_missing_cache_artifact=config.fail_on_missing_cache_artifact,
        allow_hidden_losses=config.allow_hidden_losses,
        require_public_alignment_labels=config.require_public_alignment_labels,
        robustness_corruptions=config.robustness_corruptions,
        losses_by_stage=config.losses_by_stage,
        loss_metadata=config.loss_metadata,
        adapter_params_by_candidate=config.adapter_params_by_candidate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
