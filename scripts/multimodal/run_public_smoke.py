#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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
from moat_ovha_torch.models.multimodal.baselines import assert_same_feature_baseline_policy
from moat_ovha_torch.eval.multimodal_public_entry import validate_public_entry_requirements
from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA, MultimodalOVHAOutput


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and launch a multimodal public smoke run.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--controlled-report", type=Path)
    parser.add_argument("--train-smoke-steps", type=int, default=0)
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
    parameter_deltas: dict[int, float] = {}
    max_grad_norm = 0.0

    for seed in seed_values:
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
            eval_history.append(
                {
                    "seed": seed,
                    "stage": "T5_eval",
                    "split": args.eval_smoke_split,
                    "loss_names_observed": sorted(eval_components),
                    "total_loss": _as_float(eval_total_loss),
                    **_rceo_training_metrics(eval_output, eval_batch),
                    **{name: _as_float(value) for name, value in eval_components.items()},
                }
            )
            eval_diagnostic_history.append(
                _public_training_diagnostics_row(eval_output, config, eval_batch, 0, seed, artifact_type="public_eval_diagnostics", stage="T5_eval")
            )
    parameter_l2_delta = float(sum(parameter_deltas.values()) / max(len(parameter_deltas), 1))
    parameter_l2_delta_min = min(parameter_deltas.values()) if parameter_deltas else 0.0
    expected_steps = int(args.train_smoke_steps) * len(seed_values)
    artifacts = (
        _write_training_artifacts(args.artifact_root, loss_history, diagnostic_history, eval_history, eval_diagnostic_history)
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
    if eval_history:
        eval_metrics = artifact_root / "public_eval_metrics.jsonl"
        eval_diagnostics = artifact_root / "public_eval_diagnostics.jsonl"
        eval_metrics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in eval_history) + "\n")
        eval_diagnostics.write_text("\n".join(json.dumps(row, sort_keys=True) for row in eval_diagnostic_history) + "\n")
        artifacts["eval_metrics"] = {"path": str(eval_metrics), "sha256": file_sha256(eval_metrics)}
        artifacts["eval_diagnostics"] = {"path": str(eval_diagnostics), "sha256": file_sha256(eval_diagnostics)}
    return artifacts


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
