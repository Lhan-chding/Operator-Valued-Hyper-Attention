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
from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import (
    CONTROLLED_FAMILY_ACTIVE_OPERATOR,
    CONTROLLED_MULTIMODAL_FAMILIES,
    ControlledSyntheticMultimodalAdapter,
)
from moat_ovha_torch.data.multimodal.cache_schema import file_sha256
from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report
from moat_ovha_torch.eval.multimodal_oracle import controlled_row_from_oracle_report, evaluate_oracle_matrix
from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA, MultimodalOVHAOutput


CONTROLLED_SPECIALIST_WARMUP_FAMILIES = (
    "tleo_local_evidence",
    "spo_global_prototype",
    "lrio_low_rank_interaction",
    "cato_alignment_transport",
)
CONTROLLED_CANDIDATE_NAMES = ("TLEO", "SPO", "LRIO", "CATO")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real controlled multimodal OVHA training smoke.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--field-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--steps-per-stage", type=int)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--memory-tokens", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()

    config = MultimodalExperimentConfig.from_file(args.config) if args.config else None
    if config is not None and config.task_type not in {"controlled_multimodal", "controlled_relation_operator"}:
        raise ValueError("run_controlled_training_smoke.py only accepts controlled multimodal configs")
    seed = int(args.seed if args.seed is not None else (config.seeds[0] if config is not None else 7))
    torch.manual_seed(seed)
    device = torch.device(args.device)
    adapter = ControlledSyntheticMultimodalAdapter(
        seed=seed + 4,
        output_dim=args.output_dim,
        field_dim=args.field_dim,
    )
    model = MultimodalOVHA(
        field_dims={"text": args.field_dim, "region": args.field_dim, "audio": args.field_dim},
        query_dim=args.field_dim,
        output_dim=args.output_dim,
        d_model=args.d_model,
        memory_tokens=args.memory_tokens,
    ).to(device)
    initial_parameters = _parameter_vector(model)
    before = _evaluate_task_losses(model, adapter, args, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_result = _run_training(model, optimizer, adapter, args, device, config)

    after = _evaluate_task_losses(model, adapter, args, device)
    parameter_l2_delta = torch.linalg.vector_norm(_parameter_vector(model) - initial_parameters).item()
    rows = _controlled_rows(
        model,
        adapter,
        args,
        device,
        training_steps=int(train_result["optimizer_steps"]),
        config_name=config.name if config is not None else None,
    )
    evidence_artifacts = _write_evidence_artifacts(args.artifact_root, rows) if args.artifact_root else None
    mean_before = before["mean_task_loss"]
    mean_after = after["mean_task_loss"]
    ok = bool(parameter_l2_delta > 0.0 and train_result["max_grad_norm"] > 0.0 and mean_after < mean_before)
    payload = {
        "ok": ok,
        "mode": "trained_smoke",
        "training": {
            "config_name": config.name if config is not None else None,
            "validated_training_stages": list(config.training_stages) if config is not None else [],
            "optimizer": "AdamW",
            "optimizer_steps": train_result["optimizer_steps"],
            "learning_rate": args.learning_rate,
            "seed": seed,
            "mean_task_loss_before": mean_before,
            "mean_task_loss_after": mean_after,
            "task_loss_by_family_before": before["task_loss_by_family"],
            "task_loss_by_family_after": after["task_loss_by_family"],
            "parameter_l2_delta": float(parameter_l2_delta),
            "max_grad_norm": train_result["max_grad_norm"],
            "stage_history": train_result["stage_history"],
            "loss_history": train_result["loss_history"],
        },
        "rows": rows,
        "controlled_report": build_controlled_report(rows, evidence_artifacts=evidence_artifacts),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 2


def _run_training(
    model: MultimodalOVHA,
    optimizer: torch.optim.Optimizer,
    adapter: ControlledSyntheticMultimodalAdapter,
    args: argparse.Namespace,
    device: torch.device,
    config: MultimodalExperimentConfig | None,
) -> dict[str, object]:
    model.train()
    if config is None:
        return _run_flat_training(model, optimizer, adapter, args, device)
    return _run_configured_stage_training(model, optimizer, adapter, args, device, config)


def _run_flat_training(
    model: MultimodalOVHA,
    optimizer: torch.optim.Optimizer,
    adapter: ControlledSyntheticMultimodalAdapter,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    loss_history: list[dict[str, object]] = []
    max_grad_norm = 0.0
    for step in range(args.steps):
        family = CONTROLLED_MULTIMODAL_FAMILIES[step % len(CONTROLLED_MULTIMODAL_FAMILIES)]
        batch = _sample(adapter, family, args, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        components = _controlled_loss_components(output, batch)
        total_loss = components["task_loss"] + 0.1 * components["router_ce_true_active_operator"] + 0.1 * components["candidate_oracle_mse"]
        total_loss.backward()
        grad_norm = _grad_l2_norm(model)
        max_grad_norm = max(max_grad_norm, grad_norm)
        optimizer.step()
        loss_history.append(
            {
                "step": step + 1,
                "family": family,
                "task_loss": _as_float(components["task_loss"]),
                "router_ce_true_active_operator": _as_float(components["router_ce_true_active_operator"]),
                "candidate_oracle_mse": _as_float(components["candidate_oracle_mse"]),
                "total_loss": _as_float(total_loss),
                "grad_l2_norm": grad_norm,
            }
        )
    return {
        "optimizer_steps": int(args.steps),
        "max_grad_norm": max_grad_norm,
        "stage_history": [],
        "loss_history": loss_history,
    }


def _run_configured_stage_training(
    model: MultimodalOVHA,
    optimizer: torch.optim.Optimizer,
    adapter: ControlledSyntheticMultimodalAdapter,
    args: argparse.Namespace,
    device: torch.device,
    config: MultimodalExperimentConfig,
) -> dict[str, object]:
    steps_per_stage = int(args.steps_per_stage or max(1, args.steps))
    loss_history: list[dict[str, object]] = []
    stage_history: list[dict[str, object]] = []
    max_grad_norm = 0.0
    optimizer_steps = 0
    for stage in config.training_stages:
        configured_losses = tuple((config.losses_by_stage or {}).get(stage, ()))
        if stage == "T0":
            stage_history.append(
                {
                    "stage": stage,
                    "loss_names_configured": list(configured_losses),
                    "loss_names_observed": ["cache_validation"],
                    "optimizer_steps": 0,
                    "policy": "controlled synthetic in-memory batch generation validated before optimizer stages",
                }
            )
            continue
        stage_families = _families_for_stage(stage)
        stage_rows: list[dict[str, object]] = []
        parameter_scopes: list[dict[str, object]] = []
        for stage_step in range(steps_per_stage):
            family = _family_for_training_step(
                stage,
                stage_families,
                stage_step=stage_step,
                optimizer_steps=optimizer_steps,
            )
            specialist_candidate = _specialist_candidate_for_stage(stage, family)
            parameter_scope = _apply_trainable_parameter_scope(model, stage, family)
            parameter_scopes.append(parameter_scope)
            batch = _sample(adapter, family, args, device)
            optimizer.zero_grad(set_to_none=True)
            router_weight_override = _router_weight_override_for_stage(stage, batch)
            output = model(batch, router_weight_override=router_weight_override)
            components = _controlled_loss_components(output, batch, specialist_candidate=specialist_candidate)
            stage_components = {name: components[name] for name in configured_losses if name in components}
            total_loss = _stage_total_loss(stage_components, output)
            total_loss.backward()
            grad_norm = _grad_l2_norm(model)
            max_grad_norm = max(max_grad_norm, grad_norm)
            optimizer.step()
            optimizer_steps += 1
            step_row = {
                "step": optimizer_steps,
                "stage": stage,
                "family": family,
                "loss_names_observed": sorted(stage_components),
                "route_override_mode": _route_override_mode(router_weight_override),
                "trainable_parameter_scope": parameter_scope["trainable_parameter_scope"],
                "trainable_parameter_groups": parameter_scope["trainable_parameter_groups"],
                "total_loss": _as_float(total_loss),
                "grad_l2_norm": grad_norm,
                **{name: _as_float(value) for name, value in stage_components.items()},
            }
            if specialist_candidate is not None:
                step_row["specialist_candidate"] = specialist_candidate
            stage_rows.append(step_row)
            loss_history.append(step_row)
        stage_parameter_scope = _merge_parameter_scopes(stage, parameter_scopes)
        stage_history.append(
            {
                "stage": stage,
                "loss_names_configured": list(configured_losses),
                "loss_names_observed": sorted({name for row in stage_rows for name in row["loss_names_observed"]}),
                "optimizer_steps": len(stage_rows),
                "family_schedule_scope": _family_schedule_scope(stage),
                "route_override_mode": _stage_route_override_mode(stage),
                **stage_parameter_scope,
                "mean_total_loss": float(sum(float(row["total_loss"]) for row in stage_rows) / max(len(stage_rows), 1)),
                "families_seen": sorted({str(row["family"]) for row in stage_rows}),
            }
        )
    return {
        "optimizer_steps": optimizer_steps,
        "max_grad_norm": max_grad_norm,
        "stage_history": stage_history,
        "loss_history": loss_history,
    }


def _stage_total_loss(stage_components: dict[str, torch.Tensor], output: MultimodalOVHAOutput) -> torch.Tensor:
    if not stage_components:
        return output.y_hat.sum() * 0.0
    return torch.stack([value for value in stage_components.values()]).sum()


def _families_for_stage(stage: str) -> tuple[str, ...]:
    if stage == "T1":
        return CONTROLLED_SPECIALIST_WARMUP_FAMILIES
    return CONTROLLED_MULTIMODAL_FAMILIES


def _family_for_training_step(
    stage: str,
    families: tuple[str, ...],
    *,
    stage_step: int,
    optimizer_steps: int,
) -> str:
    if stage == "T1":
        return families[stage_step % len(families)]
    return families[optimizer_steps % len(families)]


def _family_schedule_scope(stage: str) -> str:
    if stage == "T1":
        return "single_candidate_specialist_warmup"
    return "full_controlled_family_coverage"


def _router_weight_override_for_stage(stage: str, batch: Any) -> torch.Tensor | None:
    if stage not in {"T1", "T2"}:
        return None
    hidden = batch.hidden or {}
    override = hidden.get("true_router_weights")
    return override if isinstance(override, torch.Tensor) else None


def _route_override_mode(router_weight_override: torch.Tensor | None) -> str:
    return "true_router_weights" if router_weight_override is not None else "learned_router"


def _stage_route_override_mode(stage: str) -> str:
    return "true_router_weights" if stage in {"T1", "T2"} else "learned_router"


def _specialist_candidate_for_stage(stage: str, family: str) -> str | None:
    if stage != "T1":
        return None
    active = CONTROLLED_FAMILY_ACTIVE_OPERATOR[family]
    if active not in CONTROLLED_CANDIDATE_NAMES:
        raise ValueError(f"T1 specialist warmup requires a single candidate family, got {family}: {active}")
    return active


def _apply_trainable_parameter_scope(model: MultimodalOVHA, stage: str, family: str) -> dict[str, object]:
    trainable_groups = _trainable_parameter_groups_for_stage(stage, family)
    observed_groups: set[str] = set()
    frozen_groups: set[str] = set()
    for name, parameter in model.named_parameters():
        group = _parameter_group_for_name(name)
        is_trainable = group in trainable_groups
        parameter.requires_grad_(is_trainable)
        if is_trainable:
            observed_groups.add(group)
        else:
            frozen_groups.add(group)
    return {
        "trainable_parameter_scope": _trainable_parameter_scope_name(stage),
        "trainable_parameter_groups": sorted(observed_groups),
        "frozen_parameter_groups": sorted(frozen_groups),
    }


def _trainable_parameter_groups_for_stage(stage: str, family: str) -> set[str]:
    if stage == "T1":
        candidate = _specialist_candidate_for_stage(stage, family)
        return {
            f"candidate_primitives.{candidate}",
            f"joint_router_adapter.hyper_adapter.{candidate}",
        }
    if stage == "T2":
        return _adapter_candidate_parameter_groups()
    if stage == "T3":
        return {"joint_router_adapter.router"}
    groups = {
        "candidate_primitives",
        "evidence_encoder",
        "joint_router_adapter.hyper_adapter",
        "joint_router_adapter.router",
        "memory_encoder",
        "reliability_prior",
    }
    groups.update(f"candidate_primitives.{candidate}" for candidate in CONTROLLED_CANDIDATE_NAMES)
    groups.update(f"joint_router_adapter.hyper_adapter.{candidate}" for candidate in CONTROLLED_CANDIDATE_NAMES)
    return groups


def _adapter_candidate_parameter_groups() -> set[str]:
    return {
        *(f"candidate_primitives.{candidate}" for candidate in CONTROLLED_CANDIDATE_NAMES),
        *(f"joint_router_adapter.hyper_adapter.{candidate}" for candidate in CONTROLLED_CANDIDATE_NAMES),
    }


def _trainable_parameter_scope_name(stage: str) -> str:
    if stage == "T1":
        return "candidate_only_specialist_warmup"
    if stage == "T3":
        return "router_only_warmup"
    if stage == "T2":
        return "oracle_router_adapter_candidate_warmup"
    if stage == "T4":
        return "joint_controlled_training"
    return "full_model"


def _parameter_group_for_name(name: str) -> str:
    if name.startswith("candidate_primitives."):
        parts = name.split(".")
        return f"candidate_primitives.{parts[1]}"
    if name.startswith("joint_router_adapter.router."):
        return "joint_router_adapter.router"
    if name.startswith("joint_router_adapter.hyper_adapter."):
        parts = name.split(".")
        if len(parts) > 3 and parts[2] == "heads":
            return f"joint_router_adapter.hyper_adapter.{parts[3]}"
        return "joint_router_adapter.hyper_adapter"
    return name.split(".", 1)[0]


def _merge_parameter_scopes(stage: str, parameter_scopes: list[dict[str, object]]) -> dict[str, object]:
    trainable: set[str] = set()
    frozen: set[str] = set()
    for scope in parameter_scopes:
        trainable.update(str(group) for group in scope.get("trainable_parameter_groups", []))
        frozen.update(str(group) for group in scope.get("frozen_parameter_groups", []))
    return {
        "trainable_parameter_scope": _trainable_parameter_scope_name(stage),
        "trainable_parameter_groups": sorted(trainable),
        "frozen_parameter_groups": sorted(_with_parent_group_aliases(frozen)),
    }


def _with_parent_group_aliases(groups: set[str]) -> set[str]:
    expanded = set(groups)
    if any(group.startswith("candidate_primitives.") for group in groups):
        expanded.add("candidate_primitives")
    if any(group.startswith("joint_router_adapter.hyper_adapter.") for group in groups):
        expanded.add("joint_router_adapter.hyper_adapter")
    return expanded


def _controlled_losses(output: MultimodalOVHAOutput, batch: Any) -> dict[str, torch.Tensor]:
    components = _controlled_loss_components(output, batch)
    return {
        "task_loss": components["task_loss"],
        "router_ce_true_active_operator": components["router_ce_true_active_operator"],
        "candidate_oracle_mse": components["candidate_oracle_mse"],
        "total_loss": components["task_loss"]
        + 0.1 * components["router_ce_true_active_operator"]
        + 0.1 * components["candidate_oracle_mse"],
    }


def _controlled_loss_components(
    output: MultimodalOVHAOutput,
    batch: Any,
    *,
    specialist_candidate: str | None = None,
) -> dict[str, torch.Tensor]:
    active = batch.hidden["true_active_operator"]
    router_ce = torch.nn.functional.cross_entropy(
        output.router_logits.reshape(-1, output.router_logits.shape[-1]),
        active.reshape(-1),
    )
    candidate_oracle_mse = (output.candidate_values - batch.hidden["true_candidate_values"]).square().mean()
    task_loss = _task_loss(output.y_hat, batch)
    return {
        "task_loss": task_loss,
        "candidate_individual_loss": _candidate_individual_loss(output, batch, specialist_candidate),
        "router_ce_true_active_operator": router_ce,
        "candidate_oracle_mse": candidate_oracle_mse,
        "adapter_kl_true_params": _adapter_true_param_loss(output, batch),
        "cato_alignment_ce": _candidate_oracle_loss(output, batch, "CATO"),
        "lrio_rank_kl": _candidate_oracle_loss(output, batch, "LRIO"),
        "spo_prototype_kl": _candidate_oracle_loss(output, batch, "SPO"),
        "tleo_lengthscale_huber": _tleo_lengthscale_loss(output, batch),
        "rceo_reliability_huber": _rceo_reliability_loss(output, batch),
    }


def _candidate_individual_loss(
    output: MultimodalOVHAOutput,
    batch: Any,
    specialist_candidate: str | None,
) -> torch.Tensor:
    if specialist_candidate is None:
        return (output.candidate_values - batch.target_y.unsqueeze(-2)).square().mean()
    candidate_index = CONTROLLED_CANDIDATE_NAMES.index(specialist_candidate)
    return (output.candidate_values[..., candidate_index, :] - batch.target_y).square().mean()


def _evaluate_task_losses(
    model: MultimodalOVHA,
    adapter: ControlledSyntheticMultimodalAdapter,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for family in CONTROLLED_MULTIMODAL_FAMILIES:
            batch = _sample(adapter, family, args, device)
            output = model(batch)
            losses[family] = _as_float(_task_loss(output.y_hat, batch))
    return {
        "mean_task_loss": float(sum(losses.values()) / len(losses)),
        "task_loss_by_family": losses,
    }


def _controlled_rows(
    model: MultimodalOVHA,
    adapter: ControlledSyntheticMultimodalAdapter,
    args: argparse.Namespace,
    device: torch.device,
    *,
    training_steps: int,
    config_name: str | None = None,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for family in CONTROLLED_MULTIMODAL_FAMILIES:
            batch = _sample(adapter, family, args, device)
            output = model(batch)
            oracle_report = evaluate_oracle_matrix(
                batch,
                learned_candidate_values=output.candidate_values,
                learned_router_weights=output.router_weights,
            )
            row = controlled_row_from_oracle_report(
                family,
                CONTROLLED_FAMILY_ACTIVE_OPERATOR[family],
                oracle_report,
                router_accuracy=_router_accuracy(output, batch),
                stackability_passed=bool(output.diagnostics.get("stackability_passed")),
                diagnostics=_row_diagnostics(output, batch),
            )
            row["training_mode"] = "trained_smoke"
            row["training_steps"] = int(training_steps)
            if config_name is not None:
                row["training_config_name"] = config_name
            row["learned_task_loss"] = _as_float(_task_loss(output.y_hat, batch))
            rows.append(row)
    return rows


def _row_diagnostics(output: MultimodalOVHAOutput, batch: Any) -> dict[str, object]:
    router_entropy = -(output.router_weights * output.router_weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
    diagnostics: dict[str, object] = {
        "learned_router_entropy": _as_float(router_entropy),
        "candidate_oracle_mse": _as_float((output.candidate_values - batch.hidden["true_candidate_values"]).square().mean()),
    }
    reliability = output.reliability_prior
    if reliability is not None and "mean_reliability" in reliability.diagnostics:
        diagnostics["learned_mean_reliability"] = _as_float(reliability.diagnostics["mean_reliability"])
    return diagnostics


def _task_loss(prediction: torch.Tensor, batch: Any) -> torch.Tensor:
    mask = batch.target_mask.to(dtype=prediction.dtype, device=prediction.device).unsqueeze(-1)
    return ((prediction - batch.target_y).square() * mask).sum() / mask.sum().clamp_min(1.0)


def _candidate_oracle_loss(output: MultimodalOVHAOutput, batch: Any, candidate_name: str) -> torch.Tensor:
    index = ("TLEO", "SPO", "LRIO", "CATO").index(candidate_name)
    return (output.candidate_values[..., index, :] - batch.hidden["true_candidate_values"][..., index, :]).square().mean()


def _adapter_true_param_loss(output: MultimodalOVHAOutput, batch: Any) -> torch.Tensor:
    hidden_params = (batch.hidden or {}).get("true_adapter_params", {}).get("params_by_operator", {})
    detail = output.diagnostics.get("adapter_params_detail", {})
    losses: list[torch.Tensor] = []
    for operator, params in hidden_params.items():
        operator_detail = detail.get(operator, {})
        for key, true_value in params.items():
            predicted = operator_detail.get(f"{key}_mean")
            if predicted is None:
                continue
            true_mean = true_value.to(device=predicted.device, dtype=predicted.dtype).mean()
            losses.append(torch.nn.functional.smooth_l1_loss(predicted, true_mean))
    if not losses:
        return output.y_hat.sum() * 0.0
    return torch.stack(losses).sum()


def _tleo_lengthscale_loss(output: MultimodalOVHAOutput, batch: Any) -> torch.Tensor:
    predicted = output.diagnostics.get("adapter_params_detail", {}).get("TLEO", {}).get("lengthscale_mean")
    true_lengthscale = (batch.hidden or {}).get("true_lengthscale")
    if predicted is None or true_lengthscale is None:
        return _candidate_oracle_loss(output, batch, "TLEO")
    true_mean = true_lengthscale.to(device=predicted.device, dtype=predicted.dtype).mean()
    return torch.nn.functional.smooth_l1_loss(predicted, true_mean)


def _rceo_reliability_loss(output: MultimodalOVHAOutput, batch: Any) -> torch.Tensor:
    true_reliability = (batch.hidden or {}).get("true_reliability")
    if output.reliability_prior is None or true_reliability is None:
        return output.y_hat.sum() * 0.0
    predicted = output.reliability_prior.modality_reliability.mean(dim=-1, keepdim=True)
    target = true_reliability.to(device=predicted.device, dtype=predicted.dtype).reshape(predicted.shape[0], -1).mean(dim=-1, keepdim=True)
    return torch.nn.functional.smooth_l1_loss(predicted, target)


def _router_accuracy(output: MultimodalOVHAOutput, batch: Any) -> float:
    predicted = output.router_weights.argmax(dim=-1)
    active = batch.hidden["true_active_operator"]
    return _as_float((predicted == active).to(dtype=torch.float32).mean())


def _parameter_vector(model: MultimodalOVHA) -> torch.Tensor:
    parts = [param.detach().reshape(-1).cpu() for param in model.parameters() if param.requires_grad]
    return torch.cat(parts) if parts else torch.zeros(0)


def _grad_l2_norm(model: MultimodalOVHA) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().square().sum().cpu())
    return float(total**0.5)


def _sample(
    adapter: ControlledSyntheticMultimodalAdapter,
    family: str,
    args: argparse.Namespace,
    device: torch.device,
):
    return adapter.sample_batch(
        family=family,
        batch_size=args.batch_size,
        query_count=args.query_count,
        device=str(device),
    )


def _write_evidence_artifacts(artifact_root: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    controlled_rows = artifact_root / "controlled_training_rows.jsonl"
    diagnostics_report = artifact_root / "controlled_training_diagnostics.jsonl"
    _write_jsonl(controlled_rows, rows)
    _write_jsonl(diagnostics_report, [_diagnostics_row(row) for row in rows])
    return {
        "task": "controlled_multimodal",
        "generated_by": "scripts/multimodal/run_controlled_training_smoke.py",
        "controlled_rows": {"path": str(controlled_rows), "sha256": file_sha256(controlled_rows)},
        "diagnostics_report": {"path": str(diagnostics_report), "sha256": file_sha256(diagnostics_report)},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _diagnostics_row(row: dict[str, object]) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "artifact_type": "controlled_training_diagnostics",
        "family": row["family"],
        "training_mode": row["training_mode"],
        "training_steps": row["training_steps"],
        "learned_task_loss": row["learned_task_loss"],
        "stackability_passed": row["stackability_passed"],
        "oracle_matrix": row["oracle_matrix"],
    }
    for key in (
        "TLEO_oracle_gap",
        "SPO_oracle_gap",
        "LRIO_oracle_gap",
        "CATO_oracle_gap",
        "candidate_oracle_mse",
        "learned_router_entropy",
        "learned_mean_reliability",
        "rceo_prior_effect",
        "router_accuracy",
    ):
        if key in row:
            diagnostics[key] = row[key]
    if "training_config_name" in row:
        diagnostics["training_config_name"] = row["training_config_name"]
    return diagnostics


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
