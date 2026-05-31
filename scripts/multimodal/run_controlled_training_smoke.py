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

from moat_ovha_torch.data.multimodal.adapters.controlled_synthetic import (
    CONTROLLED_FAMILY_ACTIVE_OPERATOR,
    CONTROLLED_MULTIMODAL_FAMILIES,
    ControlledSyntheticMultimodalAdapter,
)
from moat_ovha_torch.data.multimodal.cache_schema import file_sha256
from moat_ovha_torch.eval.multimodal_controlled_report import build_controlled_report
from moat_ovha_torch.eval.multimodal_oracle import controlled_row_from_oracle_report, evaluate_oracle_matrix
from moat_ovha_torch.models.multimodal.ovha_multimodal import MultimodalOVHA, MultimodalOVHAOutput


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real controlled multimodal OVHA training smoke.")
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--field-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--memory-tokens", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    adapter = ControlledSyntheticMultimodalAdapter(
        seed=args.seed + 4,
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
    loss_history: list[dict[str, object]] = []
    max_grad_norm = 0.0
    model.train()
    for step in range(args.steps):
        family = CONTROLLED_MULTIMODAL_FAMILIES[step % len(CONTROLLED_MULTIMODAL_FAMILIES)]
        batch = adapter.sample_batch(
            family=family,
            batch_size=args.batch_size,
            query_count=args.query_count,
            device=str(device),
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        losses = _controlled_losses(output, batch)
        losses["total_loss"].backward()
        grad_norm = _grad_l2_norm(model)
        max_grad_norm = max(max_grad_norm, grad_norm)
        optimizer.step()
        loss_history.append(
            {
                "step": step + 1,
                "family": family,
                "task_loss": _as_float(losses["task_loss"]),
                "router_ce_true_active_operator": _as_float(losses["router_ce_true_active_operator"]),
                "candidate_oracle_mse": _as_float(losses["candidate_oracle_mse"]),
                "total_loss": _as_float(losses["total_loss"]),
                "grad_l2_norm": grad_norm,
            }
        )

    after = _evaluate_task_losses(model, adapter, args, device)
    parameter_l2_delta = torch.linalg.vector_norm(_parameter_vector(model) - initial_parameters).item()
    rows = _controlled_rows(model, adapter, args, device)
    evidence_artifacts = _write_evidence_artifacts(args.artifact_root, rows) if args.artifact_root else None
    mean_before = before["mean_task_loss"]
    mean_after = after["mean_task_loss"]
    ok = bool(parameter_l2_delta > 0.0 and max_grad_norm > 0.0 and mean_after < mean_before)
    payload = {
        "ok": ok,
        "mode": "trained_smoke",
        "training": {
            "optimizer": "AdamW",
            "optimizer_steps": args.steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "mean_task_loss_before": mean_before,
            "mean_task_loss_after": mean_after,
            "task_loss_by_family_before": before["task_loss_by_family"],
            "task_loss_by_family_after": after["task_loss_by_family"],
            "parameter_l2_delta": float(parameter_l2_delta),
            "max_grad_norm": max_grad_norm,
            "loss_history": loss_history,
        },
        "rows": rows,
        "controlled_report": build_controlled_report(rows, evidence_artifacts=evidence_artifacts),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 2


def _controlled_losses(output: MultimodalOVHAOutput, batch: Any) -> dict[str, torch.Tensor]:
    active = batch.hidden["true_active_operator"]
    router_ce = torch.nn.functional.cross_entropy(
        output.router_logits.reshape(-1, output.router_logits.shape[-1]),
        active.reshape(-1),
    )
    candidate_oracle_mse = (output.candidate_values - batch.hidden["true_candidate_values"]).square().mean()
    task_loss = _task_loss(output.y_hat, batch)
    return {
        "task_loss": task_loss,
        "router_ce_true_active_operator": router_ce,
        "candidate_oracle_mse": candidate_oracle_mse,
        "total_loss": task_loss + 0.1 * router_ce + 0.1 * candidate_oracle_mse,
    }


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
            batch = adapter.sample_batch(
                family=family,
                batch_size=args.batch_size,
                query_count=args.query_count,
                device=str(device),
            )
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
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for family in CONTROLLED_MULTIMODAL_FAMILIES:
            batch = adapter.sample_batch(
                family=family,
                batch_size=args.batch_size,
                query_count=args.query_count,
                device=str(device),
            )
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
            row["training_steps"] = int(args.steps)
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
    return diagnostics


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
