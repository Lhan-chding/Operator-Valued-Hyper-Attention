from __future__ import annotations

from moat_ovha_torch.models.multimodal.operator_bank import MULTIMODAL_CANDIDATE_NAMES


def evaluate_oracle_matrix(batch) -> dict[str, dict[str, object] | float]:
    if batch.hidden is None:
        raise ValueError("controlled oracle matrix requires batch.hidden")
    required = ("true_candidate_values", "true_router_weights")
    missing = [name for name in required if name not in batch.hidden]
    if missing:
        raise ValueError(f"controlled oracle matrix missing hidden truth: {missing}")
    candidate_values = batch.hidden["true_candidate_values"]
    router_weights = batch.hidden["true_router_weights"]
    true_true = (router_weights.unsqueeze(-1) * candidate_values).sum(dim=-2)
    mse = (true_true - batch.target_y).square().mean()
    report: dict[str, dict[str, object] | float] = {
        "true_true": {"mse": mse, "purpose": "candidate expressivity upper bound"},
        "true_learned": {"mse": mse, "purpose": "adapter/candidate isolation scaffold"},
        "learned_true": {"mse": mse, "purpose": "router isolation scaffold"},
        "learned_learned": {"mse": mse, "purpose": "full model placeholder for trained eval"},
    }
    for index, name in enumerate(MULTIMODAL_CANDIDATE_NAMES):
        candidate_mse = (candidate_values[..., index, :] - batch.target_y).square().mean()
        report[f"{name}_oracle_gap"] = candidate_mse
    return report
