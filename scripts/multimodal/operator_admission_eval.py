#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate typed operator admission from per-sample predictions.")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alignment-threshold", type=float, default=0.0)
    parser.add_argument("--non-interference-tolerance", type=float, default=0.0)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)
    args = parser.parse_args()

    try:
        payload = evaluate_operator_admission(
            _read_jsonl(args.predictions_jsonl),
            base_model=args.base_model,
            candidate_model=args.candidate_model,
            split=args.split,
            alpha_max=float(args.alpha_max),
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.seed),
            alignment_threshold=float(args.alignment_threshold),
            non_interference_tolerance=float(args.non_interference_tolerance),
            bootstrap_alpha=float(args.bootstrap_alpha),
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2, sort_keys=True))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "output": str(args.output), "admission_decision": payload["admission_decision"]}, sort_keys=True))
    return 0


def evaluate_operator_admission(
    rows: list[dict[str, Any]],
    *,
    base_model: str,
    candidate_model: str,
    split: str = "val",
    alpha_max: float = 1.0,
    bootstrap_samples: int = 1000,
    seed: int = 0,
    alignment_threshold: float = 0.0,
    non_interference_tolerance: float = 0.0,
    bootstrap_alpha: float = 0.05,
) -> dict[str, Any]:
    joined = _joined_prediction_rows(rows, base_model=base_model, candidate_model=candidate_model, split=split)
    if not joined:
        raise ValueError("no joined base/candidate rows with truth were found")
    residuals = [item["truth"] - item["base_prediction"] for item in joined]
    candidate_deltas = [item["candidate_delta"] for item in joined]
    base_errors = [(item["truth"] - item["base_prediction"]) ** 2 for item in joined]
    corrected_errors = [(item["truth"] - (item["base_prediction"] + item["candidate_delta"])) ** 2 for item in joined]
    residual_alignment = _mean([residual * delta for residual, delta in zip(residuals, candidate_deltas)])
    delta_norm = sum(delta * delta for delta in candidate_deltas)
    alpha_star = 0.0 if delta_norm <= 1e-12 else max(0.0, min(float(alpha_max), sum(residual * delta for residual, delta in zip(residuals, candidate_deltas)) / delta_norm))
    optimal_errors = [(residual - alpha_star * delta) ** 2 for residual, delta in zip(residuals, candidate_deltas)]
    base_mse = _mean(base_errors)
    corrected_mse = _mean(corrected_errors)
    optimal_alpha_mse = _mean(optimal_errors)
    non_interference_delta = corrected_mse - base_mse
    optimal_alpha_delta = optimal_alpha_mse - base_mse
    bootstrap = _paired_bootstrap(
        base_errors,
        corrected_errors,
        samples=max(0, int(bootstrap_samples)),
        seed=seed,
    )
    decision = _admission_decision(
        residual_alignment=residual_alignment,
        alignment_threshold=alignment_threshold,
        non_interference_delta=non_interference_delta,
        non_interference_tolerance=non_interference_tolerance,
        paired_bootstrap_p=bootstrap["paired_bootstrap_p"],
        bootstrap_alpha=bootstrap_alpha,
    )
    return {
        "ok": True,
        "mode": "operator_admission_eval",
        "policy": "admit only positive residual utility with non-interference and paired bootstrap support",
        "split": split,
        "base_model": base_model,
        "candidate_model": candidate_model,
        "sample_count": len(joined),
        "residual_alignment": residual_alignment,
        "base_mse": base_mse,
        "candidate_corrected_mse": corrected_mse,
        "non_interference_delta": non_interference_delta,
        "optimal_alpha": alpha_star,
        "optimal_alpha_distribution": {
            "mean": alpha_star,
            "median": alpha_star,
            "min": alpha_star,
            "max": alpha_star,
            "alpha_max": float(alpha_max),
        },
        "optimal_alpha_mse": optimal_alpha_mse,
        "optimal_alpha_delta": optimal_alpha_delta,
        **bootstrap,
        "admission_decision": decision,
        "gate_results": {
            "residual_utility_gate": residual_alignment > alignment_threshold,
            "non_interference_gate": non_interference_delta <= non_interference_tolerance,
            "statistical_gate": bootstrap["paired_bootstrap_p"] <= bootstrap_alpha,
        },
    }


def _joined_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    base_model: str,
    candidate_model: str,
    split: str,
) -> list[dict[str, float]]:
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        if str(row.get("split", split)) != split:
            continue
        model = str(row.get("model", ""))
        if model not in {base_model, candidate_model}:
            continue
        key = str(row.get("sample_id", row.get("source_id", index)))
        by_key.setdefault(key, {})[model] = row
    joined = []
    for key, values in sorted(by_key.items()):
        base = values.get(base_model)
        candidate = values.get(candidate_model)
        if base is None or candidate is None:
            continue
        truth = _scalar(base.get("truth", base.get("target", base.get("label"))), field=f"{key}.truth")
        base_prediction = _scalar(base.get("prediction", base.get("y_hat")), field=f"{key}.base_prediction")
        candidate_delta = _scalar(
            candidate.get("candidate_delta", candidate.get("residual_prediction", candidate.get("prediction", candidate.get("y_hat")))),
            field=f"{key}.candidate_delta",
        )
        joined.append({"truth": truth, "base_prediction": base_prediction, "candidate_delta": candidate_delta})
    return joined


def _paired_bootstrap(base_errors: list[float], corrected_errors: list[float], *, samples: int, seed: int) -> dict[str, Any]:
    observed_delta = _mean(corrected_errors) - _mean(base_errors)
    if not base_errors or samples <= 0:
        return {"paired_bootstrap_p": 1.0, "bootstrap_delta_mean": observed_delta, "bootstrap_delta_ci95": [observed_delta, observed_delta]}
    rng = random.Random(seed)
    deltas = []
    n = len(base_errors)
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(_mean([corrected_errors[index] - base_errors[index] for index in indices]))
    p_value = sum(1 for delta in deltas if delta <= 0.0) / len(deltas)
    return {
        "paired_bootstrap_p": p_value,
        "bootstrap_delta_mean": _mean(deltas),
        "bootstrap_delta_ci95": [_quantile(deltas, 0.025), _quantile(deltas, 0.975)],
    }


def _admission_decision(
    *,
    residual_alignment: float,
    alignment_threshold: float,
    non_interference_delta: float,
    non_interference_tolerance: float,
    paired_bootstrap_p: float,
    bootstrap_alpha: float,
) -> str:
    if residual_alignment <= alignment_threshold:
        return "rejected"
    if non_interference_delta > non_interference_tolerance:
        return "diagnostic_only"
    if paired_bootstrap_p > bootstrap_alpha:
        return "pending_statistical_gate"
    return "admitted"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return rows


def _scalar(value: Any, *, field: str) -> float:
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, list):
        flat = _flatten(value)
        if not flat:
            raise ValueError(f"{field} must not be empty")
        number = statistics.fmean(float(item) for item in flat)
    else:
        raise ValueError(f"{field} must be a number or numeric list")
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _flatten(value: list[Any]) -> list[Any]:
    values = []
    for item in value:
        if isinstance(item, list):
            values.extend(_flatten(item))
        else:
            values.append(item)
    return values


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return statistics.fmean(values)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute quantile for empty sequence")
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


if __name__ == "__main__":
    raise SystemExit(main())
