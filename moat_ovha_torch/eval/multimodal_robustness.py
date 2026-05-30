from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_robustness_rows(
    rows: list[dict[str, Any]],
    *,
    full_model: str,
    baseline_model: str,
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
    relative_drop = {model: _relative_drop(model_rows) for model, model_rows in by_model.items()}
    auc = {model: _auc_over_corruption(model_rows) for model, model_rows in by_model.items()}
    full_rows = sorted(by_model.get(full_model, []), key=lambda row: float(row["corruption_strength"]))
    reliability_monotonic = _non_increasing([float(row.get("rceo_reliability", 0.0)) for row in full_rows])
    load_shift = _operator_load_shift(full_rows)
    return {
        "full_model": full_model,
        "baseline_model": baseline_model,
        "relative_drop": relative_drop,
        "auc_over_corruption_strength": auc,
        "rceo_reliability_monotonic": reliability_monotonic,
        "operator_load_shift": load_shift,
        "full_drop_less_than_baseline": relative_drop.get(full_model, float("inf")) < relative_drop.get(baseline_model, float("-inf")),
    }


def _relative_drop(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("inf")
    ordered = sorted(rows, key=lambda row: float(row["corruption_strength"]))
    clean = float(ordered[0]["score"])
    corrupted = float(ordered[-1]["score"])
    return (clean - corrupted) / max(abs(clean), 1e-12)


def _auc_over_corruption(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    ordered = sorted(rows, key=lambda row: float(row["corruption_strength"]))
    if len(ordered) == 1:
        return float(ordered[0]["score"])
    auc = 0.0
    for left, right in zip(ordered, ordered[1:]):
        x0 = float(left["corruption_strength"])
        x1 = float(right["corruption_strength"])
        y0 = float(left["score"])
        y1 = float(right["score"])
        auc += (x1 - x0) * (y0 + y1) / 2.0
    span = float(ordered[-1]["corruption_strength"]) - float(ordered[0]["corruption_strength"])
    return auc / max(span, 1e-12)


def _non_increasing(values: list[float]) -> bool:
    return all(right <= left + 1e-12 for left, right in zip(values, values[1:]))


def _operator_load_shift(rows: list[dict[str, Any]]) -> dict[str, float]:
    if len(rows) < 2:
        return {}
    first = rows[0].get("router_load_by_candidate", {}) or {}
    last = rows[-1].get("router_load_by_candidate", {}) or {}
    names = sorted(set(first) | set(last))
    return {name: float(last.get(name, 0.0)) - float(first.get(name, 0.0)) for name in names}
