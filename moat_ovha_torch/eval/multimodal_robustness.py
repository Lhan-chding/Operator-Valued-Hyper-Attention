from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


DEFAULT_REQUIRED_STRESS_TARGETS = {
    "missing_text": ("missing_text", "text_missing"),
    "missing_vision": ("missing_vision", "missing_visual", "vision_missing", "visual_missing"),
    "missing_audio": ("missing_audio", "audio_missing"),
    "image_blur": ("image_blur", "visual_blur"),
    "image_crop": ("image_crop", "visual_crop"),
    "image_occlusion": ("image_occlusion", "visual_occlusion"),
    "audio_noise": ("audio_noise",),
    "audio_masking": ("audio_masking", "audio_mask"),
    "text_token_mask": ("text_token_mask", "text_mask"),
    "text_paraphrase": ("paraphrase_noise", "text_paraphrase"),
    "hard_negative_caption_mismatch": ("hard_negative_caption_mismatch", "caption_mismatch"),
    "hard_negative_region_mismatch": ("hard_negative_region_mismatch", "region_mismatch"),
    "hard_negative_audio_mismatch": ("hard_negative_audio_mismatch", "audio_mismatch"),
}
DEFAULT_REQUIRED_STRESS_FAMILIES = DEFAULT_REQUIRED_STRESS_TARGETS
TEMPORAL_STRESS_TARGETS = {"temporal_shift": ("temporal_shift", "temporal_shift_sec")}
_RCEO_OBSERVED_RELIABILITY_KEYS = (
    "rceo_observed_reliability",
    "observed_reliability",
    "target_reliability",
    "reliability_target",
    "observed_accuracy",
)


def summarize_robustness_rows(
    rows: list[dict[str, Any]],
    *,
    full_model: str,
    baseline_model: str,
    required_ablation_models: tuple[str, ...] = ("ovha_no_rceo", "ovha_no_evidence_router"),
    required_stress_families: dict[str, tuple[str, ...]] | None = None,
    temporal_data: bool = False,
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
    clean_score = {model: _endpoint_score(model_rows, first=True) for model, model_rows in by_model.items()}
    corrupted_score = {model: _endpoint_score(model_rows, first=False) for model, model_rows in by_model.items()}
    relative_drop = {model: _relative_drop(model_rows) for model, model_rows in by_model.items()}
    auc = {model: _auc_over_corruption(model_rows) for model, model_rows in by_model.items()}
    full_rows = sorted(_valid_strength_rows(by_model.get(full_model, [])), key=lambda row: float(row["corruption_strength"]))
    reliability_curve = _rceo_reliability_curve(full_rows)
    reliability_monotonic = _non_increasing([point["mean_reliability"] for point in reliability_curve])
    reliability_shift = _rceo_reliability_shift(full_rows)
    load_shift = _operator_load_shift(full_rows)
    candidate_loss_shift = _candidate_loss_shift(full_rows)
    reliability_calibration = _rceo_reliability_calibration(full_rows)
    required_ablation_degradation = _required_ablation_degradation(
        relative_drop,
        full_model,
        required_ablation_models,
    )
    robustness_significance = _robustness_significance(
        by_model,
        full_model=full_model,
        baseline_model=baseline_model,
    )
    required_stress_coverage = _required_stress_coverage(
        rows,
        required_stress_families=required_stress_families,
        temporal_data=temporal_data,
    )
    return {
        "full_model": full_model,
        "baseline_model": baseline_model,
        "clean_score": clean_score,
        "corrupted_score": corrupted_score,
        "relative_drop": relative_drop,
        "auc_over_corruption_strength": auc,
        "rceo_reliability_monotonic": reliability_monotonic,
        "rceo_reliability_shift": reliability_shift,
        "rceo_reliability_curve": reliability_curve,
        "rceo_reliability_calibration": reliability_calibration,
        "operator_load_shift": load_shift,
        "candidate_loss_shift": candidate_loss_shift,
        "required_stress_coverage": required_stress_coverage,
        "required_ablation_degradation": required_ablation_degradation,
        "robustness_significance": robustness_significance,
        "full_drop_less_than_baseline": relative_drop.get(full_model, float("inf")) < relative_drop.get(baseline_model, float("-inf")),
    }


def _relative_drop(rows: list[dict[str, Any]]) -> float:
    valid_rows = _valid_strength_rows(rows)
    if not valid_rows:
        return float("inf")
    clean = _endpoint_score(valid_rows, first=True)
    corrupted = _endpoint_score(valid_rows, first=False)
    return (clean - corrupted) / max(abs(clean), 1e-12)


def _endpoint_score(rows: list[dict[str, Any]], *, first: bool) -> float:
    if not rows:
        return float("inf")
    by_strength: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        strength = _valid_corruption_strength(row.get("corruption_strength"))
        score = _finite_float(row.get("score"))
        if strength is None or score is None:
            continue
        by_strength[strength].append(score)
    if not by_strength:
        return float("inf")
    strength = min(by_strength) if first else max(by_strength)
    values = by_strength[strength]
    return sum(values) / len(values)


def _auc_over_corruption(rows: list[dict[str, Any]]) -> float:
    score_curve = _mean_score_by_strength(rows)
    if not score_curve:
        return 0.0
    if len(score_curve) == 1:
        return score_curve[0][1]
    auc = 0.0
    for left, right in zip(score_curve, score_curve[1:]):
        x0, y0 = left
        x1, y1 = right
        auc += (x1 - x0) * (y0 + y1) / 2.0
    span = score_curve[-1][0] - score_curve[0][0]
    return auc / max(span, 1e-12)


def _mean_score_by_strength(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    by_strength: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        strength = _valid_corruption_strength(row.get("corruption_strength"))
        score = _finite_float(row.get("score"))
        if strength is None or score is None:
            continue
        by_strength[strength].append(score)
    return [
        (strength, sum(values) / len(values))
        for strength, values in sorted(by_strength.items())
    ]


def _non_increasing(values: list[float]) -> bool:
    return all(right <= left + 1e-12 for left, right in zip(values, values[1:]))


def _rceo_reliability_curve(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    reliability_by_strength: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        strength = _valid_corruption_strength(row.get("corruption_strength"))
        reliability = _finite_float(row.get("rceo_reliability"))
        if strength is None or reliability is None:
            continue
        reliability_by_strength[strength].append(reliability)
    return [
        {
            "corruption_strength": strength,
            "mean_reliability": sum(values) / len(values),
        }
        for strength, values in sorted(reliability_by_strength.items())
    ]


def _rceo_reliability_calibration(rows: list[dict[str, Any]], *, bin_count: int = 5) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        predicted = _finite_probability(row.get("rceo_reliability"))
        observed = _observed_reliability(row)
        if predicted is None or observed is None:
            continue
        pairs.append((predicted, observed))
    if not pairs:
        return {
            "ece": None,
            "expected_calibration_error": None,
            "bin_count": 0,
            "calibration_curve": [],
            "condition": "RCEO reliability calibration requires explicit observed reliability labels",
        }

    bucket_count = max(1, int(bin_count))
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for predicted, observed in pairs:
        bucket = min(int(predicted * bucket_count), bucket_count - 1)
        buckets[bucket].append((predicted, observed))

    curve: list[dict[str, float | int]] = []
    for output_bin, bucket in enumerate(sorted(buckets)):
        values = buckets[bucket]
        mean_confidence = sum(predicted for predicted, _ in values) / len(values)
        observed_accuracy = sum(observed for _, observed in values) / len(values)
        curve.append(
            {
                "bin": output_bin,
                "mean_confidence": mean_confidence,
                "observed_accuracy": observed_accuracy,
                "count": len(values),
            }
        )

    total = sum(int(point["count"]) for point in curve)
    ece = sum(
        int(point["count"]) / total * abs(float(point["mean_confidence"]) - float(point["observed_accuracy"]))
        for point in curve
    )
    return {
        "ece": ece,
        "expected_calibration_error": ece,
        "bin_count": len(curve),
        "calibration_curve": curve,
        "condition": "Expected calibration error between predicted RCEO reliability and observed reliability",
    }


def _operator_load_shift(rows: list[dict[str, Any]]) -> dict[str, float]:
    if len(rows) < 2:
        return {}
    first = rows[0].get("router_load_by_candidate", {}) or {}
    last = rows[-1].get("router_load_by_candidate", {}) or {}
    names = sorted(set(first) | set(last))
    return {name: float(last.get(name, 0.0)) - float(first.get(name, 0.0)) for name in names}


def _rceo_reliability_shift(rows: list[dict[str, Any]]) -> float | None:
    curve = _rceo_reliability_curve(rows)
    if len(curve) < 2:
        return None
    return curve[-1]["mean_reliability"] - curve[0]["mean_reliability"]


def _candidate_loss_shift(rows: list[dict[str, Any]]) -> dict[str, float]:
    if len(rows) < 2:
        return {}
    first = rows[0].get("candidate_loss", {}) or {}
    last = rows[-1].get("candidate_loss", {}) or {}
    if not isinstance(first, dict) or not isinstance(last, dict):
        return {}
    names = sorted(set(first) | set(last))
    shifts: dict[str, float] = {}
    for name in names:
        try:
            shifts[name] = float(last.get(name, 0.0)) - float(first.get(name, 0.0))
        except (TypeError, ValueError):
            return {}
    return shifts


def _required_ablation_degradation(
    relative_drop: dict[str, float],
    full_model: str,
    required_ablation_models: tuple[str, ...],
) -> dict[str, Any]:
    reasons: list[str] = []
    values: dict[str, float] = {}
    full_drop = relative_drop.get(full_model)
    if full_drop is None:
        reasons.append(f"missing robustness rows for full model: {full_model}")
        full_drop = float("inf")
    for model in required_ablation_models:
        if model not in relative_drop:
            reasons.append(f"missing robustness ablation rows: {model}")
            continue
        delta = float(relative_drop[model]) - float(full_drop)
        values[model] = delta
        if delta <= 0.0:
            reasons.append(f"robustness ablation does not degrade more than full: {model}")
    return {
        "passed": not reasons,
        "value": values,
        "condition": "required robustness ablations must show larger relative drop than ovha_full",
        "reasons": reasons,
    }


def _robustness_significance(
    by_model: dict[str, list[dict[str, Any]]],
    *,
    full_model: str,
    baseline_model: str,
) -> dict[str, Any]:
    full_by_seed = _relative_drop_by_seed(by_model.get(full_model, []))
    baseline_by_seed = _relative_drop_by_seed(by_model.get(baseline_model, []))
    common = sorted(set(full_by_seed) & set(baseline_by_seed))
    per_seed = [
        {
            "seed": seed,
            "full_relative_drop": full_by_seed[seed],
            "baseline_relative_drop": baseline_by_seed[seed],
            "drop_delta": baseline_by_seed[seed] - full_by_seed[seed],
        }
        for seed in common
    ]
    deltas = [row["drop_delta"] for row in per_seed]
    drop_delta = _mean(deltas) if deltas else None
    return {
        "model_delta": f"{baseline_model}_relative_drop_minus_{full_model}_relative_drop",
        "metric": "relative_drop_delta",
        "delta_interpretation": f"positive means {full_model} drops less under robustness stress",
        "common_seed_count": len(common),
        "drop_delta": drop_delta,
        "paired_permutation_p": _paired_sign_permutation_p(deltas),
        "paired_bootstrap_ci95": list(_bootstrap_ci95(deltas)),
        "per_seed_drop_delta": per_seed,
    }


def _relative_drop_by_seed(rows: list[dict[str, Any]]) -> dict[int, float]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed = _seed_value(row.get("seed"))
        if seed is not None:
            grouped[seed].append(row)
    drops: dict[int, float] = {}
    for seed, seed_rows in grouped.items():
        drop = _relative_drop(seed_rows)
        if math.isfinite(drop):
            drops[seed] = drop
    return drops


def _seed_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _paired_sign_permutation_p(deltas: list[float]) -> float:
    if not deltas:
        return 1.0
    observed = abs(sum(deltas))
    total = 2 ** len(deltas)
    extreme = 0
    for mask in range(total):
        signed_sum = 0.0
        for index, delta in enumerate(deltas):
            signed_sum += delta if mask & (1 << index) else -delta
        if abs(signed_sum) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def _bootstrap_ci95(deltas: list[float]) -> tuple[float, float]:
    if not deltas:
        return (0.0, 0.0)
    if len(deltas) == 1:
        return (deltas[0], deltas[0])
    import random

    rng = random.Random(1729)
    means = []
    for _ in range(10000):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(_mean(sample))
    means = sorted(means)
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return (means[low_index], means[high_index])


def _required_stress_coverage(
    rows: list[dict[str, Any]],
    *,
    required_stress_families: dict[str, tuple[str, ...]] | None,
    temporal_data: bool,
) -> dict[str, Any]:
    required = dict(required_stress_families or DEFAULT_REQUIRED_STRESS_TARGETS)
    if temporal_data:
        required.update(TEMPORAL_STRESS_TARGETS)
    observed = _observed_stress_families(rows, required)
    missing = sorted(target for target in required if target not in observed)
    return {
        "passed": not missing,
        "observed": sorted(observed),
        "required": sorted(required),
        "condition": "robustness rows must cover every required Step 6 stress target",
        "reasons": [f"missing robustness stress target: {target}" for target in missing],
    }


def _observed_stress_families(
    rows: list[dict[str, Any]],
    required: dict[str, tuple[str, ...]],
) -> set[str]:
    observed: set[str] = set()
    alias_to_family = {
        _normalize_stress_name(alias): family
        for family, aliases in required.items()
        for alias in (family, *aliases)
    }
    for row in rows:
        corruption_type = _normalize_stress_name(row.get("corruption_type", ""))
        if _valid_corruption_strength(row.get("corruption_strength")) is None:
            continue
        target = alias_to_family.get(corruption_type)
        if target is not None and _target_metadata_present(row, target):
            observed.add(target)
        observed.update(_families_from_missing_modalities(row, alias_to_family))
        if _has_temporal_shift(row):
            observed.add("temporal_shift")
    return observed


def _families_from_missing_modalities(row: dict[str, Any], alias_to_family: dict[str, str]) -> set[str]:
    missing = row.get("missing_modalities", ())
    if not isinstance(missing, (list, tuple, set)):
        return set()
    observed = set()
    for modality in missing:
        family = alias_to_family.get(f"missing_{_normalize_stress_name(modality)}")
        if family is not None:
            observed.add(family)
    return observed


def _has_mismatch_metadata(row: dict[str, Any]) -> bool:
    mismatch = row.get("mismatch_source_id")
    return isinstance(mismatch, str) and bool(mismatch.strip())


def _target_metadata_present(row: dict[str, Any], target: str) -> bool:
    if target.startswith("missing_"):
        return _missing_modalities_contains(row, target.removeprefix("missing_"))
    if target == "temporal_shift":
        return _has_temporal_shift(row)
    if target.startswith("hard_negative_") and target.endswith("_mismatch"):
        return _has_mismatch_metadata(row)
    return True


def _missing_modalities_contains(row: dict[str, Any], modality: str) -> bool:
    missing = row.get("missing_modalities", ())
    if not isinstance(missing, (list, tuple, set)):
        return False
    normalized = {_normalize_stress_name(value) for value in missing}
    if modality == "vision":
        return "vision" in normalized or "visual" in normalized
    return modality in normalized


def _has_temporal_shift(row: dict[str, Any]) -> bool:
    shift = row.get("temporal_shift_sec")
    try:
        return abs(float(shift)) > 0.0
    except (TypeError, ValueError):
        return False


def _normalize_stress_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _valid_strength_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _valid_corruption_strength(row.get("corruption_strength")) is not None]


def _valid_corruption_strength(value: Any) -> float | None:
    number = _finite_float(value)
    if number is None or number < 0.0:
        return None
    return number


def _finite_probability(value: Any) -> float | None:
    number = _finite_float(value)
    if number is None or number < 0.0 or number > 1.0:
        return None
    return number


def _observed_reliability(row: dict[str, Any]) -> float | None:
    for key in _RCEO_OBSERVED_RELIABILITY_KEYS:
        if key in row:
            return _finite_probability(row.get(key))
    return None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
