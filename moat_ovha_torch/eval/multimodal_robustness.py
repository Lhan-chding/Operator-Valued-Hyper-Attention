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
    full_rows = sorted(_finite_strength_rows(by_model.get(full_model, [])), key=lambda row: float(row["corruption_strength"]))
    reliability_curve = _rceo_reliability_curve(full_rows)
    reliability_monotonic = _non_increasing([point["mean_reliability"] for point in reliability_curve])
    reliability_shift = _rceo_reliability_shift(full_rows)
    load_shift = _operator_load_shift(full_rows)
    candidate_loss_shift = _candidate_loss_shift(full_rows)
    required_ablation_degradation = _required_ablation_degradation(
        relative_drop,
        full_model,
        required_ablation_models,
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
        "operator_load_shift": load_shift,
        "candidate_loss_shift": candidate_loss_shift,
        "required_stress_coverage": required_stress_coverage,
        "required_ablation_degradation": required_ablation_degradation,
        "full_drop_less_than_baseline": relative_drop.get(full_model, float("inf")) < relative_drop.get(baseline_model, float("-inf")),
    }


def _relative_drop(rows: list[dict[str, Any]]) -> float:
    finite_rows = _finite_strength_rows(rows)
    if not finite_rows:
        return float("inf")
    clean = _endpoint_score(finite_rows, first=True)
    corrupted = _endpoint_score(finite_rows, first=False)
    return (clean - corrupted) / max(abs(clean), 1e-12)


def _endpoint_score(rows: list[dict[str, Any]], *, first: bool) -> float:
    if not rows:
        return float("inf")
    by_strength: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        strength = _finite_float(row.get("corruption_strength"))
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
    ordered = sorted(_finite_strength_rows(rows), key=lambda row: float(row["corruption_strength"]))
    if not ordered:
        return 0.0
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


def _rceo_reliability_curve(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    reliability_by_strength: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        strength = _finite_float(row.get("corruption_strength"))
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
        if _finite_float(row.get("corruption_strength")) is None:
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


def _finite_strength_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _finite_float(row.get("corruption_strength")) is not None]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
