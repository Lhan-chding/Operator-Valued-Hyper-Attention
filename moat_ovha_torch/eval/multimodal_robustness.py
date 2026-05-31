from __future__ import annotations

from collections import defaultdict
from typing import Any


DEFAULT_REQUIRED_STRESS_FAMILIES = {
    "missing_text": ("missing_text", "text_missing"),
    "missing_vision": ("missing_vision", "missing_visual", "vision_missing", "visual_missing"),
    "missing_audio": ("missing_audio", "audio_missing"),
    "image_quality": ("image_blur", "image_crop", "image_occlusion", "visual_blur", "visual_crop", "visual_occlusion"),
    "audio_quality": ("audio_noise", "audio_masking", "audio_mask"),
    "text_noise": ("text_token_mask", "text_mask", "paraphrase_noise", "text_paraphrase"),
    "hard_negative_mismatch": (
        "hard_negative_mismatch",
        "hard_negative_caption_mismatch",
        "hard_negative_region_mismatch",
        "hard_negative_audio_mismatch",
        "caption_mismatch",
        "region_mismatch",
        "audio_mismatch",
    ),
}
TEMPORAL_STRESS_FAMILIES = {"temporal_shift": ("temporal_shift", "temporal_shift_sec")}


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
    relative_drop = {model: _relative_drop(model_rows) for model, model_rows in by_model.items()}
    auc = {model: _auc_over_corruption(model_rows) for model, model_rows in by_model.items()}
    full_rows = sorted(by_model.get(full_model, []), key=lambda row: float(row["corruption_strength"]))
    reliability_monotonic = _non_increasing([float(row.get("rceo_reliability", 0.0)) for row in full_rows])
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
        "relative_drop": relative_drop,
        "auc_over_corruption_strength": auc,
        "rceo_reliability_monotonic": reliability_monotonic,
        "rceo_reliability_shift": reliability_shift,
        "operator_load_shift": load_shift,
        "candidate_loss_shift": candidate_loss_shift,
        "required_stress_coverage": required_stress_coverage,
        "required_ablation_degradation": required_ablation_degradation,
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


def _rceo_reliability_shift(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2:
        return None
    try:
        first = float(rows[0]["rceo_reliability"])
        last = float(rows[-1]["rceo_reliability"])
    except (KeyError, TypeError, ValueError):
        return None
    return last - first


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
    required = dict(required_stress_families or DEFAULT_REQUIRED_STRESS_FAMILIES)
    if temporal_data:
        required.update(TEMPORAL_STRESS_FAMILIES)
    observed = _observed_stress_families(rows, required)
    missing = sorted(family for family in required if family not in observed)
    return {
        "passed": not missing,
        "observed": sorted(observed),
        "required": sorted(required),
        "condition": "robustness rows must cover every required Step 6 stress family",
        "reasons": [f"missing robustness stress family: {family}" for family in missing],
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
        if corruption_type in alias_to_family:
            observed.add(alias_to_family[corruption_type])
        observed.update(_families_from_missing_modalities(row, alias_to_family))
        if _has_mismatch_metadata(row):
            observed.add("hard_negative_mismatch")
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
    if isinstance(mismatch, str) and mismatch:
        return True
    corruption_type = _normalize_stress_name(row.get("corruption_type", ""))
    return "mismatch" in corruption_type or "hard_negative" in corruption_type


def _has_temporal_shift(row: dict[str, Any]) -> bool:
    shift = row.get("temporal_shift_sec")
    try:
        return abs(float(shift)) > 0.0
    except (TypeError, ValueError):
        return False


def _normalize_stress_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")
