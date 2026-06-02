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

from moat_ovha_torch.data.multimodal.adapters.base import MissingMultimodalDataError
from moat_ovha_torch.data.multimodal.cache_schema import MultimodalCacheLayout, validate_cache_layout
from scripts.multimodal.build_cache import ADAPTERS, _cache_splits_for


DEFAULT_DATASETS = ("refcoco", "cmu_mosei")
CONFIG_BY_DATASET = {
    "refcoco": "configs/multimodal_refcoco_public_smoke.json",
    "cmu_mosei": "configs/multimodal_cmu_mosei_public_smoke.json",
}
ACCEPTANCE_ARTIFACT_BY_DATASET = {
    "refcoco": "outputs/multimodal/refcoco_public_acceptance_multiseed",
    "cmu_mosei": "outputs/multimodal/cmu_mosei_public_acceptance_multiseed",
}
CMU_MOSEI_MEAN_PREPROCESSING_MARKER = "cmu-mosei-cmu-sdk-mean"
CMU_MOSEI_PUBLIC_MAIN_MIN_SAMPLE_COUNT = 10_000
CMU_MOSEI_MODALITIES = ("text", "audio", "vision")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect public multimodal data-prep state and print the next concrete commands. "
            "This is a status helper, not a training gate."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DEFAULT_DATASETS), default=list(DEFAULT_DATASETS))
    parser.add_argument("--download-root", type=Path, default=Path("data/raw_multimodal/_downloads"))
    parser.add_argument("--raw-root-base", type=Path, default=Path("data/raw_multimodal"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/multimodal_cache"))
    parser.add_argument(
        "--controlled-report",
        type=Path,
        default=Path("outputs/multimodal/controlled_v1_smoke/seed_101/controlled_report.json"),
    )
    parser.add_argument("--version", default="v0.1")
    args = parser.parse_args()

    payload = inspect_public_data_readiness(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def inspect_public_data_readiness(args: argparse.Namespace) -> dict[str, Any]:
    next_commands: list[str] = []
    datasets = list(dict.fromkeys(args.datasets))
    controlled_report = _controlled_report_phase(args.controlled_report)
    dataset_reports = {
        dataset_name: _dataset_report(
            dataset_name,
            download_root=args.download_root,
            raw_root_base=args.raw_root_base,
            cache_root=args.cache_root,
            version=args.version,
            controlled_report=args.controlled_report,
            next_commands=next_commands,
            all_datasets=datasets,
        )
        for dataset_name in datasets
    }
    ok = all(report["ok"] for report in dataset_reports.values()) and bool(controlled_report["ok"])
    return {
        "ok": ok,
        "mode": "public_data_readiness",
        "policy": "inspection-only: reports data-prep artifacts and next commands; does not block training",
        "datasets": dataset_reports,
        "controlled_report": controlled_report,
        "download_root": str(args.download_root),
        "raw_root_base": str(args.raw_root_base),
        "cache_root": str(args.cache_root),
        "version": args.version,
        "next_commands": _unique(next_commands),
    }


def _dataset_report(
    dataset_name: str,
    *,
    download_root: Path,
    raw_root_base: Path,
    cache_root: Path,
    version: str,
    controlled_report: Path,
    next_commands: list[str],
    all_datasets: list[str],
) -> dict[str, Any]:
    raw_root = raw_root_base / dataset_name
    layout = MultimodalCacheLayout(cache_root, dataset_name, version)
    phases: dict[str, dict[str, Any]] = {}

    if dataset_name == "refcoco":
        phases.update(_refcoco_download_and_stage_phases(download_root))
    elif dataset_name == "cmu_mosei":
        phases.update(_cmu_mosei_download_and_stage_phases(download_root))

    raw_manifest = _raw_manifest_phase(dataset_name, raw_root)
    cache = _cache_phase(layout, dataset_name)
    phases["raw_manifest"] = raw_manifest
    phases["cache"] = cache
    phases["public_main_quality"] = _public_main_quality_phase(dataset_name, phases, layout)

    commands = _next_commands_for_dataset(
        dataset_name,
        phases=phases,
        download_root=download_root,
        raw_root=raw_root,
        cache_root=cache_root,
        controlled_report=controlled_report,
        all_datasets=all_datasets,
    )
    next_commands.extend(commands)
    next_action = commands[0] if commands else "ready: formal cache is valid; continue with public acceptance/training"
    return {
        "ok": bool(cache["ok"]) and bool(phases["public_main_quality"]["ok"]),
        "dataset_name": dataset_name,
        "raw_root": str(raw_root),
        "cache_root": str(layout.root),
        "next_action": next_action,
        "phases": phases,
    }


def _refcoco_download_and_stage_phases(download_root: Path) -> dict[str, dict[str, Any]]:
    download_dir = download_root / "refcoco"
    stage_dir = download_root / "refcoco_stage_inputs"
    downloads = _path_phase(
        "downloads",
        [
            download_dir / "annotations_trainval2014.zip",
            download_dir / "refcoco.zip",
        ],
        optional_paths=[
            download_dir / "train2014.zip",
            download_dir / "extracted" / "annotations" / "instances_train2014.json",
        ],
    )
    stage_records = _refcoco_stage_records_phase(stage_dir)
    aligned_features = _refcoco_aligned_features_phase(stage_dir)
    return {
        "downloads": downloads,
        "stage_records": stage_records,
        "aligned_features": aligned_features,
    }


def _cmu_mosei_download_and_stage_phases(download_root: Path) -> dict[str, dict[str, Any]]:
    sdk_dir = download_root / "cmu_sdk" / "cmu_mosei"
    stage_dir = download_root / "cmu_mosei_stage_inputs"
    sequence_files = _sequence_files(sdk_dir)
    downloads = {
        "ok": bool(sequence_files),
        "name": "downloads",
        "root": str(sdk_dir),
        "existing": [str(path) for path in sequence_files[:12]],
        "missing": [] if sequence_files else [str(sdk_dir / "*.csd|*.h5|*.json")],
    }
    splits = _path_phase("splits", [download_root / "cmu_mosei_splits.json"])
    sequence_inspection = _path_phase("sequence_inspection", [download_root / "cmu_mosei_sequence_inspection.json"])
    stage_inputs = _cmu_mosei_stage_inputs_phase(stage_dir)
    return {
        "downloads": downloads,
        "splits": splits,
        "sequence_inspection": sequence_inspection,
        "stage_inputs": stage_inputs,
    }


def _path_phase(name: str, required_paths: list[Path], optional_paths: list[Path] | None = None) -> dict[str, Any]:
    existing = [path for path in required_paths if path.exists()]
    missing = [path for path in required_paths if not path.exists()]
    optional_existing = [path for path in optional_paths or [] if path.exists()]
    return {
        "ok": not missing,
        "name": name,
        "required": [str(path) for path in required_paths],
        "existing": [str(path) for path in existing],
        "missing": [str(path) for path in missing],
        "optional_existing": [str(path) for path in optional_existing],
    }


def _refcoco_stage_records_phase(stage_dir: Path) -> dict[str, Any]:
    phase = _path_phase(
        "stage_records",
        [
            stage_dir / "refcoco_splits.json",
            stage_dir / "refcoco_phrase_region_records.json",
        ],
    )
    records_path = stage_dir / "refcoco_phrase_region_records.json"
    if not phase["ok"] or not records_path.exists():
        return phase
    try:
        payload = json.loads(records_path.read_text())
    except json.JSONDecodeError as exc:
        return {**phase, "ok": False, "errors": [f"invalid stage records JSON: {exc}"]}
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        return {**phase, "ok": False, "errors": ["stage records must contain records"]}
    checked = records[: min(256, len(records))]
    degenerate = [
        str(record.get("source_id", index))
        for index, record in enumerate(checked)
        if not isinstance(record, dict)
        or not isinstance(record.get("candidate_region_boxes"), list)
        or len(record.get("candidate_region_boxes", [])) <= 1
    ]
    if degenerate:
        return {
            **phase,
            "ok": False,
            "errors": ["RefCOCO stage records are degenerate: candidate_region_boxes missing or length <= 1"],
            "degenerate_examples": degenerate[:5],
        }
    return phase


def _cmu_mosei_stage_inputs_phase(stage_dir: Path) -> dict[str, Any]:
    phase = _path_phase(
        "stage_inputs",
        [
            stage_dir / "cmu_mosei_splits.json",
            stage_dir / "cmu_mosei_text_features.npy",
            stage_dir / "cmu_mosei_audio_features.npy",
            stage_dir / "cmu_mosei_visual_features.npy",
            stage_dir / "cmu_mosei_sentiment.npy",
            stage_dir / "cmu_mosei_emotion.npy",
            stage_dir / "cmu_mosei_stage_input_manifest.json",
        ],
    )
    if not phase["ok"]:
        return phase

    manifest_path = stage_dir / "cmu_mosei_stage_input_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return {**phase, "ok": False, "errors": [f"invalid CMU-MOSEI stage input manifest: {exc}"]}
    if not isinstance(manifest, dict):
        return {**phase, "ok": False, "errors": ["CMU-MOSEI stage input manifest must be a JSON object"]}

    feature_shapes = _npy_shapes(
        {
            "text": stage_dir / "cmu_mosei_text_features.npy",
            "audio": stage_dir / "cmu_mosei_audio_features.npy",
            "vision": stage_dir / "cmu_mosei_visual_features.npy",
        }
    )
    errors = _cmu_mosei_public_main_stage_errors(manifest, feature_shapes)
    return {
        **phase,
        "ok": not errors,
        "errors": errors,
        "manifest_summary": _cmu_mosei_manifest_summary(manifest),
        "feature_shapes": feature_shapes,
    }


def _refcoco_aligned_features_phase(stage_dir: Path) -> dict[str, Any]:
    phase = _path_phase(
        "aligned_features",
        [
            stage_dir / "refcoco_text_features.npy",
            stage_dir / "refcoco_region_features.npy",
        ],
        optional_paths=[
            stage_dir / "refcoco_feature_alignment_manifest.json",
            stage_dir / "refcoco_clip_feature_manifest.json",
            stage_dir / "refcoco_text_mask.npy",
            stage_dir / "refcoco_region_mask.npy",
        ],
    )
    if not phase["ok"]:
        return phase
    manifest_candidates = [
        stage_dir / "refcoco_feature_alignment_manifest.json",
        stage_dir / "refcoco_clip_feature_manifest.json",
    ]
    if not any(path.exists() for path in manifest_candidates):
        return {
            **phase,
            "ok": False,
            "errors": ["missing RefCOCO feature manifest: expected feature_alignment or clip_feature manifest"],
        }
    errors: list[str] = []
    shapes: dict[str, list[int]] = {}
    for modality in ("text", "region"):
        path = stage_dir / f"refcoco_{modality}_features.npy"
        try:
            import numpy as np

            array = np.load(path, allow_pickle=False, mmap_mode="r")
        except (ImportError, OSError, ValueError, TypeError) as exc:
            errors.append(f"{path.name} must be a loadable numpy array: {exc}")
            continue
        shapes[modality] = [int(dim) for dim in array.shape]
        if array.ndim < 2 or int(array.shape[1]) <= 1:
            errors.append(f"{path.name} token/candidate axis must be > 1")
    return {**phase, "ok": not errors, "feature_shapes": shapes, "errors": errors}


def _public_main_quality_phase(
    dataset_name: str,
    phases: dict[str, dict[str, Any]],
    layout: MultimodalCacheLayout,
) -> dict[str, Any]:
    if dataset_name != "cmu_mosei":
        return {"ok": True, "name": "public_main_quality", "errors": [], "warnings": []}
    errors: list[str] = []
    warnings: list[str] = []
    stage_inputs = phases.get("stage_inputs", {})
    stage_errors = stage_inputs.get("errors", [])
    if _phase_has_all_required_paths(stage_inputs) and isinstance(stage_errors, list):
        errors.extend(str(error) for error in stage_errors)

    cache_report = _cmu_mosei_cache_public_main_report(layout)
    errors.extend(cache_report["errors"])
    warnings.extend(cache_report["warnings"])
    return {
        "ok": not errors,
        "name": "public_main_quality",
        "errors": _unique(errors),
        "warnings": _unique(warnings),
        "cache_report": cache_report,
    }


def _phase_has_all_required_paths(phase: dict[str, Any]) -> bool:
    required = phase.get("required", [])
    existing = phase.get("existing", [])
    return bool(required) and set(required) == set(existing)


def _cmu_mosei_cache_public_main_report(layout: MultimodalCacheLayout) -> dict[str, Any]:
    root = layout.root
    errors: list[str] = []
    warnings: list[str] = []
    feature_versions = _read_json_if_exists(root / "provenance" / "feature_versions.json")
    feature_version_text = json.dumps(feature_versions, sort_keys=True).lower() if feature_versions is not None else ""
    uses_mean_sdk = CMU_MOSEI_MEAN_PREPROCESSING_MARKER in feature_version_text
    token_shapes = _cmu_mosei_cache_token_shapes(layout)
    split_source_counts = _cmu_mosei_cache_source_counts(layout)
    sample_count = sum(split_source_counts.values())

    if uses_mean_sdk:
        errors.append(
            "CMU-MOSEI public main cache uses mean-pooled SDK feature versions; rebuild "
            "utterance/segment-level features before acceptance/training"
        )
    if token_shapes and _all_cmu_mosei_token_axes_are_degenerate(token_shapes):
        errors.append(
            "CMU-MOSEI public main cache has token axis <= 1 for text/audio/vision; this is a "
            "mean-pooled or source-level cache, not full utterance/segment-level evidence"
        )
    if uses_mean_sdk and 0 < sample_count < CMU_MOSEI_PUBLIC_MAIN_MIN_SAMPLE_COUNT:
        errors.append(
            f"CMU-MOSEI public main cache has only {sample_count} retained samples; expected "
            "full utterance/segment-level MOSEI scale, not the 3225-video/source-level mean cache"
        )
    if root.exists() and feature_versions is None:
        warnings.append("CMU-MOSEI cache exists but feature_versions.json was unavailable for public-main quality checks")
    return {
        "errors": errors,
        "warnings": warnings,
        "feature_versions_uses_mean_sdk": uses_mean_sdk,
        "token_shapes": token_shapes,
        "split_source_counts": split_source_counts,
        "sample_count": sample_count,
    }


def _cmu_mosei_public_main_stage_errors(manifest: dict[str, Any], feature_shapes: dict[str, list[int]]) -> list[str]:
    errors: list[str] = []
    preprocessing_version = str(manifest.get("preprocessing_version", "")).lower()
    temporal_policy = str(manifest.get("temporal_policy", "")).lower()
    sample_count = _int_or_zero(manifest.get("sample_count"))
    if temporal_policy == "mean" or CMU_MOSEI_MEAN_PREPROCESSING_MARKER in preprocessing_version:
        errors.append(
            "CMU-MOSEI public main stage inputs are mean-pooled; rebuild utterance/segment-level "
            "stage inputs before acceptance/training"
        )
    if feature_shapes and _all_cmu_mosei_token_axes_are_degenerate(feature_shapes):
        errors.append(
            "CMU-MOSEI public main stage inputs have token axis <= 1 for text/audio/vision; "
            "this cannot support full temporal/local evidence"
        )
    if 0 < sample_count < CMU_MOSEI_PUBLIC_MAIN_MIN_SAMPLE_COUNT:
        errors.append(
            f"CMU-MOSEI public main stage inputs contain only {sample_count} samples; expected "
            "full utterance/segment-level MOSEI scale"
        )
    return errors


def _cmu_mosei_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": manifest.get("sample_count"),
        "temporal_policy": manifest.get("temporal_policy"),
        "preprocessing_version": manifest.get("preprocessing_version"),
    }


def _cmu_mosei_cache_token_shapes(layout: MultimodalCacheLayout) -> dict[str, dict[str, list[int]]]:
    shapes: dict[str, dict[str, list[int]]] = {}
    for split in ("train", "val", "test"):
        split_shapes: dict[str, list[int]] = {}
        for modality in CMU_MOSEI_MODALITIES:
            path = layout.root / "token_fields" / f"{modality}_{split}.npy"
            shape = _npy_shape(path)
            if shape is not None:
                split_shapes[modality] = shape
        if split_shapes:
            shapes[split] = split_shapes
    return shapes


def _cmu_mosei_cache_source_counts(layout: MultimodalCacheLayout) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = layout.root / "provenance" / f"source_ids_{split}.txt"
        if path.exists():
            counts[split] = len([line for line in path.read_text().splitlines() if line.strip()])
    return counts


def _all_cmu_mosei_token_axes_are_degenerate(shapes: dict[str, Any]) -> bool:
    token_axes: list[int] = []
    for value in shapes.values():
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, list) and len(nested) >= 2:
                    token_axes.append(int(nested[1]))
        elif isinstance(value, list) and len(value) >= 2:
            token_axes.append(int(value[1]))
    return bool(token_axes) and all(axis <= 1 for axis in token_axes)


def _npy_shapes(paths: dict[str, Path]) -> dict[str, list[int]]:
    return {name: shape for name, path in paths.items() if (shape := _npy_shape(path)) is not None}


def _npy_shape(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    try:
        import numpy as np

        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except (ImportError, OSError, ValueError, TypeError):
        return None
    return [int(dim) for dim in array.shape]


def _read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _raw_manifest_phase(dataset_name: str, raw_root: Path) -> dict[str, Any]:
    adapter = ADAPTERS[dataset_name]()
    try:
        manifest = adapter.discover_raw(raw_root)
    except MissingMultimodalDataError as exc:
        return {
            "ok": False,
            "raw_root": str(raw_root),
            "errors": [str(exc)],
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "raw_root": str(raw_root),
            "errors": [str(exc)],
        }
    return {
        "ok": True,
        "raw_root": str(manifest.raw_root),
        "file_count": len(manifest.files),
        "files": sorted(manifest.files),
    }


def _cache_phase(layout: MultimodalCacheLayout, dataset_name: str) -> dict[str, Any]:
    splits = _cache_splits_for(dataset_name)
    validation = validate_cache_layout(layout, splits=splits)
    return {
        "ok": validation.ok,
        "cache_root": str(layout.root),
        "splits": list(splits),
        "errors": validation.errors,
        "warnings": validation.warnings,
    }


def _controlled_report_phase(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "path": str(path), "errors": ["controlled report does not exist"]}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "path": str(path), "errors": [str(exc)]}
    report = _controlled_report_payload(payload)
    reported_ok = bool(payload.get("ok")) if isinstance(payload, dict) else False
    go_no_go = report.get("go_no_go", {}) if isinstance(report, dict) else {}
    go_no_go_ok = (
        isinstance(go_no_go, dict)
        and go_no_go.get("controlled_multimodal_passed") is True
        and go_no_go.get("enter_public_multimodal") is True
        and go_no_go.get("reasons", []) == []
    )
    if not (reported_ok or go_no_go_ok):
        return {
            "ok": False,
            "path": str(path),
            "reported_ok": reported_ok,
            "go_no_go_ok": go_no_go_ok,
            "errors": ["controlled report must have ok=true or go_no_go public-entry flags true with empty reasons"],
        }
    return {"ok": True, "path": str(path), "reported_ok": reported_ok, "go_no_go_ok": go_no_go_ok}


def _controlled_report_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("controlled_report")
    return nested if isinstance(nested, dict) else payload


def _next_commands_for_dataset(
    dataset_name: str,
    *,
    phases: dict[str, dict[str, Any]],
    download_root: Path,
    raw_root: Path,
    cache_root: Path,
    controlled_report: Path,
    all_datasets: list[str],
) -> list[str]:
    commands: list[str] = []
    if phases["cache"]["ok"] and phases.get("public_main_quality", {}).get("ok", True):
        return [_acceptance_command(dataset_name, raw_root, cache_root, controlled_report)]
    if dataset_name == "cmu_mosei" and not phases.get("public_main_quality", {}).get("ok", True):
        return _cmu_mosei_public_main_rebuild_commands(download_root)
    if _needs_download_command(dataset_name, phases):
        commands.extend(_bootstrap_commands(all_datasets, download_root))

    if dataset_name == "refcoco":
        _append_refcoco_commands(commands, phases, download_root, raw_root, cache_root, controlled_report)
    elif dataset_name == "cmu_mosei":
        _append_cmu_mosei_commands(commands, phases, download_root, raw_root, cache_root, controlled_report)
    return _unique(commands)


def _needs_download_command(dataset_name: str, phases: dict[str, dict[str, Any]]) -> bool:
    if phases.get("downloads", {}).get("ok", False):
        return False
    if phases.get("raw_manifest", {}).get("ok", False) or phases.get("cache", {}).get("ok", False):
        return False
    if dataset_name == "refcoco":
        return not phases.get("stage_records", {}).get("ok", False)
    if dataset_name == "cmu_mosei":
        return not phases.get("stage_inputs", {}).get("ok", False)
    return True


def _append_refcoco_commands(
    commands: list[str],
    phases: dict[str, dict[str, Any]],
    download_root: Path,
    raw_root: Path,
    cache_root: Path,
    controlled_report: Path,
) -> None:
    stage_dir = download_root / "refcoco_stage_inputs"
    if not phases["stage_records"]["ok"]:
        commands.append(
            "python scripts/multimodal/build_refcoco_stage_records.py refcoco "
            f"{stage_dir} "
            "--refs data/raw_multimodal/_downloads/refcoco/extracted/replace_with_refcoco_refs.json "
            "--instances data/raw_multimodal/_downloads/refcoco/extracted/annotations/instances_train2014.json "
            "--candidate-region-source coco_gt_box"
        )
    if phases["stage_records"]["ok"] and not phases["aligned_features"]["ok"]:
        commands.append(
            "python scripts/multimodal/extract_refcoco_clip_features.py refcoco "
            f"{stage_dir} "
            f"--splits {stage_dir / 'refcoco_splits.json'} "
            f"--records {stage_dir / 'refcoco_phrase_region_records.json'} "
            "--refs data/raw_multimodal/_downloads/refcoco/extracted/replace_with_refcoco_refs.p "
            "--image-root data/raw_multimodal/_downloads/refcoco/extracted/train2014 "
            "--image-root data/raw_multimodal/_downloads/refcoco/extracted/val2014 "
            "--device cuda --model openai/clip-vit-base-patch32 --revision main"
        )
    if phases["stage_records"]["ok"] and phases["aligned_features"]["ok"] and not phases["raw_manifest"]["ok"]:
        mask_args = _refcoco_mask_stage_args(stage_dir)
        commands.append(
            "python scripts/multimodal/stage_refcoco_raw.py refcoco "
            f"{raw_root} "
            f"--splits {stage_dir / 'refcoco_splits.json'} "
            f"--records {stage_dir / 'refcoco_phrase_region_records.json'} "
            f"--text-features {stage_dir / 'refcoco_text_features.npy'} "
            f"--region-features {stage_dir / 'refcoco_region_features.npy'} "
            f"{mask_args}"
            "--license-tag refcoco-coco2014 "
            "--preprocessing-version refcoco-frozen-features-v0.1"
        )
    _append_cache_or_acceptance_command(commands, "refcoco", raw_root, cache_root, controlled_report, phases)


def _refcoco_mask_stage_args(stage_dir: Path) -> str:
    args = []
    text_mask = stage_dir / "refcoco_text_mask.npy"
    region_mask = stage_dir / "refcoco_region_mask.npy"
    if text_mask.exists():
        args.append(f"--text-mask {text_mask}")
    if region_mask.exists():
        args.append(f"--region-mask {region_mask}")
    return " ".join(args) + (" " if args else "")


def _append_cmu_mosei_commands(
    commands: list[str],
    phases: dict[str, dict[str, Any]],
    download_root: Path,
    raw_root: Path,
    cache_root: Path,
    controlled_report: Path,
) -> None:
    if not phases.get("public_main_quality", {}).get("ok", True):
        commands.extend(_cmu_mosei_public_main_rebuild_commands(download_root))
        return
    stage_dir = download_root / "cmu_mosei_stage_inputs"
    if not phases["splits"]["ok"]:
        commands.append(f"python scripts/multimodal/write_cmu_sdk_splits.py cmu_mosei {download_root / 'cmu_mosei_splits.json'}")
    if phases["splits"]["ok"] and not phases["sequence_inspection"]["ok"]:
        commands.append(
            "python scripts/multimodal/inspect_cmu_sdk_sequences.py cmu_mosei "
            f"{download_root / 'cmu_sdk' / 'cmu_mosei'} "
            f"--splits {download_root / 'cmu_mosei_splits.json'} "
            f"--stage-output-dir {stage_dir} "
            "--temporal-policy mean "
            "--preprocessing-version cmu-mosei-cmu-sdk-mean-v0.1 "
            f"--output {download_root / 'cmu_mosei_sequence_inspection.json'}"
        )
    if phases["sequence_inspection"]["ok"] and not phases["stage_inputs"]["ok"]:
        commands.append(
            "python scripts/multimodal/extract_cmu_sdk_stage_inputs.py cmu_mosei "
            f"{stage_dir} "
            f"--splits {download_root / 'cmu_mosei_splits.json'} "
            "--text-sequence data/raw_multimodal/_downloads/cmu_sdk/cmu_mosei/replace_with_text_feature.csd "
            "--audio-sequence data/raw_multimodal/_downloads/cmu_sdk/cmu_mosei/replace_with_audio_feature.csd "
            "--visual-sequence data/raw_multimodal/_downloads/cmu_sdk/cmu_mosei/replace_with_visual_feature.csd "
            "--label-sequence data/raw_multimodal/_downloads/cmu_sdk/cmu_mosei/replace_with_labels.csd "
            "--temporal-policy mean "
            "--sentiment-column 0 "
            "--emotion-columns 1: "
            "--license-tag cmu-multimodal-sdk "
            "--preprocessing-version cmu-mosei-cmu-sdk-mean-v0.1"
        )
    if phases["stage_inputs"]["ok"] and not phases["raw_manifest"]["ok"]:
        commands.append(
            "python scripts/multimodal/stage_cmu_sentiment_raw.py cmu_mosei "
            f"{raw_root} "
            f"--splits {stage_dir / 'cmu_mosei_splits.json'} "
            f"--text-features {stage_dir / 'cmu_mosei_text_features.npy'} "
            f"--audio-features {stage_dir / 'cmu_mosei_audio_features.npy'} "
            f"--visual-features {stage_dir / 'cmu_mosei_visual_features.npy'} "
            f"--sentiment-labels {stage_dir / 'cmu_mosei_sentiment.npy'} "
            f"--emotion-labels {stage_dir / 'cmu_mosei_emotion.npy'} "
            "--feature-version text=cmu-mosei-cmu-sdk-mean-v0.1:text "
            "--feature-version audio=cmu-mosei-cmu-sdk-mean-v0.1:audio "
            "--feature-version vision=cmu-mosei-cmu-sdk-mean-v0.1:vision "
            "--license-tag cmu-multimodal-sdk "
            "--preprocessing-version cmu-mosei-cmu-sdk-mean-v0.1"
        )
    _append_cache_or_acceptance_command(commands, "cmu_mosei", raw_root, cache_root, controlled_report, phases)


def _cmu_mosei_public_main_rebuild_commands(download_root: Path) -> list[str]:
    stage_dir = download_root / "cmu_mosei_stage_inputs"
    sdk_dir = download_root / "cmu_sdk" / "cmu_mosei"
    split_path = download_root / "cmu_mosei_splits.json"
    return [
        (
            "python scripts/multimodal/write_cmu_sdk_splits.py cmu_mosei "
            f"{split_path} "
            f"--expand-with-sequence {sdk_dir / 'replace_with_utterance_or_segment_sequence.csd'}"
        ),
        (
            "python scripts/multimodal/inspect_cmu_sdk_sequences.py cmu_mosei "
            f"{sdk_dir} "
            f"--splits {split_path} "
            f"--stage-output-dir {stage_dir} "
            "--temporal-policy strict "
            "--preprocessing-version cmu-mosei-cmu-sdk-utterance-v0.1 "
            f"--output {download_root / 'cmu_mosei_sequence_inspection.json'}"
        ),
    ]


def _append_cache_or_acceptance_command(
    commands: list[str],
    dataset_name: str,
    raw_root: Path,
    cache_root: Path,
    controlled_report: Path,
    phases: dict[str, dict[str, Any]],
) -> None:
    if phases["raw_manifest"]["ok"] and not phases["cache"]["ok"]:
        commands.append(
            f"python scripts/multimodal/build_cache.py {dataset_name} {raw_root} {cache_root} --version v0.1"
        )
    if phases["cache"]["ok"]:
        commands.append(_acceptance_command(dataset_name, raw_root, cache_root, controlled_report))


def _bootstrap_commands(datasets: list[str], download_root: Path) -> list[str]:
    output_script = "/tmp/ovha_public_downloads.sh"
    return [
        (
            "python scripts/multimodal/bootstrap_public_downloads.py "
            f"--datasets {' '.join(datasets)} "
            "--repo-root \"$PWD\" "
            f"--download-root {download_root} "
            "--include-system-packages "
            "--use-hf-mirror "
            f"--output {output_script}"
        ),
        f"bash -n {output_script}",
        f"bash {output_script}",
    ]


def _acceptance_command(dataset_name: str, raw_root: Path, cache_root: Path, controlled_report: Path) -> str:
    config = CONFIG_BY_DATASET[dataset_name]
    artifact_root = ACCEPTANCE_ARTIFACT_BY_DATASET[dataset_name]
    return (
        f"python scripts/multimodal/accept_public_data.py {config} "
        f"--raw-root {raw_root} "
        f"--cache-root {cache_root} "
        f"--controlled-report {controlled_report} "
        "--train-smoke-steps 1 "
        "--train-baseline-smoke-steps 1 "
        "--train-all-config-seeds "
        "--train-split train "
        "--eval-smoke-split val "
        f"--artifact-root {artifact_root}"
    )


def _sequence_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in ("*.csd", "*.h5", "*.hdf5", "*.json"):
        files.extend(sorted(root.rglob(pattern)))
    return sorted(path for path in files if path.is_file())


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


if __name__ == "__main__":
    raise SystemExit(main())
