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
        "ok": bool(cache["ok"]),
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
    stage_inputs = _path_phase(
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
    if phases["cache"]["ok"]:
        return [_acceptance_command(dataset_name, raw_root, cache_root, controlled_report)]
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
        commands.append(
            "python scripts/multimodal/stage_refcoco_raw.py refcoco "
            f"{raw_root} "
            f"--splits {stage_dir / 'refcoco_splits.json'} "
            f"--records {stage_dir / 'refcoco_phrase_region_records.json'} "
            f"--text-features {stage_dir / 'refcoco_text_features.npy'} "
            f"--region-features {stage_dir / 'refcoco_region_features.npy'} "
            "--license-tag refcoco-coco2014 "
            "--preprocessing-version refcoco-frozen-features-v0.1"
        )
    _append_cache_or_acceptance_command(commands, "refcoco", raw_root, cache_root, controlled_report, phases)


def _append_cmu_mosei_commands(
    commands: list[str],
    phases: dict[str, dict[str, Any]],
    download_root: Path,
    raw_root: Path,
    cache_root: Path,
    controlled_report: Path,
) -> None:
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
