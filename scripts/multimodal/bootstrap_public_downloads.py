#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
from typing import Callable


SUPPORTED_DATASETS = (
    "refcoco",
    "refcoco_plus",
    "refcocog",
    "flickr30k_entities",
    "visual_genome_metadata",
    "visual_genome_images",
    "cmu_mosei",
    "cmu_mosi",
    "meld",
)
DEFAULT_DATASETS = ("refcoco", "cmu_mosei")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an Ubuntu download bootstrap script for the public multimodal "
            "datasets used by the OVHA top-conference mainline."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--download-root", default="data/raw_multimodal/_downloads")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--use-hf-mirror", action="store_true")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--include-system-packages", action="store_true")
    parser.add_argument("--skip-pip-packages", action="store_true")
    args = parser.parse_args()

    datasets = _normalize_datasets(args.datasets)
    script = build_script(
        datasets=datasets,
        repo_root=args.repo_root,
        download_root=args.download_root,
        include_system_packages=bool(args.include_system_packages),
        include_pip_packages=not bool(args.skip_pip_packages),
        hf_endpoint=args.hf_endpoint if args.use_hf_mirror else None,
    )

    output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(script)
        current_mode = output.stat().st_mode
        output.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    payload = {
        "ok": True,
        "mode": "ubuntu_public_multimodal_download_bootstrap",
        "policy": "script generation only; run the emitted shell script on the Ubuntu data host",
        "datasets": list(datasets),
        "repo_root": args.repo_root,
        "download_root": args.download_root,
        "output": str(output) if output is not None else None,
        "run": f"bash {output}" if output is not None else None,
    }
    if output is None:
        payload["script"] = script
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_script(
    *,
    datasets: tuple[str, ...],
    repo_root: str,
    download_root: str,
    include_system_packages: bool,
    include_pip_packages: bool,
    hf_endpoint: str | None,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"REPO_ROOT={_bash_value(repo_root)}",
        f"DOWNLOAD_ROOT={_bash_value(download_root)}",
        'cd "$REPO_ROOT"',
        "",
        'if [ -d ".venv" ]; then',
        '  source ".venv/bin/activate"',
        "fi",
        "",
        'export HF_HOME="$REPO_ROOT/.hf_cache"',
        'export HF_HUB_CACHE="$HF_HOME/hub"',
        "export HF_XET_HIGH_PERFORMANCE=1",
        "export HF_HUB_DOWNLOAD_TIMEOUT=60",
        "export HF_HUB_ETAG_TIMEOUT=30",
        'export OVHA_DOWNLOAD_ROOT="$DOWNLOAD_ROOT"',
    ]
    if hf_endpoint:
        lines.append(f"export HF_ENDPOINT={_bash_env_value(hf_endpoint)}")
    lines.extend(
        [
            "",
            "python3 - <<'PY'",
            "import sys",
            "try:",
            "    import torch",
            "except Exception as exc:",
            "    print('torch_import_error', repr(exc))",
            "else:",
            "    print('python', sys.executable)",
            "    print('torch', torch.__version__)",
            "    print('cuda_available', torch.cuda.is_available())",
            "PY",
            "",
            'df -h "$REPO_ROOT"',
            'mkdir -p "$DOWNLOAD_ROOT"',
        ]
    )

    if include_system_packages:
        lines.extend(
            [
                "",
                "sudo NEEDRESTART_MODE=l DEBIAN_FRONTEND=noninteractive apt-get update",
                (
                    "sudo NEEDRESTART_MODE=l DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "aria2 git git-lfs unzip p7zip-full pigz zstd ffmpeg libsndfile1 rsync"
                ),
                "git lfs install",
            ]
        )

    if include_pip_packages:
        lines.extend(
            [
                "",
                "python3 -m pip install -U pip setuptools wheel",
                (
                    "python3 -m pip install -U "
                    "huggingface_hub hf_xet datasets kaggle gdown requests tqdm "
                    "pandas pyarrow fastparquet h5py scipy numpy pillow opencv-python-headless "
                    "soundfile librosa"
                ),
            ]
        )

    emitters: dict[str, Callable[[list[str]], None]] = {
        "refcoco": _emit_refcoco,
        "refcoco_plus": _emit_refcoco_plus,
        "refcocog": _emit_refcocog,
        "flickr30k_entities": _emit_flickr30k_entities,
        "visual_genome_metadata": _emit_visual_genome_metadata,
        "visual_genome_images": _emit_visual_genome_images,
        "cmu_mosei": lambda target: _emit_cmu_sdk(target, ("cmu_mosei",)),
        "cmu_mosi": lambda target: _emit_cmu_sdk(target, ("cmu_mosi",)),
        "meld": _emit_meld,
    }
    for dataset in datasets:
        lines.append("")
        lines.append(f"# {dataset}")
        emitters[dataset](lines)

    lines.extend(
        [
            "",
            "echo 'Download bootstrap complete.'",
            "echo 'Next: convert downloaded archives/features into data/raw_multimodal/<dataset> raw manifests.'",
            "echo 'Then run scripts/multimodal/build_cache.py and scripts/multimodal/validate_cache.py.'",
            "",
        ]
    )
    return "\n".join(lines)


def _normalize_datasets(values: list[str]) -> tuple[str, ...]:
    normalized = []
    for value in values:
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _bash_value(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _bash_env_value(value: str) -> str:
    if all(character.isalnum() or character in ".:/_+-" for character in value):
        return value
    return _bash_value(value)


def _emit_refcoco(lines: list[str]) -> None:
    _emit_coco2014_core(lines)
    lines.extend(
        _aria2_with_fallback(
            "$DOWNLOAD_ROOT/refcoco",
            "refcoco.zip",
            "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
            "https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
        )
    )
    _emit_refcoco_unpack(lines, ("refcoco.zip",))


def _emit_refcoco_plus(lines: list[str]) -> None:
    _emit_coco2014_core(lines)
    lines.extend(
        _aria2("$DOWNLOAD_ROOT/refcoco", "refcoco_plus.zip", "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip")
    )
    _emit_refcoco_unpack(lines, ("refcoco_plus.zip",))


def _emit_refcocog(lines: list[str]) -> None:
    _emit_coco2014_core(lines)
    lines.extend(
        _aria2("$DOWNLOAD_ROOT/refcoco", "refcocog.zip", "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip")
    )
    _emit_refcoco_unpack(lines, ("refcocog.zip",))


def _emit_coco2014_core(lines: list[str]) -> None:
    lines.extend(
        [
            'mkdir -p "$DOWNLOAD_ROOT/refcoco"',
            *_aria2("$DOWNLOAD_ROOT/refcoco", "train2014.zip", "http://images.cocodataset.org/zips/train2014.zip"),
            *_aria2(
                "$DOWNLOAD_ROOT/refcoco",
                "annotations_trainval2014.zip",
                "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            ),
        ]
    )


def _emit_refcoco_unpack(lines: list[str], archives: tuple[str, ...]) -> None:
    archive_list = " ".join(("annotations_trainval2014.zip", *archives))
    lines.extend(
        [
            'mkdir -p "$DOWNLOAD_ROOT/refcoco/extracted"',
            f"for archive in {archive_list}; do",
            '  if [ -f "$DOWNLOAD_ROOT/refcoco/$archive" ]; then',
            '    unzip -n -q "$DOWNLOAD_ROOT/refcoco/$archive" -d "$DOWNLOAD_ROOT/refcoco/extracted"',
            "  fi",
            "done",
        ]
    )


def _emit_cmu_sdk(lines: list[str], datasets: tuple[str, ...]) -> None:
    targets = ", ".join(f'"{name}": mmdatasdk.{name}' for name in datasets)
    lines.extend(
        [
            "python3 -m pip install -U git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git",
            'mkdir -p "$DOWNLOAD_ROOT/cmu_sdk"',
            "python3 - <<'PY'",
            "from pathlib import Path",
            "import os",
            "import mmdatasdk",
            "",
            f"targets = {{{targets}}}",
            "root = Path(os.environ['OVHA_DOWNLOAD_ROOT']) / 'cmu_sdk'",
            "for name, spec in targets.items():",
            "    out = root / name",
            "    out.mkdir(parents=True, exist_ok=True)",
            "    recipe = {}",
            "    for attr in ('highlevel', 'labels'):",
            "        value = getattr(spec, attr, {})",
            "        if isinstance(value, dict):",
            "            recipe.update(value)",
            "    print(f'Downloading {name} sequences -> {out}')",
            "    mmdatasdk.mmdataset(recipe, str(out))",
            "PY",
        ]
    )


def _emit_flickr30k_entities(lines: list[str]) -> None:
    lines.extend(
        [
            'mkdir -p "$DOWNLOAD_ROOT/flickr30k_entities"',
            'if [ ! -d "$DOWNLOAD_ROOT/flickr30k_entities/annotations_repo/.git" ]; then',
            (
                "  git clone --depth 1 https://github.com/BryanPlummer/flickr30k_entities.git "
                '"$DOWNLOAD_ROOT/flickr30k_entities/annotations_repo"'
            ),
            "fi",
            (
                "hf download cjc/flickr30k --repo-type dataset "
                '--local-dir "$DOWNLOAD_ROOT/flickr30k_entities/hf_flickr30k" '
                "--local-dir-use-symlinks False"
            ),
        ]
    )


def _emit_visual_genome_metadata(lines: list[str]) -> None:
    lines.extend(
        [
            'mkdir -p "$DOWNLOAD_ROOT/visual_genome"',
            *_aria2(
                "$DOWNLOAD_ROOT/visual_genome",
                "image_data.json.zip",
                "https://visualgenome.org/static/data/dataset/image_data.json.zip",
            ),
            *_aria2(
                "$DOWNLOAD_ROOT/visual_genome",
                "region_descriptions.json.zip",
                "https://visualgenome.org/static/data/dataset/region_descriptions.json.zip",
            ),
        ]
    )


def _emit_visual_genome_images(lines: list[str]) -> None:
    _emit_visual_genome_metadata(lines)
    lines.extend(
        [
            *_aria2("$DOWNLOAD_ROOT/visual_genome", "images.zip", "https://visualgenome.org/static/data/dataset/images.zip"),
            *_aria2("$DOWNLOAD_ROOT/visual_genome", "images2.zip", "https://visualgenome.org/static/data/dataset/images2.zip"),
        ]
    )


def _emit_meld(lines: list[str]) -> None:
    lines.extend(
        [
            'mkdir -p "$DOWNLOAD_ROOT/meld"',
            *_aria2(
                "$DOWNLOAD_ROOT/meld",
                "MELD.Raw.tar.gz",
                "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz",
            ),
            'mkdir -p "$DOWNLOAD_ROOT/meld/extracted"',
            'tar -xzf "$DOWNLOAD_ROOT/meld/MELD.Raw.tar.gz" -C "$DOWNLOAD_ROOT/meld/extracted"',
        ]
    )


def _aria2(directory: str, output: str, url: str) -> list[str]:
    return [
        "aria2c -c -x16 -s16 -k1M \\",
        f"  -d {_shell_path(directory)} \\",
        f"  -o {_bash_value(output)} \\",
        f"  {_bash_value(url)}",
    ]


def _aria2_with_fallback(directory: str, output: str, primary_url: str, fallback_url: str) -> list[str]:
    primary = _aria2(directory, output, primary_url)
    fallback = _aria2(directory, output, fallback_url)
    primary[-1] = primary[-1] + " || \\"
    return [*primary, *("  " + line for line in fallback)]


def _shell_path(value: str) -> str:
    if value.startswith("$"):
        return f'"{value}"'
    return _bash_value(value)


if __name__ == "__main__":
    raise SystemExit(main())
