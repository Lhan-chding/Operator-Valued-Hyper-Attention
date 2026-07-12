#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMDET_COMMIT="cfd5d3a985b0249de009b67d04f37263e11cdf3d"
MMDET_REPOSITORY="https://github.com/open-mmlab/mmdetection.git"

VENV_DIR="${PROJECT_DIR}/.venv-server"
MMDET_DIR="${PROJECT_DIR}/.deps/mmdetection"
PYTHON_BIN="python3"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"

usage() {
  printf '%s\n' \
    "Usage: $0 [options]" \
    "" \
    "Options:" \
    "  --venv PATH             Virtual environment path." \
    "  --mmdet-dir PATH        Pinned MMDetection checkout path." \
    "  --python PATH           Bootstrap Python executable." \
    "  --torch-index-url URL   PyTorch CUDA wheel index." \
    "  --torch-version VER     PyTorch version without CUDA suffix." \
    "  --torchvision-version VER" \
    "  -h, --help              Show this help." \
    "" \
    "Defaults target the locked CUDA 12.1 environment. Override the index only" \
    "when the server driver requires another official PyTorch CUDA build."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      [[ $# -ge 2 ]] || { printf 'missing value for --venv\n' >&2; exit 2; }
      VENV_DIR="$2"
      shift 2
      ;;
    --mmdet-dir)
      [[ $# -ge 2 ]] || { printf 'missing value for --mmdet-dir\n' >&2; exit 2; }
      MMDET_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { printf 'missing value for --python\n' >&2; exit 2; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --torch-index-url)
      [[ $# -ge 2 ]] || { printf 'missing value for --torch-index-url\n' >&2; exit 2; }
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --torch-version)
      [[ $# -ge 2 ]] || { printf 'missing value for --torch-version\n' >&2; exit 2; }
      TORCH_VERSION="$2"
      shift 2
      ;;
    --torchvision-version)
      [[ $# -ge 2 ]] || { printf 'missing value for --torchvision-version\n' >&2; exit 2; }
      TORCHVISION_VERSION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
}

case "${TORCH_INDEX_URL}" in
  https://download.pytorch.org/whl/cu118|https://download.pytorch.org/whl/cu124|https://download.pytorch.org/whl/cu126) ;;
  *) printf 'torch index must be an approved official PyTorch CUDA index\n' >&2; exit 2 ;;
esac

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_MIM="${VENV_DIR}/bin/mim"

"${VENV_PYTHON}" -m pip install --upgrade 'pip==24.0' 'setuptools==69.2.0' 'wheel==0.43.0'
"${VENV_PYTHON}" -m pip install \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
"${VENV_PYTHON}" -m pip install -r "${PROJECT_DIR}/environment/requirements.txt"
"${VENV_MIM}" install 'mmcv==2.1.0'

if [[ -e "${MMDET_DIR}" && ! -d "${MMDET_DIR}/.git" ]]; then
  printf 'Refusing to overwrite non-repository path: %s\n' "${MMDET_DIR}" >&2
  exit 2
fi

if [[ ! -d "${MMDET_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${MMDET_DIR}")"
  git clone "${MMDET_REPOSITORY}" "${MMDET_DIR}"
fi

MMDET_ORIGIN="$(git -C "${MMDET_DIR}" remote get-url origin)"
if [[ "${MMDET_ORIGIN}" != "${MMDET_REPOSITORY}" && "${MMDET_ORIGIN}" != "git@github.com:open-mmlab/mmdetection.git" ]]; then
  printf 'Unexpected MMDetection origin; only the official credential-free remote is allowed\n' >&2
  exit 2
fi

git -C "${MMDET_DIR}" fetch --depth 1 origin "${MMDET_COMMIT}"
git -C "${MMDET_DIR}" checkout --detach "${MMDET_COMMIT}"
ACTUAL_COMMIT="$(git -C "${MMDET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${MMDET_COMMIT}" ]]; then
  printf 'MMDetection commit mismatch: expected %s, got %s\n' \
    "${MMDET_COMMIT}" "${ACTUAL_COMMIT}" >&2
  exit 2
fi

EXPECTED_TREE="e389bc213f4772c481a0b5f81f0d1786c0b79e66"
ACTUAL_TREE="$(git -C "${MMDET_DIR}" rev-parse HEAD^{tree})"
if [[ "${ACTUAL_TREE}" != "${EXPECTED_TREE}" ]]; then
  printf 'MMDetection tree mismatch: expected %s, got %s\n' "${EXPECTED_TREE}" "${ACTUAL_TREE}" >&2
  exit 2
fi
DIRTY_STATE="$(git -C "${MMDET_DIR}" status --porcelain --untracked-files=all)"
if [[ -n "${DIRTY_STATE}" ]]; then
  printf 'MMDetection checkout must be clean before installation\n%s\n' "${DIRTY_STATE}" >&2
  exit 2
fi

"${VENV_PYTHON}" -m pip install --no-deps -e "${MMDET_DIR}"
"${VENV_PYTHON}" -m pip check

PYTHONPATH="${PROJECT_DIR}:${MMDET_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${VENV_PYTHON}" -c \
  'import mmcv, mmdet, mmengine, torch; from mmcv.ops import MultiScaleDeformableAttention; print({"torch": torch.__version__, "mmcv": mmcv.__version__, "mmengine": mmengine.__version__, "mmdet": mmdet.__version__})'

printf '\nEnvironment ready. Activate with:\n  source %s/bin/activate\n' "${VENV_DIR}"
printf 'Export project paths with:\n  export PYTHONPATH=%s:%s${PYTHONPATH:+:$PYTHONPATH}\n' \
  "${PROJECT_DIR}" "${MMDET_DIR}"
