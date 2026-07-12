#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMDET_COMMIT="cfd5d3a985b0249de009b67d04f37263e11cdf3d"
MMDET_REPOSITORY="https://github.com/open-mmlab/mmdetection.git"
ENVIRONMENT_PROFILE="cu121-wheel"

VENV_DIR="${PROJECT_DIR}/.venv-server"
MMDET_DIR="${PROJECT_DIR}/.deps/mmdetection"
PYTHON_BIN="python3"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
TORCH_VERSION="2.1.0"
TORCHVISION_VERSION="0.16.0"
MMCV_WHEEL_URL="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl"
MMCV_WHEEL_SHA256="a95a64e12fa88c568d4384e6d5f42230e9e407e2b052f0b8da2ef859dc079b53"

usage() {
  printf '%s\n' \
    "Usage: $0 [options]" \
    "" \
    "Options:" \
    "  --venv PATH             Virtual environment path." \
    "  --mmdet-dir PATH        Pinned MMDetection checkout path." \
    "  --python PATH           Bootstrap Python executable." \
    "  -h, --help              Show this help." \
    "" \
    "Locked profile: cu121-wheel (Python 3.10, Linux x86_64, PyTorch 2.1.0," \
    "torchvision 0.16.0, official binary MMCV 2.1.0). No nvcc is invoked."
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

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
  printf 'The locked cu121-wheel profile requires Linux x86_64\n' >&2
  exit 2
}
PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_VERSION}" == "3.10" ]] || {
  printf 'The locked cu121-wheel profile requires Python 3.10; got %s\n' "${PYTHON_VERSION}" >&2
  exit 2
}

[[ -n "${HOME:-}" ]] || { printf 'HOME is required\n' >&2; exit 2; }
HOME_REAL="$(realpath -e "${HOME}")"
for target_parent in "$(dirname "${VENV_DIR}")" "$(dirname "${MMDET_DIR}")"; do
  PARENT_REAL="$(realpath -e "${target_parent}")" || {
    printf 'Target parent must exist before setup: %s\n' "${target_parent}" >&2
    exit 2
  }
  [[ "${PARENT_REAL}" == "${HOME_REAL}" || "${PARENT_REAL}" == "${HOME_REAL}/"* ]] || {
    printf 'Target parent must stay under HOME: %s\n' "${PARENT_REAL}" >&2
    exit 2
  }
  PARENT_OWNER="$(stat -c %u "${PARENT_REAL}")"
  PARENT_MODE="$(stat -c %a "${PARENT_REAL}")"
  if [[ "${PARENT_OWNER}" != "$(id -u)" ]] || (( (8#${PARENT_MODE} & 022) != 0 )); then
    printf 'Target parent must be user-owned and not group/other writable: %s\n' \
      "${PARENT_REAL}" >&2
    exit 2
  fi
done

if [[ -e "${VENV_DIR}" || -L "${VENV_DIR}" ]]; then
  printf 'Refusing to modify an existing or symlinked venv: %s\n' "${VENV_DIR}" >&2
  exit 2
fi
if [[ -e "${MMDET_DIR}" || -L "${MMDET_DIR}" ]]; then
  printf 'Refusing to reuse an existing MMDetection destination: %s\n' "${MMDET_DIR}" >&2
  exit 2
fi
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
VENV_PYTHON="${VENV_DIR}/bin/python"

"${VENV_PYTHON}" -m pip install --upgrade 'pip==24.0' 'setuptools==69.2.0' 'wheel==0.43.0'
"${VENV_PYTHON}" -m pip install \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --only-binary=:all: \
  --index-url "${TORCH_INDEX_URL}"
"${VENV_PYTHON}" -m pip install -r "${PROJECT_DIR}/environment/requirements.txt"
"${VENV_PYTHON}" -m pip install --no-deps --only-binary=:all: \
  "${MMCV_WHEEL_URL}#sha256=${MMCV_WHEEL_SHA256}"

mkdir -p "$(dirname "${MMDET_DIR}")"
git clone "${MMDET_REPOSITORY}" "${MMDET_DIR}"

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
  'import mmcv, mmdet, mmengine, torch, torchvision; from mmcv.ops import MultiScaleDeformableAttention; expected={"torch":"2.1.0", "torchvision":"0.16.0", "mmcv":"2.1.0", "mmengine":"0.10.3"}; observed={"torch":torch.__version__.split("+", 1)[0], "torchvision":torchvision.__version__.split("+", 1)[0], "mmcv":mmcv.__version__, "mmengine":mmengine.__version__}; assert observed == expected, (observed, expected); assert torch.version.cuda == "12.1", torch.version.cuda; print({**observed, "torch_cuda":torch.version.cuda, "mmdet":mmdet.__version__, "profile":"cu121-wheel"})'

MANIFEST="${VENV_DIR}/ovha_rod_environment.txt"
{
  printf 'profile=%s\n' "${ENVIRONMENT_PROFILE}"
  printf 'mmcv_wheel_url=%s\n' "${MMCV_WHEEL_URL}"
  printf 'mmcv_wheel_sha256=%s\n' "${MMCV_WHEEL_SHA256}"
  printf 'mmdetection_origin=%s\n' "${MMDET_ORIGIN}"
  printf 'mmdetection_commit=%s\n' "${ACTUAL_COMMIT}"
  printf 'mmdetection_tree=%s\n' "${ACTUAL_TREE}"
  "${VENV_PYTHON}" -m pip freeze --all
} > "${MANIFEST}"
chmod 600 "${MANIFEST}"

printf '\nEnvironment ready. Activate with:\n  source %s/bin/activate\n' "${VENV_DIR}"
printf 'Export project paths with:\n  export PYTHONPATH=%s:%s${PYTHONPATH:+:$PYTHONPATH}\n' \
  "${PROJECT_DIR}" "${MMDET_DIR}"
