#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/minivllm-work}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR="$WORK_ROOT/cache/pip-deepspec"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK_ROOT/cache/uv}"
export TMPDIR="$WORK_ROOT/cache/tmp"
export PYTHONDONTWRITEBYTECODE=1
PACKAGE_INDEX="${DEEPSPEC_PACKAGE_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# 单一镜像解析，避免为普通依赖访问跨境 PyTorch 源。
unset UV_INDEX UV_EXTRA_INDEX_URL UV_INDEX_URL UV_DEFAULT_INDEX UV_TORCH_BACKEND PIP_EXTRA_INDEX_URL
mkdir -p "$TMPDIR"
SOURCE="$WORK_ROOT/third_party/DeepSpec"
ENVIRONMENT="$WORK_ROOT/venv-deepspec"
PYTHON_BIN="${PYTHON_BIN:-python}"
REUSE_SYSTEM_TORCH="${DEEPSPEC_REUSE_SYSTEM_TORCH:-0}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11, or set PYTHON_BIN to its executable path"'
PIN="$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, sys.argv[1]); from protocol import DEEPSPEC_COMMIT; print(DEEPSPEC_COMMIT)" "$HERE")"
mkdir -p "$WORK_ROOT/third_party"
if [ ! -e "$SOURCE" ]; then
    git clone https://github.com/deepseek-ai/DeepSpec.git "$SOURCE"
    git -C "$SOURCE" checkout --detach "$PIN"
fi
"$PYTHON_BIN" -c "import sys; sys.path.insert(0, sys.argv[1]); from protocol import source_fingerprint; source_fingerprint(sys.argv[2])" "$HERE" "$SOURCE"
if [ "$REUSE_SYSTEM_TORCH" = 1 ]; then
    "$PYTHON_BIN" - <<'PY'
import torch
import triton
assert torch.__version__.split('+')[0] == '2.9.1', 'Base Python must provide PyTorch 2.9.1'
assert torch.version.cuda == '12.8', 'Base Python must provide CUDA 12.8'
assert triton.__version__ == '3.5.1', 'Base Python must provide Triton 3.5.1'
print('Reusing preinstalled PyTorch:', torch.__file__, flush=True)
PY
    # 保留环境内已有包，使基础环境的 GPU 依赖对本实验可见。
    "$PYTHON_BIN" -m venv --without-pip --system-site-packages "$ENVIRONMENT"
fi
if [ ! -x "$ENVIRONMENT/bin/python" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$PYTHON_BIN" "$ENVIRONMENT"
    else
        # 部分云镜像的 ensurepip 不可用，由外部安装器补齐 pip。
        "$PYTHON_BIN" -m venv --without-pip "$ENVIRONMENT"
    fi
fi
# 独立环境避免升级现有 Mini-vLLM 的 Transformers 4.x。
if [ "$REUSE_SYSTEM_TORCH" = 1 ]; then
    # pip 识别继承环境中已经满足的 torch/triton，不重新从镜像解析它们。
    "$PYTHON_BIN" -m pip --isolated --python "$ENVIRONMENT/bin/python" install \
        --index-url "$PACKAGE_INDEX" --cache-dir "$PIP_CACHE_DIR" \
        -r "$SOURCE/requirements.txt" huggingface-hub pip
elif command -v uv >/dev/null 2>&1; then
    uv --no-config pip install --python "$ENVIRONMENT/bin/python" \
        --index-url "$PACKAGE_INDEX" \
        -r "$SOURCE/requirements.txt" huggingface-hub pip
else
    # pip >=22.3 可管理尚未安装 pip 的另一个 Python 环境。
    "$PYTHON_BIN" -m pip --isolated --python "$ENVIRONMENT/bin/python" install \
        --index-url "$PACKAGE_INDEX" --cache-dir "$PIP_CACHE_DIR" \
        -r "$SOURCE/requirements.txt" huggingface-hub pip
fi
"$ENVIRONMENT/bin/python" -m pip check
"$ENVIRONMENT/bin/python" - <<'PY'
import torch
print('Installed PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda, flush=True)
assert torch.__version__.split('+')[0] == '2.9.1', 'Unexpected PyTorch version'
assert torch.version.cuda == '12.8', 'Expected CUDA 12.8; inspect the installed wheel before benchmarking'
assert torch.cuda.is_available(), 'GPU is unavailable'
PY
echo "READY: $ENVIRONMENT/bin/python"
echo "DeepSpec: $SOURCE"
echo "GPU experiments have not run."
