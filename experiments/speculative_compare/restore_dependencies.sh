#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/hy-tmp/minivllm-work}"
export PYTHONDONTWRITEBYTECODE=1
BASE_PYTHON="${BASE_PYTHON:-/usr/local/bin/python}"
VENV_PYTHON="$WORK_ROOT/venv-deepspec/bin/python"
INDEX="${RECOVERY_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
test -x "$VENV_PYTHON"
mkdir -p "$WORK_ROOT/cache/pip-recovery" "$WORK_ROOT/cache/tmp" "$WORK_ROOT/recovery-environment"
export TMPDIR="$WORK_ROOT/cache/tmp"
"$VENV_PYTHON" - <<'PY'
import sys, torch, triton
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split('+')[0] == '2.9.1'
assert torch.version.cuda == '12.8' and triton.__version__ == '3.5.1'
print('REUSE GPU STACK:', torch.__version__, torch.version.cuda, triton.__version__, flush=True)
PY
PLAN="$WORK_ROOT/recovery-environment/pip-plan.json"
LOCKED="$WORK_ROOT/recovery-environment/python-wheels.txt"
# 先解析轻量 Python 依赖，拒绝任何替换 GPU 软件栈的安装计划。
"$BASE_PYTHON" -m pip --isolated --disable-pip-version-check --python "$VENV_PYTHON" install \
    --dry-run --report "$PLAN" --index-url "$INDEX" --only-binary=:all: \
    --cache-dir "$WORK_ROOT/cache/pip-recovery" --timeout 30 --retries 2 \
    -r "$HERE/requirements-recovery.txt" -c "$HERE/constraints-recovery.txt"
"$BASE_PYTHON" - "$PLAN" "$LOCKED" <<'PY'
import json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
rows = []
for item in plan['install']:
    name = item['metadata']['name'].lower().replace('_', '-')
    if name in ('torch','triton','torchvision','torchaudio','pytorch-triton') or name.startswith('nvidia-'):
        raise RuntimeError('GPU package replacement blocked: ' + name)
    info = item['download_info']
    sha = info['archive_info']['hashes']['sha256']
    rows.append(f"{name} @ {info['url']} --hash=sha256:{sha}")
Path(sys.argv[2]).write_text('\n'.join(rows) + '\n')
print('LOCKED PYTHON WHEELS:', len(rows), '; PyTorch/CUDA/Triton replacement blocked', flush=True)
PY
"$BASE_PYTHON" -m pip --isolated --disable-pip-version-check --python "$VENV_PYTHON" install \
    --no-deps --require-hashes --cache-dir "$WORK_ROOT/cache/pip-recovery" \
    --timeout 30 --retries 2 -r "$LOCKED"
"$VENV_PYTHON" - "$WORK_ROOT/third_party/DeepSpec" <<'PY'
import sys
import torch, transformers, triton, numpy
sys.path.insert(0, sys.argv[1])
from deepspec.eval.dspark import Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Qwen3Eagle3Evaluator
assert torch.__version__.split('+')[0] == '2.9.1' and torch.version.cuda == '12.8'
assert transformers.__version__ == '5.10.2' and triton.__version__ == '3.5.1'
print('DEPENDENCIES READY:', torch.__version__, transformers.__version__, triton.__version__, numpy.__version__, flush=True)
print('Official evaluator imports passed; no model loaded, no GPU benchmark.', flush=True)
PY
