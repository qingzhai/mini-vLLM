#!/usr/bin/env bash
set -euo pipefail
export WORK_ROOT="${WORK_ROOT:-/hy-tmp/minivllm-work}"
BASE_PYTHON="${BASE_PYTHON:-/usr/local/bin/python}"
VENV="$WORK_ROOT/venv-sharegpt"
INDEX="${SHAREGPT_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
mkdir -p "$WORK_ROOT/cache/tmp" "$WORK_ROOT/cache/pip-sharegpt" "$WORK_ROOT/sharegpt-environment"
export TMPDIR="$WORK_ROOT/cache/tmp"
# Reuse the image's GPU stack; never resolve Torch/CUDA through a package mirror.
"$BASE_PYTHON" - <<'PY'
import sys, torch, triton
assert sys.version_info[:2] == (3, 11), 'Use a Python 3.11 image'
assert torch.__version__.split('+')[0] == '2.9.1' and torch.version.cuda == '12.8'
assert triton.__version__ == '3.5.1'
print('REUSE GPU STACK:', torch.__version__, torch.version.cuda, triton.__version__)
PY
if [ ! -x "$VENV/bin/python" ]; then
    "$BASE_PYTHON" -m venv --system-site-packages --without-pip "$VENV"
fi
PLAN="$WORK_ROOT/sharegpt-environment/pip-plan.json"
LOCKED="$WORK_ROOT/sharegpt-environment/wheels.txt"
"$BASE_PYTHON" -m pip --isolated --disable-pip-version-check --python "$VENV/bin/python" install \
    --dry-run --report "$PLAN" --index-url "$INDEX" --only-binary=:all: \
    --cache-dir "$WORK_ROOT/cache/pip-sharegpt" --timeout 30 --retries 2 \
    'transformers==4.57.3' 'huggingface-hub==0.36.0' 'safetensors==0.8.0' \
    'numpy==2.2.6' 'xxhash==4.0.1'
"$BASE_PYTHON" - "$PLAN" "$LOCKED" <<'PY'
import json, sys
from pathlib import Path
rows = []
for item in json.loads(Path(sys.argv[1]).read_text())['install']:
    name = item['metadata']['name'].lower().replace('_', '-')
    if name in ('torch','triton','torchvision','torchaudio','pytorch-triton') or name.startswith('nvidia-'):
        raise RuntimeError('GPU dependency replacement blocked: ' + name)
    info = item['download_info']
    digest = info['archive_info']['hashes']['sha256']
    rows.append(f"{name} @ {info['url']} --hash=sha256:{digest}")
Path(sys.argv[2]).write_text('\n'.join(rows)+'\n')
print('Locked Python wheels:', len(rows), '; GPU packages unchanged')
PY
"$BASE_PYTHON" -m pip --isolated --disable-pip-version-check --python "$VENV/bin/python" install \
    --no-deps --require-hashes --cache-dir "$WORK_ROOT/cache/pip-sharegpt" \
    --timeout 30 --retries 2 -r "$LOCKED"
"$VENV/bin/python" - <<'PY'
import torch, transformers, triton, numpy, safetensors
assert torch.__version__.split('+')[0] == '2.9.1' and torch.version.cuda == '12.8'
assert transformers.__version__ == '4.57.3' and triton.__version__ == '3.5.1'
print('SHAREGPT ENV READY:', torch.__version__, transformers.__version__, numpy.__version__, safetensors.__version__)
print('No model download or GPU benchmark started.')
PY
