#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/hy-tmp/minivllm-work}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="$WORK_ROOT/cache/huggingface"
export UV_CACHE_DIR="$WORK_ROOT/cache/uv"
export UV_INDEX_STRATEGY=unsafe-best-match
export PIP_CACHE_DIR="$WORK_ROOT/cache/pip-deepspec"
export TMPDIR="$WORK_ROOT/cache/tmp"
export TORCHINDUCTOR_CACHE_DIR="$WORK_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$WORK_ROOT/cache/triton"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
mkdir -p "$WORK_ROOT/results" "$TMPDIR"
cd "$HERE"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
"$PYTHON_BIN" - "$WORK_ROOT" <<'PY'
from pathlib import Path
import shutil
import sys
root = Path(sys.argv[1])
free = shutil.disk_usage(root).free / 2**30
print(f'Data directory: {root}; available: {free:.2f} GiB', flush=True)
if free < 35:
    raise SystemExit('At least 35 GiB free is required for this startup. Check the data disk before downloading.')
PY

OFFLINE="$WORK_ROOT/minivllm-deepspec-offline-v1/install_offline.py"
if [ ! -f "$OFFLINE" ]; then
    echo "Missing offline source installer: $OFFLINE"
    exit 1
fi
"$PYTHON_BIN" "$OFFLINE" --work-root "$WORK_ROOT"
bash setup.sh

echo "===== SMALL GPU CHECK ====="
PER_DATASET=1 REPEATS=1 OUTPUT_TOKENS=32 bash run_gpu.sh

echo "===== FORMAL THREE-REPEAT BENCHMARK ====="
PER_DATASET=4 REPEATS=3 OUTPUT_TOKENS=256 bash run_gpu.sh

echo "ALL_STAGES_FINISHED"
