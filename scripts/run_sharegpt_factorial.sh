#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export WORK_ROOT="${WORK_ROOT:-/hy-tmp/minivllm-work}"
PYTHON="${SHAREGPT_PYTHON:-$WORK_ROOT/venv-sharegpt/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
test -x "$PYTHON"
"$PYTHON" - <<'PY'
import sys, torch, transformers, triton
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split('+')[0] == '2.9.1' and torch.version.cuda == '12.8'
assert transformers.__version__ == '4.57.3' and triton.__version__ == '3.5.1'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print('GPU:', torch.cuda.get_device_name(0), '; four modes, uniform Decode-first, no Draft models', flush=True)
PY
exec "$PYTHON" -u "$REPO/scripts/run.py" sharegpt-factorial \
    --work-root "$WORK_ROOT" -- --repeats "${REPEATS:-3}"
