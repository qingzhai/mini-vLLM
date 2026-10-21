#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/hy-tmp/minivllm-work}"
export PYTHONDONTWRITEBYTECODE=1
BASE_PYTHON="${BASE_PYTHON:-/usr/local/bin/python}"
LOCK="$WORK_ROOT/results/speculative-formal-20261019-090231Z-cUSKhB/prepared/lock.json"
test -f "$LOCK"
bash "$HERE/restore_dependencies.sh"
"$BASE_PYTHON" -u "$HERE/restore_model_weights.py" --root "$WORK_ROOT" --lock "$LOCK" --workers 2
echo "RECOVERY STAGE 2 COMPLETE. Environment and original weights restored; no benchmark started."
echo "Next entry: $WORK_ROOT/minivllm-speculative-fp16-v1/run_fp16_regression.sh"
