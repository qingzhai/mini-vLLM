#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/hy-tmp/minivllm-work}"
export PYTHONDONTWRITEBYTECODE=1
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
# 原准备记录使用这一绝对路径；换平台时先做路径迁移，不能默默用错旧 lock。
if [ "$WORK_ROOT" != /hy-tmp/minivllm-work ]; then
    echo "This handoff targets /hy-tmp/minivllm-work; rebase the archived lock before changing platforms." >&2
    exit 1
fi
"$BOOTSTRAP_PYTHON" -m zipfile -e "$HERE/minivllm-recovery-v1.zip" "$HERE/stage1"
"$BOOTSTRAP_PYTHON" "$HERE/stage1/minivllm-recovery-v1/recover_workspace.py" --work-root "$WORK_ROOT"
BASE_PYTHON="$("$BOOTSTRAP_PYTHON" - "$WORK_ROOT/recovery-environment.json" <<'PY'
import json, sys
from pathlib import Path
for row in json.loads(Path(sys.argv[1]).read_text()):
    packages = row.get('packages', {})
    if (row.get('version', [])[:2] == [3, 11] and (packages.get('torch') or '').split('+')[0] == '2.9.1'
            and packages.get('triton') == '3.5.1' and row.get('cuda_runtime') == '12.8'
            and row.get('cuda_available')):
        print(row['python'])
        break
else:
    raise SystemExit('No compatible preinstalled GPU stack; choose the planned image instead of reinstalling CUDA blindly')
PY
)"
export BASE_PYTHON
"$BOOTSTRAP_PYTHON" -m zipfile -e "$HERE/minivllm-recovery-stage2-v1.zip" "$WORK_ROOT"
cp "$HERE/run_fp16_formal.sh" "$WORK_ROOT/minivllm-speculative-fp16-v1/run_fp16_formal.sh"
bash "$WORK_ROOT/minivllm-recovery-stage2-v1/run_recovery_stage2.sh"
echo "NEW MACHINE RESTORE COMPLETE. No GPU benchmark started."
echo "Next: bash $WORK_ROOT/minivllm-speculative-fp16-v1/run_fp16_regression.sh"
