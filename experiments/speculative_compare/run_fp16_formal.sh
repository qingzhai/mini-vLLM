#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/hy-tmp/minivllm-work}"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
PYTHON="$WORK_ROOT/venv-deepspec/bin/python"
PREVIOUS="$(cat "$WORK_ROOT/results/speculative-fp16-latest.txt")"
"$PYTHON" - "$PREVIOUS" <<'PY'
import json, sys
from pathlib import Path
folder = Path(sys.argv[1]) / 'measured'
report = json.loads((folder/'report.json').read_text())
config = json.loads((folder/'run-config.json').read_text())
assert report['qualified_greedy_comparison'], 'FP16 regression failed; preserve it instead of starting the full run'
assert config['target_backbone'] == 'fp16' and config['target_head'] == 'fp32'
assert config['full_fp16_reduction'] and not config['full_bf16_reduction']
assert config['max_new_tokens'] == 256 and report['cases'] == 3
print('Prior FP16 regression passed; continuing with the same precision settings.', flush=True)
PY
LOCK="$WORK_ROOT/results/speculative-formal-20261019-090231Z-cUSKhB/prepared/lock.json"
test -f "$LOCK"
RUN_DIR="$(mktemp -d "$WORK_ROOT/results/speculative-fp16-formal-$(date -u +%Y%m%d-%H%M%SZ)-XXXXXX")"
printf '%s\n' "$RUN_DIR" > "$WORK_ROOT/results/speculative-fp16-formal-latest.txt"
ln -sfn "$RUN_DIR/measured-records.zip" "$WORK_ROOT/speculative-fp16-formal-latest-records.zip"
echo "RESULTS: $RUN_DIR"
echo "Original twelve prompts; 256 output tokens, three repeats; all seven methods plus paired AR."
echo "FP16 Target / FP32 Head / BF16 Draft; same settings as the passed regression."
set +e
"$PYTHON" -u "$HERE/benchmark.py" --lock "$LOCK" --output "$RUN_DIR/measured" \
    --repeats 3 --max-new-tokens 256 --warmup-tokens 32 \
    --target-backbone fp16 --target-head fp32 --full-fp16-reduction \
    2>&1 | tee "$RUN_DIR/run.log"
STATUS=${PIPESTATUS[0]}
set -e
echo "EXIT STATUS: $STATUS"
echo "RECORDS: $RUN_DIR/measured-records.zip"
exit "$STATUS"
