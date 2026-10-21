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
REFERENCE="$HERE/formal-workload.json"
if [ ! -f "$REFERENCE" ]; then
    REFERENCE="$HERE/../../results/speculative_compare/formal-20261019/workload.json"
fi
test -x "$PYTHON"
test -f "$REFERENCE"
mkdir -p "$WORK_ROOT/results"
RUN_DIR="$(mktemp -d "$WORK_ROOT/results/speculative-fp16-regression-$(date -u +%Y%m%d-%H%M%SZ)-XXXXXX")"
printf '%s\n' "$RUN_DIR" > "$WORK_ROOT/results/speculative-fp16-latest.txt"
ln -sfn "$RUN_DIR/measured-records.zip" "$WORK_ROOT/speculative-fp16-latest-records.zip"
echo "RESULTS: $RUN_DIR"
echo "FP16 Target parameters + FP32 Head; BF16 Draft; RoPE FP32 buffers preserved."
echo "Only FP16 reduced-precision reduction is disabled; BF16 Draft reduction remains as before."
echo "Three existing failures, 256 output tokens, one repeat, all seven methods and paired AR; no formal rerun."
EXTRA=()
if [ -n "${SOURCE_LOCK:-}" ]; then
    EXTRA+=(--source-lock "$SOURCE_LOCK")
fi
"$PYTHON" -u "$HERE/prepare_fp16_regression.py" --root "$WORK_ROOT" \
    --canonical-workload "$REFERENCE" --output "$RUN_DIR/prepared" "${EXTRA[@]}" \
    2>&1 | tee "$RUN_DIR/selection.log"
set +e
"$PYTHON" -u "$HERE/benchmark.py" --lock "$RUN_DIR/prepared/lock.json" \
    --output "$RUN_DIR/measured" --repeats 1 --max-new-tokens 256 --warmup-tokens 32 \
    --target-backbone fp16 --target-head fp32 --full-fp16-reduction \
    2>&1 | tee "$RUN_DIR/run.log"
STATUS=${PIPESTATUS[0]}
set -e
echo "EXIT STATUS: $STATUS"
echo "RECORDS: $RUN_DIR/measured-records.zip"
exit "$STATUS"
