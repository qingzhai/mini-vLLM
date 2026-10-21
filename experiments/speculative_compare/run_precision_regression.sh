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
LOCK="$WORK_ROOT/results/prepared-local-20261019-065730Z/lock.json"
test -x "$PYTHON"
test -f "$LOCK"
mkdir -p "$WORK_ROOT/results"
RUN_DIR="$(mktemp -d "$WORK_ROOT/results/speculative-fp32head-regression-$(date -u +%Y%m%d-%H%M%SZ)-XXXXXX")"
printf '%s\n' "$RUN_DIR" > "$WORK_ROOT/results/speculative-fp32head-latest.txt"
echo "RESULTS: $RUN_DIR"
echo "Regression only: original three prompts, 32 generated tokens, one repeat, all modes including confidence 0.8."
echo "All AR/speculative modes share BF16 Target backbone and FP32 Target Head; Draft remains BF16."
"$PYTHON" -u "$HERE/benchmark.py" --lock "$LOCK" --output "$RUN_DIR/measured" \
    --repeats 1 --max-new-tokens 32 --warmup-tokens 32 --target-head fp32 \
    2>&1 | tee "$RUN_DIR/run.log"
echo "SHORT REGRESSION COMPLETE; no formal benchmark was started."
