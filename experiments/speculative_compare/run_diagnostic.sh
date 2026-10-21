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
EVIDENCE="$WORK_ROOT/results/gpu-smoke-20261019-070027Z/measured/dspark-records.jsonl"
test -x "$PYTHON"
test -f "$LOCK"
test -f "$EVIDENCE"
mkdir -p "$WORK_ROOT/results"
RUN_DIR="$(mktemp -d "$WORK_ROOT/results/dspark-greedy-diagnostic-$(date -u +%Y%m%d-%H%M%SZ)-XXXXXX")"
printf '%s\n' "$RUN_DIR" > "$WORK_ROOT/results/dspark-greedy-latest.txt"
echo "RESULTS: $RUN_DIR"
"$PYTHON" -u "$HERE/diagnose_greedy.py" --lock "$LOCK" --evidence "$EVIDENCE" \
    --case humaneval:19 --tokens 32 --focus-round 5 --output "$RUN_DIR/diagnostic" \
    "$@" 2>&1 | tee "$RUN_DIR/run.log"
