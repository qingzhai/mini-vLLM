#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/minivllm-work}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
PYTHON="$WORK_ROOT/venv-deepspec/bin/python"
SOURCE="$WORK_ROOT/third_party/DeepSpec"
mkdir -p "$WORK_ROOT/results"
RUN_ROOT="$(mktemp -d "$WORK_ROOT/results/speculative-compare-$(date -u +%Y%m%d-%H%M%SZ)-XXXXXX")"
echo "RESULTS: $RUN_ROOT"
target_args=()
if [ -f "$WORK_ROOT/models/Qwen3-8B/config.json" ]; then
    target_args=(--target "$WORK_ROOT/models/Qwen3-8B")
fi
"$PYTHON" "$HERE/prepare.py" --source "$SOURCE" --output "$RUN_ROOT/prepared" \
    --per-dataset "${PER_DATASET:-4}" "${target_args[@]}"
"$PYTHON" -u "$HERE/benchmark.py" --lock "$RUN_ROOT/prepared/lock.json" \
    --output "$RUN_ROOT/measured" --repeats "${REPEATS:-3}" --max-new-tokens "${OUTPUT_TOKENS:-256}"
