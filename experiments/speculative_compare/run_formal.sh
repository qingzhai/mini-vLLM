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
mkdir -p "$WORK_ROOT/results"
RUN_DIR="$(mktemp -d "$WORK_ROOT/results/speculative-formal-$(date -u +%Y%m%d-%H%M%SZ)-XXXXXX")"
printf '%s\n' "$RUN_DIR" > "$WORK_ROOT/results/speculative-formal-latest.txt"
echo "RESULTS: $RUN_DIR"
echo "Formal protocol: GSM8K/HumanEval/MT-Bench, four prompts each, 256 output tokens, three repeats."
echo "Batch=1, offline local models, BF16 Target backbone and Draft, FP32 Target Head; all confidence modes retained."
echo "No setup, downloads or GPU smoke/regression stage."

# 原协议的十二条输入；仅使用本地模型、分词器和 DeepSpec 自带数据。
"$PYTHON" -u "$HERE/prepare.py" \
    --source "$WORK_ROOT/third_party/DeepSpec" \
    --target "$WORK_ROOT/models/Qwen3-8B" \
    --drafts-root "$WORK_ROOT/models" \
    --per-dataset 4 --max-input-tokens 1536 --seed 20261019 \
    --output "$RUN_DIR/prepared" 2>&1 | tee "$RUN_DIR/prepare.log"

# 完整测完各 worker 后统一汇总；输出分叉保留，不跳过样本或后续模式。
"$PYTHON" -u "$HERE/benchmark.py" \
    --lock "$RUN_DIR/prepared/lock.json" --output "$RUN_DIR/measured" \
    --repeats 3 --max-new-tokens 256 --warmup-tokens 32 --target-head fp32 \
    2>&1 | tee "$RUN_DIR/run.log"
