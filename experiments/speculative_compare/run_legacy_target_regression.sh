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
RUN_DIR="$(mktemp -d "$WORK_ROOT/results/speculative-legacy-target-regression-$(date -u +%Y%m%d-%H%M%SZ)-XXXXXX")"
printf '%s\n' "$RUN_DIR" > "$WORK_ROOT/results/speculative-legacy-target-latest.txt"
ln -sfn "$RUN_DIR/measured-records.zip" "$WORK_ROOT/speculative-legacy-target-latest-records.zip"
echo "RESULTS: $RUN_DIR"
echo "Archived MinivLLM FP16 Target + FP32 Head; current official BF16 Drafts."
echo "Fixed-shape arithmetic, per-query PagedAttention; Draft SDPA MATH; no CUDA Graph."
echo "Three original prompts, 256 output tokens, one repeat, all seven modes and paired AR."
echo "Target Top-2 collection is enabled: diagnostic timing, not a formal performance result."
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
    --target-backend legacy_minivllm --target-backbone fp16 --target-head fp32 \
    --full-fp16-reduction --full-bf16-reduction --target-logit-diagnostics \
    2>&1 | tee "$RUN_DIR/run.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ -f "$RUN_DIR/measured/report.json" ]; then
    "$PYTHON" "$HERE/analyze_divergence.py" --records-dir "$RUN_DIR/measured" \
        --output-dir "$RUN_DIR/measured/divergence" || echo "Analysis incomplete; raw records preserved."
    "$PYTHON" -c 'import shutil,sys; shutil.make_archive(sys.argv[1]+"-records", "zip", sys.argv[1])' \
        "$RUN_DIR/measured"
fi
echo "EXIT STATUS: $STATUS"
echo "RECORDS: $RUN_DIR/measured-records.zip"
exit "$STATUS"
