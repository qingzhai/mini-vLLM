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
test -x "$PYTHON"
mkdir -p "$WORK_ROOT/results"
PREVIOUS="${REGRESSION_DIR:-$(cat "$WORK_ROOT/results/speculative-legacy-target-latest.txt")}"
RUN_DIR="$(mktemp -d "$WORK_ROOT/results/speculative-legacy-target-formal-$(date -u +%Y%m%d-%H%M%SZ)-XXXXXX")"
printf '%s\n' "$RUN_DIR" > "$WORK_ROOT/results/speculative-legacy-target-formal-latest.txt"
ln -sfn "$RUN_DIR/measured-records.zip" "$WORK_ROOT/speculative-legacy-target-formal-latest-records.zip"
echo "RESULTS: $RUN_DIR"
"$PYTHON" -u "$HERE/prepare_legacy_formal.py" --previous "$PREVIOUS" \
    --output "$RUN_DIR/prepared" 2>&1 | tee "$RUN_DIR/selection.log"
echo "Same passed Target arithmetic and official Drafts; Top-2 collection disabled."
echo "Twelve original prompts, 256 output tokens, three repeats; seven modes and paired AR."
echo "No download, installation, extra regression or CUDA Graph."
set +e
"$PYTHON" -u "$HERE/benchmark.py" --lock "$RUN_DIR/prepared/lock.json" \
    --output "$RUN_DIR/measured" --repeats 3 --max-new-tokens 256 --warmup-tokens 32 \
    --target-backend legacy_minivllm --target-backbone fp16 --target-head fp32 \
    --full-fp16-reduction --full-bf16-reduction \
    2>&1 | tee "$RUN_DIR/run.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ -f "$RUN_DIR/measured/report.json" ]; then
    cp "$RUN_DIR/prepared/selection.json" "$RUN_DIR/measured/formal-selection.json"
    "$PYTHON" "$HERE/analyze_divergence.py" --records-dir "$RUN_DIR/measured" \
        --output-dir "$RUN_DIR/measured/divergence" || echo "Analysis incomplete; raw records preserved."
    "$PYTHON" -c 'import shutil,sys; shutil.make_archive(sys.argv[1]+"-records", "zip", sys.argv[1])' \
        "$RUN_DIR/measured"
fi
echo "EXIT STATUS: $STATUS"
echo "RECORDS: $RUN_DIR/measured-records.zip"
exit "$STATUS"
