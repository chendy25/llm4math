#!/usr/bin/env bash
# Collect physics multimodal ReForm SFT (vision + physsearch + optional web + verify).
# Success → OUT; failures → <outdir>/failed/<name>.failed.jsonl
set -euo pipefail
cd /mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTHONUNBUFFERED=1

if [[ -f poc_phyx/.llm.env ]]; then
  # shellcheck disable=SC1091
  source poc_phyx/.llm.env
elif [[ -f poc_phyx/.dashscope.env ]]; then
  # shellcheck disable=SC1091
  source poc_phyx/.dashscope.env
else
  # shellcheck disable=SC1091
  source poc_phyx/.tokenhub.env
fi
# shellcheck disable=SC1091
source poc_phyx/.serper.env

export VERIFY_URL="${VERIFY_URL:-http://10.120.1.3:8001}"
export VERIFY_KEY="${VERIFY_KEY:-reform-lean-key}"
SEARCH_URL="${SEARCH_URL:-http://10.120.5.252:8000}"
PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
MODEL="${AUTO_MODEL:-gpt-4o}"

IN="${1:-sft_collect/batches/pilot_physics100.jsonl}"
OUT="${2:-sft_collect/physics/sft_physics_mm_pilot100.jsonl}"
LOG="${3:-sft_collect/logs/collect_physics_pilot.log}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-3}"

echo "PHYSICS-MM collect MODEL=$MODEL IN=$IN OUT=$OUT SEARCH=$SEARCH_URL VERIFY=$VERIFY_URL max_tool_calls=$MAX_TOOL_CALLS"
mkdir -p "$(dirname "$OUT")" sft_collect/logs

exec "$PY" run_collect_sft_data.py \
  --input "$IN" \
  --output "$OUT" \
  --max-rounds 3 \
  --max-tool-calls "$MAX_TOOL_CALLS" \
  --include-image \
  --use-lean-verify \
  --joint-retrieval \
  --use-web-search \
  --physsearch-url "$SEARCH_URL" \
  --leansearch-url "$SEARCH_URL" \
  --auto-model "$MODEL" \
  --auto-base-url "$TOKENHUB_BASE_URL" \
  --auto-api-key "$TOKENHUB_API_KEY" \
  --critic-model "$MODEL" \
  --critic-base-url "$TOKENHUB_BASE_URL" \
  --critic-api-key "$TOKENHUB_API_KEY" \
  >> "$LOG" 2>&1
