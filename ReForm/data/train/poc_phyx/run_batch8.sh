#!/usr/bin/env bash
set -euo pipefail
cd /mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTHONUNBUFFERED=1
# Prefer DashScope / .llm.env; fall back to TokenHub.
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
PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
MODEL="${AUTO_MODEL:-gpt-4o}"
OUT=poc_phyx/sft_phyx_batch8.jsonl
TMP=${OUT}.tmp
LOG=poc_phyx/run_batch8.log
echo "MODEL=$MODEL"
rm -f "$TMP"
$PY run_collect_sft_data.py \
  --input poc_phyx/phyx_batch8.jsonl \
  --output "$TMP" \
  --max-items 8 \
  --max-rounds 2 \
  --max-tool-calls 3 \
  --include-image \
  --joint-retrieval \
  --physsearch-url http://127.0.0.1:8000 \
  --leansearch-url http://127.0.0.1:8000 \
  --auto-model "$MODEL" \
  --auto-base-url "$TOKENHUB_BASE_URL" \
  --auto-api-key "$TOKENHUB_API_KEY" \
  --critic-model "$MODEL" \
  --critic-base-url "$TOKENHUB_BASE_URL" \
  --critic-api-key "$TOKENHUB_API_KEY" \
  2>&1 | tee "$LOG"
mv -f "$TMP" "$OUT"
echo EXIT:$?
