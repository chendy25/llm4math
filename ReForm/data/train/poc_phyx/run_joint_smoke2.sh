#!/usr/bin/env bash
# Smoke: 1 math + 1 physics (PhyX vision multimodal) with --joint-retrieval.
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
SEARCH_URL="${SEARCH_URL:-http://10.120.5.252:8000}"
PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
MODEL="${AUTO_MODEL:-gpt-4o}"
OUT=poc_phyx/sft_joint_smoke2.jsonl
TMP=${OUT}.tmp
LOG=poc_phyx/run_joint_smoke2.log
echo "MODEL=$MODEL joint-retrieval+vision SEARCH_URL=$SEARCH_URL"
rm -f "$TMP"
$PY run_collect_sft_data.py \
  --input poc_phyx/joint_smoke2.jsonl \
  --output "$TMP" \
  --max-items 2 \
  --max-rounds 3 \
  --max-tool-calls 3 \
  --include-image \
  --joint-retrieval \
  --physsearch-url "$SEARCH_URL" \
  --leansearch-url "$SEARCH_URL" \
  --auto-model "$MODEL" \
  --auto-base-url "$TOKENHUB_BASE_URL" \
  --auto-api-key "$TOKENHUB_API_KEY" \
  --critic-model "$MODEL" \
  --critic-base-url "$TOKENHUB_BASE_URL" \
  --critic-api-key "$TOKENHUB_API_KEY" \
  2>&1 | tee "$LOG"
mv -f "$TMP" "$OUT"
echo EXIT:$?
