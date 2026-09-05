#!/usr/bin/env bash
# 12-item joint retrieval + Lean verify (4 math AIME + 8 PhyX vision multimodal).
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
export VERIFY_URL="${VERIFY_URL:-http://10.120.1.3:8001}"
export VERIFY_KEY="${VERIFY_KEY:-reform-lean-key}"
SEARCH_URL="${SEARCH_URL:-http://10.120.5.252:8000}"
PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
MODEL="${AUTO_MODEL:-gpt-4o}"
OUT=poc_phyx/sft_joint_batch12_verify.jsonl
TMP=${OUT}.tmp
LOG=poc_phyx/run_joint_batch12_verify.log
echo "MODEL=$MODEL joint+vision+verify n=12 SEARCH_URL=$SEARCH_URL VERIFY_URL=$VERIFY_URL"
rm -f "$TMP"
$PY run_collect_sft_data.py \
  --input poc_phyx/joint_batch12.jsonl \
  --output "$TMP" \
  --max-items 12 \
  --max-rounds 3 \
  --max-tool-calls 3 \
  --include-image \
  --use-lean-verify \
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
# quick summary
$PY - <<'PY'
import json
from collections import Counter
path="poc_phyx/sft_joint_batch12_verify.jsonl"
rows=[json.loads(l) for l in open(path)]
print("n=", len(rows))
print("syntax_pass", sum(1 for r in rows if r.get("syntax_pass")), "/", len(rows))
print("success", sum(1 for r in rows if r.get("success")), "/", len(rows))
print("by_domain", Counter((r.get("retrieval_domain"), r.get("syntax_pass"), r.get("success")) for r in rows))
for r in rows:
    print(
        f"  {r.get('retrieval_domain')}/{r.get('dataset_name')}/{r.get('category')}: "
        f"syntax={r.get('syntax_pass')} success={r.get('success')} "
        f"rounds={r.get('num_rounds')} tools={r.get('tools_used')} "
        f"rewrites={r.get('syntax_rewrites')}"
    )
PY
