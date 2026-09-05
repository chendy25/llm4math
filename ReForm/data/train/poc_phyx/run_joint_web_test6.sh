#!/usr/bin/env bash
# Joint leansearch+physsearch + web_search + vision multimodal + Lean verify
# (2 math + 4 physics). PhyX rows must carry `image` paths; SFT dump keeps
# multipart user messages with image_url.
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
# shellcheck disable=SC1091
source poc_phyx/.serper.env
export VERIFY_URL="${VERIFY_URL:-http://10.120.1.3:8001}"
export VERIFY_KEY="${VERIFY_KEY:-reform-lean-key}"
# Local :8000 is often down on login; default to master dual-corpus serve.
SEARCH_URL="${SEARCH_URL:-http://10.120.5.252:8000}"
PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
MODEL="${AUTO_MODEL:-gpt-4o}"
OUT=poc_phyx/sft_joint_web_test6.jsonl
TMP=${OUT}.tmp
LOG=poc_phyx/run_joint_web_test6.log
INPUT=poc_phyx/joint_web_test6.jsonl

# Build 2 math + 4 physics from joint_batch12 (physics-first categories that
# previously struggled: Waves, Modern, Mechanics, EM). Preserve image paths.
$PY - <<'PY'
import json
from pathlib import Path
src = Path("poc_phyx/joint_batch12.jsonl")
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
math = [r for r in rows if (r.get("dataset_name") or "").lower() != "phyx"][:2]
phys_all = [r for r in rows if (r.get("dataset_name") or "").lower() == "phyx"]
prefer = {"Waves/Acoustics", "Modern Physics", "Mechanics", "Electromagnetism"}
phys = [r for r in phys_all if r.get("category") in prefer]
for r in phys_all:
    if len(phys) >= 4:
        break
    if r not in phys:
        phys.append(r)
phys = phys[:4]
out = Path("poc_phyx/joint_web_test6.jsonl")
with out.open("w") as f:
    for r in math + phys:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(
    "wrote", out, "n=", len(math) + len(phys),
    "phys_with_image", sum(1 for r in phys if r.get("image")),
    "phys", [(r.get("category"), bool(r.get("image"))) for r in phys],
)
PY

echo "MODEL=$MODEL joint+vision+web+verify n=6 SEARCH_URL=$SEARCH_URL VERIFY_URL=$VERIFY_URL"
rm -f "$TMP"
$PY run_collect_sft_data.py \
  --input "$INPUT" \
  --output "$TMP" \
  --max-items 6 \
  --max-rounds 4 \
  --max-tool-calls 6 \
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
  2>&1 | tee "$LOG"
mv -f "$TMP" "$OUT"
echo EXIT:$?
$PY - <<'PY'
import json
from collections import Counter
path = "poc_phyx/sft_joint_web_test6.jsonl"
rows = [json.loads(l) for l in open(path)]
print("n=", len(rows))
print("syntax_pass", sum(1 for r in rows if r.get("syntax_pass")), "/", len(rows))
print("success", sum(1 for r in rows if r.get("success")), "/", len(rows))
print("tools", Counter(t for r in rows for t in (r.get("tools_used") or [])))
print("by_domain", Counter((r.get("retrieval_domain"), r.get("syntax_pass"), r.get("success")) for r in rows))
for r in rows:
    print(
        f"  {r.get('retrieval_domain')}/{r.get('dataset_name')}/{r.get('category')}: "
        f"syntax={r.get('syntax_pass')} success={r.get('success')} "
        f"rounds={r.get('num_rounds')} tools={r.get('tools_used')} "
        f"rewrites={r.get('syntax_rewrites')}"
    )
PY
