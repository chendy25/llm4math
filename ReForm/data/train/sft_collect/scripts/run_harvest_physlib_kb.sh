#!/usr/bin/env bash
# Launch a real KB harvest (reverify + informalize) detached.
set -euo pipefail
cd /mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
set -a
# shellcheck disable=SC1091
source poc_phyx/.llm.env
set +a
export VERIFY_URL="${VERIFY_URL:-http://10.120.1.3:8001}"
export VERIFY_KEY="${VERIFY_KEY:-reform-lean-key}"
export PYTHONUNBUFFERED=1
PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
mkdir -p sft_collect/logs sft_collect/physlib_kb
LOG=sft_collect/logs/harvest_physlib_kb.log
: > "$LOG"
# Extra args: default trusts collect-time syntax_pass (Kimina often saturated
# during math collect). Pass REVERIFY=1 to force --reverify.
EXTRA=(--informalize)
if [[ "${REVERIFY:-0}" == "1" ]]; then
  EXTRA=(--reverify --informalize --verify-timeout "${VERIFY_TIMEOUT:-120}")
else
  EXTRA=(--no-reverify --informalize)
fi
setsid "$PY" -u sft_collect/scripts/harvest_physlib_kb.py \
  --inputs \
    sft_collect/physics/sft_physics_mm_smoke10.jsonl \
    'sft_collect/physics/sft_physics_mm_pilot100_shard*.jsonl' \
    'sft_collect/physics/sft_physics_mm_full_shard*.jsonl' \
  "${EXTRA[@]}" \
  --auto-model "$AUTO_MODEL" \
  --auto-base-url "$TOKENHUB_BASE_URL" \
  --auto-api-key "$TOKENHUB_API_KEY" \
  >> "$LOG" 2>&1 < /dev/null &
echo $! > sft_collect/logs/harvest_physlib_kb.pid
echo "started pid=$(cat sft_collect/logs/harvest_physlib_kb.pid) log=$LOG args=${EXTRA[*]}"
