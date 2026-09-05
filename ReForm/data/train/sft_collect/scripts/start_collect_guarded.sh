#!/usr/bin/env bash
# Preflight search/verify/local-LLM, then launch model-tagged math+physics
# collect with a watchdog that stops on search/verify/LLM failure.
set -euo pipefail
cd /mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

set -a
# shellcheck disable=SC1091
if [[ -f poc_phyx/.local_llm.env ]]; then
  source poc_phyx/.local_llm.env
else
  source poc_phyx/.llm.env
fi
# shellcheck disable=SC1091
source poc_phyx/.serper.env
set +a

export SEARCH_URL="${SEARCH_URL:-http://10.120.5.252:8000}"
export VERIFY_URL="${VERIFY_URL:-http://10.120.1.3:8001}"
export VERIFY_KEY="${VERIFY_KEY:-reform-lean-key}"
export LLM_PROBE_TIMEOUT="${LLM_PROBE_TIMEOUT:-90}"
export PYTHONUNBUFFERED=1
PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
LOGDIR=sft_collect/logs
mkdir -p "$LOGDIR"

echo "PREFLIGHT search=$SEARCH_URL verify=$VERIFY_URL llm=$TOKENHUB_BASE_URL model=$AUTO_MODEL"
"$PY" - <<'PY'
import os, sys
sys.path.insert(0, ".")
from pipeline_guard import check_llm, check_search, check_verify
ok_s, ms = check_search(os.environ["SEARCH_URL"])
ok_v, mv = check_verify(os.environ["VERIFY_URL"])
ok_l, ml = check_llm()
print(f"search: {ok_s} {ms}")
print(f"verify: {ok_v} {mv}")
print(f"llm: {ok_l} {ml}")
if not (ok_s and ok_v and ok_l):
    sys.exit(1)
PY

bash sft_collect/scripts/relaunch_math_shards.sh
bash sft_collect/scripts/relaunch_physics_shards.sh

WATCH_LOG="$LOGDIR/watchdog_collect.log"
setsid "$PY" -u sft_collect/scripts/watchdog_collect.py \
  < /dev/null >> "$WATCH_LOG" 2>&1 &
echo $! > "$LOGDIR/watchdog_collect.pid"
disown
echo "LAUNCH_OK watchdog pid=$(cat "$LOGDIR/watchdog_collect.pid") log=$WATCH_LOG"
echo "collect+watchdog started"
