#!/usr/bin/env bash
# Resume physics shards that API (qwen3.5-plus) has not finished.
# Output dir is model-tagged so DashScope rows stay in sft_collect/physics/.
set -uo pipefail
cd /mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
SEARCH_URL="${SEARCH_URL:-http://10.120.5.252:8000}"

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

export PYTHONUNBUFFERED=1
export VERIFY_URL="${VERIFY_URL:-http://10.120.1.3:8001}"
export VERIFY_KEY="${VERIFY_KEY:-reform-lean-key}"
export PHYSSEARCH_KB_MODE=local
export PHYSSEARCH_INCLUDE_MATHLIB=1
MODEL="${AUTO_MODEL:-Qwen3.8-27B}"
TAG="${MODEL_TAG:-qwen38_27b}"
OUTDIR="sft_collect/physics_${TAG}"

LOGDIR=sft_collect/logs
mkdir -p "$LOGDIR" "$OUTDIR"

for s in 00 01 02 03; do
  IN="${PHYS_IN_DIR:-sft_collect/batches/shards_sft_physics_mm_full_api_left}/phys_${s}.jsonl"
  OUT="${OUTDIR}/sft_physics_mm_full_shard${s}.jsonl"
  LOG="$LOGDIR/collect_physics_${TAG}_shard${s}.log"
  setsid "$PY" run_collect_sft_data.py \
    --input "$IN" \
    --output "$OUT" \
    --max-rounds 3 \
    --max-tool-calls 3 \
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
    < /dev/null >> "$LOG" 2>&1 &
  echo $! > "$LOGDIR/collect_physics_${TAG}_shard${s}.pid"
  disown
  echo "LAUNCH_OK physics ${TAG} shard${s} pid=$(cat "$LOGDIR/collect_physics_${TAG}_shard${s}.pid") out=$OUT"
done
