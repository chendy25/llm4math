#!/usr/bin/env bash
# =====================================================================
# LeanSearch-v2 dual-corpus serve (mathlib + physlib)
# 用 AFS 上的 conda/leansearch，不要用 CCI 镜像里的 vLLM python。
# =====================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-/mnt/afs_reason/chendongyang/conda/leansearch/bin/python}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
NUM_GPUS="${NUM_GPUS:-2}"
CORPORA="${CORPORA:-mathlib,physlib}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/mnt/afs_reason/chendongyang/code/data/models/Qwen3-Embedding-8B}"
RERANKER_MODEL_PATH="${RERANKER_MODEL_PATH:-/mnt/afs_reason/chendongyang/code/data/models/Qwen3-Reranker-8B}"
LOGDIR="${LOGDIR:-$REPO/data/cuvs/physlib}"

if [[ ! -x "$PY" ]]; then
  echo "leansearch python missing: $PY"
  exit 1
fi
if [[ ! -f "$EMBEDDING_MODEL_PATH/config.json" ]]; then
  echo "embedding weights missing: $EMBEDDING_MODEL_PATH"
  exit 1
fi
if [[ ! -f "$RERANKER_MODEL_PATH/config.json" ]]; then
  echo "reranker weights missing: $RERANKER_MODEL_PATH"
  exit 1
fi
if [[ ! -f "$REPO/data/cuvs/mathlib-v4.28.0-rc1/cuvs_index.bin" ]]; then
  echo "mathlib cuVS index missing"
  exit 1
fi
if [[ ! -f "$REPO/data/cuvs/physlib/cuvs_index.bin" ]]; then
  echo "physlib cuVS index missing"
  exit 1
fi

mkdir -p "$LOGDIR"
LOG="$LOGDIR/serve_physlib_cci_$(date +%Y%m%d_%H%M%S).log"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export EMBEDDING_MODEL_PATH RERANKER_MODEL_PATH
export CORPORA HOST PORT NUM_GPUS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cd "$REPO"
echo "[deploy] python=$PY"
echo "[deploy] hostname=$(hostname) ips=$(hostname -I 2>/dev/null || true)"
echo "[deploy] corpora=$CORPORA port=$PORT gpus=$NUM_GPUS"
echo "[deploy] log=$LOG"
echo "$LOG" > "$LOGDIR/serve_physlib.logpath"

exec "$PY" -m uvicorn leansearchv2.server:app --host "$HOST" --port "$PORT" \
  2>&1 | tee "$LOG"
