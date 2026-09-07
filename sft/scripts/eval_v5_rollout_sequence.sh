#!/usr/bin/env bash
# v5 rollout eval, same order as v2/v3/v4:
#   AIME x3 -> math-rest 1-pass -> physics 1-pass -> phyx-af 1-pass
# Prompt must match training: system = identity + schema, user = instruction + problem.
# EVAL_PROMPT_PATCH=0.
set -euo pipefail

ROOT=/mnt/afs_reason/chendongyang/code
SFT=$ROOT/llm4math/sft
cd "$SFT"

MODEL_URL=${MODEL_URL:?set MODEL_URL e.g. http://10.x.x.x:8000/v1}
MODEL_NAME=${MODEL_NAME:-qwen35-9b-reform-sft-v5}
SEARCH_URL=${SEARCH_URL:-http://10.120.1.119:8000}
VERIFY_URL=${VERIFY_URL:-http://10.120.1.3:8001}
VERIFY_KEY=${VERIFY_KEY:-reform-lean-key}
WORKERS=${WORKERS:-16}
SPLIT=${SPLIT:-all}
EVAL_PROMPT_PATCH=${EVAL_PROMPT_PATCH:-0}
PHYSSEARCH_ON_MATH=${PHYSSEARCH_ON_MATH:-0}
VERIFY_MAX_CONCURRENCY=${VERIFY_MAX_CONCURRENCY:-1}
VERIFY_TIMEOUT=${VERIFY_TIMEOUT:-900}
VERIFY_CANONICAL_HEADER=${VERIFY_CANONICAL_HEADER:-1}

export MODEL_URL MODEL_NAME SEARCH_URL VERIFY_URL VERIFY_KEY
export WORKERS SPLIT EVAL_PROMPT_PATCH PHYSSEARCH_ON_MATH
export VERIFY_MAX_CONCURRENCY VERIFY_TIMEOUT VERIFY_CANONICAL_HEADER
export EVAL_SCRIPT=eval_rollout.sh

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "[$(date)] v5 sequence start MODEL_URL=$MODEL_URL MODEL_NAME=$MODEL_NAME"

echo "[$(date)] AIME x3"
REPEATS=3 TAG=rollout_v5_nopatch SUITES=aime bash scripts/eval_repeat.sh

echo "[$(date)] math-rest 1-pass"
REPEATS=1 TAG=rollout_v5_nopatch_mathrest SUITES=math-rest bash scripts/eval_repeat.sh

echo "[$(date)] physics 1-pass"
REPEATS=1 TAG=rollout_v5_nopatch_physics SUITES=physics bash scripts/eval_repeat.sh

echo "[$(date)] phyx-af 1-pass"
REPEATS=1 TAG=rollout_v5_nopatch_phyxaf SUITES=phyx-af bash scripts/eval_repeat.sh

echo "[$(date)] v5 sequence done"
