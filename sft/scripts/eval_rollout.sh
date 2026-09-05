#!/usr/bin/env bash
# Evaluate the ReForm SFT model with single-shot ("RL rollout") generation:
# one continuous assistant turn for the whole <think>/<round>.../<answer>
# trajectory, real tools spliced into the same sequence, one token budget.
#
#   MODEL_URL=http://<POD_IP>:8000/v1 SUITES=aime bash scripts/eval_rollout.sh
#   SUITES=math-rest MODEL_URL=... bash scripts/eval_rollout.sh
#   SUITES=physics MODEL_URL=... bash scripts/eval_rollout.sh
#   SUITES=phyx-af MODEL_URL=... bash scripts/eval_rollout.sh
#
# SUITES accepts aime | math | math-rest | physics | phyx-af | comma-separated
# names (aime2025, minif2f, proofnet, putnam, phyx, seephys, physunibench,
# phyx-af). math-rest = three official math sets after AIME. physics = PhyX
# test-mini + SeePhys + PhysUniBench (vision + physsearch). phyx-af is the
# MMFormalizer 256-item mix (not the PhyX-1000 set); do not mix those numbers.
set -euo pipefail

ROOT=/mnt/afs_reason/chendongyang/code
TRAIN=$ROOT/llm4math/ReForm/data/train
EVAL_DIR=$ROOT/llm4math/sft/eval
SETS=$EVAL_DIR/sets
TAG=${TAG:-rollout}
OUT_DIR=${OUT_DIR:-$EVAL_DIR/runs/${TAG}_$(date +%Y%m%d_%H%M%S)}
WORKERS=${WORKERS:-16}
mkdir -p "$OUT_DIR"

MODEL_URL=${MODEL_URL:?set MODEL_URL e.g. http://10.x.x.x:8000/v1}
MODEL_NAME=${MODEL_NAME:-qwen35-9b-reform-sft-v3}
SEARCH_URL=${SEARCH_URL:-http://10.120.1.119:8000}
VERIFY_URL=${VERIFY_URL:-http://10.120.1.3:8001}
VERIFY_KEY=${VERIFY_KEY:-reform-lean-key}
export VERIFY_URL VERIFY_KEY
# Kimina keeps one REPL per import set; exceeding the warm count makes it spawn
# a cold one (~5min for `import Mathlib`), which is worse than queueing.
export VERIFY_MAX_CONCURRENCY="${VERIFY_MAX_CONCURRENCY:-1}"
export VERIFY_RETRIES="${VERIFY_RETRIES:-8}"
export VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-900}"
export VERIFY_CANONICAL_HEADER="${VERIFY_CANONICAL_HEADER:-1}"
export CRITIC_NO_THINKING="${CRITIC_NO_THINKING:-1}"
export PHYSSEARCH_ON_MATH="${PHYSSEARCH_ON_MATH:-0}"
# Collection-time constraints appended to the user message (not a system turn).
# Default off: the trained user prompt only. Set EVAL_PROMPT_PATCH=1 for a
# separate column; do not mix the two.
EVAL_PROMPT_PATCH=${EVAL_PROMPT_PATCH:-0}

MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-20480}
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-9}
MAX_TOOL_RESPONSE_TOKENS=${MAX_TOOL_RESPONSE_TOKENS:-1536}
MAX_TOKENS_CONSISTENCY=${MAX_TOKENS_CONSISTENCY:-4096}
TEMPERATURE=${TEMPERATURE:-0.6}

if [[ -f $TRAIN/poc_phyx/.tokenhub.env ]]; then
  # shellcheck disable=SC1091
  source "$TRAIN/poc_phyx/.tokenhub.env"
fi
if [[ -f $TRAIN/poc_phyx/.serper.env ]]; then
  # shellcheck disable=SC1091
  source "$TRAIN/poc_phyx/.serper.env"
fi
CRITIC_MODEL=${CRITIC_MODEL:-qwen3.7-plus}
CRITIC_BASE_URL=${CRITIC_BASE_URL:-https://tokenhub.sensetime.com/v1}
CRITIC_API_KEY=${CRITIC_API_KEY:-${TOKENHUB_API_KEY:-}}
if [[ -z "$CRITIC_API_KEY" ]]; then
  echo "set CRITIC_API_KEY or TOKENHUB_API_KEY for the TokenHub judge" >&2
  exit 1
fi

PY=${PY:-/mnt/afs_reason/chendongyang/conda/leansearch/bin/python}
SUITES=${SUITES:-math}
SPLIT=${SPLIT:-all}
MAX_ITEMS=${MAX_ITEMS:-}
if [[ "$SPLIT" != "all" && "$SPLIT" != "heldout" ]]; then
  echo "SPLIT must be all or heldout, got: $SPLIT" >&2
  exit 1
fi

cd "$TRAIN"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTHONUNBUFFERED=1

run_one() {
  local name="$1" input="$2"
  shift 2
  local out="$OUT_DIR/${name}.jsonl"
  local log="$OUT_DIR/${name}.log"
  echo "==== $name ===="
  echo "IN=$input OUT=$out"
  local args=(
    "$PY" run_eval_rollout.py
    --input "$input"
    --output "$out"
    --workers "$WORKERS"
    --temperature "$TEMPERATURE"
    --max-response-tokens "$MAX_RESPONSE_TOKENS"
    --max-tool-calls "$MAX_TOOL_CALLS"
    --max-tool-response-tokens "$MAX_TOOL_RESPONSE_TOKENS"
    --max-tokens-consistency "$MAX_TOKENS_CONSISTENCY"
    --use-lean-verify
    --use-leansearch
    --leansearch-url "$SEARCH_URL"
    --use-physsearch
    --physsearch-url "$SEARCH_URL"
    --use-web-search
    --auto-model "$MODEL_NAME"
    --auto-base-url "$MODEL_URL"
    --auto-api-key EMPTY
    --critic-model "$CRITIC_MODEL"
    --critic-base-url "$CRITIC_BASE_URL"
    --critic-api-key "$CRITIC_API_KEY"
    --save-response
    "$@"
  )
  if [[ -n "$MAX_ITEMS" ]]; then
    args+=(--max-items "$MAX_ITEMS")
  fi
  case "${EVAL_PROMPT_PATCH,,}" in
    1|true|yes) args+=(--eval-prompt-patch) ;;
    *) args+=(--no-eval-prompt-patch) ;;
  esac
  "${args[@]}" >>"$log" 2>&1 || true
  "$PY" - <<PY
import json
from collections import Counter
from pathlib import Path

rows = []
p = Path("$out")
if p.exists():
    rows = [json.loads(l) for l in p.open() if l.strip()]
n = len(rows)
infra = sum(1 for r in rows if r.get("verify_error"))
ok = sum(1 for r in rows if r.get("success"))
syn = sum(1 for r in rows if r.get("syntax_pass"))
den = n - infra
patch = rows[0].get("eval_prompt_patch") if rows else None
print(f"[$name] n={n} success={ok} syntax_pass={syn} verify_error={infra} "
      f"rate={ok/den if den else 0:.3f} (over {den} verified) "
      f"eval_prompt_patch={patch}")
if rows:
    toks = sorted(r.get("response_tokens") or 0 for r in rows)
    q = lambda f: toks[min(len(toks) - 1, int(f * (len(toks) - 1)))]
    print(f"[$name] response_tokens p50={q(.5)} p90={q(.9)} max={toks[-1]} "
          f"budget={rows[0].get('budget')}")
    print(f"[$name] stop_reason={dict(Counter(r.get('stop_reason') for r in rows))}")
    print(f"[$name] answer_tag={sum(1 for r in rows if r.get('answer_tag'))}/{n} "
          f"truncated={sum(1 for r in rows if r.get('truncated'))}/{n} "
          f"used_vision={sum(1 for r in rows if r.get('used_vision'))}/{n} "
          f"rounds={dict(Counter(r.get('num_rounds') for r in rows))} "
          f"tool_calls={dict(Counter(r.get('num_tool_calls') for r in rows))} "
          f"tools={dict(Counter(t for r in rows for t in (r.get('tools_used') or [])))}")
    both = [r for r in rows if r.get("self_verdict") is not None]
    if both:
        agree = sum(1 for r in both if bool(r["self_verdict"]) == bool(r.get("success")))
        selfok = sum(1 for r in both if r["self_verdict"])
        print(f"[$name] self_verdict Correct={selfok}/{len(both)} "
              f"agrees_with_critic={agree}/{len(both)}")
PY
}

echo "MODEL_URL=$MODEL_URL SEARCH=$SEARCH_URL VERIFY=$VERIFY_URL SUITES=$SUITES SPLIT=$SPLIT WORKERS=$WORKERS"
echo "CRITIC=$CRITIC_MODEL @ $CRITIC_BASE_URL no_thinking=$CRITIC_NO_THINKING"
echo "BUDGET response=$MAX_RESPONSE_TOKENS tool_calls<=$MAX_TOOL_CALLS tool_response<=$MAX_TOOL_RESPONSE_TOKENS temp=$TEMPERATURE"
echo "EVAL_PROMPT_PATCH=$EVAL_PROMPT_PATCH (user-append; off = trained prompt only)"
echo "OUT_DIR=$OUT_DIR"
{
  echo "EVAL_PROMPT_PATCH=$EVAL_PROMPT_PATCH"
  echo "MAX_RESPONSE_TOKENS=$MAX_RESPONSE_TOKENS"
  echo "PHYSSEARCH_ON_MATH=$PHYSSEARCH_ON_MATH"
  echo "MODEL_URL=$MODEL_URL"
  echo "MODEL_NAME=$MODEL_NAME"
  echo "SUITES=$SUITES"
  echo "SPLIT=$SPLIT"
  echo "TAG=$TAG"
} >"$OUT_DIR/eval_config.txt"

suite_file() {
  case "$1" in
    aime2025) echo "$SETS/math_aime2025_${SPLIT}.jsonl" ;;
    minif2f)  echo "$SETS/math_minif2f_testset_${SPLIT}.jsonl" ;;
    proofnet) echo "$SETS/math_proofnet_testset_${SPLIT}.jsonl" ;;
    putnam)   echo "$SETS/math_putnam_v4_${SPLIT}.jsonl" ;;
    phyx|phyx_test_mini) echo "$SETS/physics_phyx_test_mini_${SPLIT}.jsonl" ;;
    seephys) echo "$SETS/physics_seephys_${SPLIT}.jsonl" ;;
    physunibench) echo "$SETS/physics_physunibench_${SPLIT}.jsonl" ;;
    phyx-af|phyxaf|phyx_af) echo "$SETS/phyx_af_${SPLIT}.jsonl" ;;
    *) echo "unknown suite: $1" >&2; return 1 ;;
  esac
}

suite_extra() {
  case "$1" in
    phyx|phyx_test_mini|seephys|physunibench|phyx-af|phyxaf|phyx_af) echo "--include-image" ;;
  esac
}

declare -a SUITE_NAMES=()
IFS=',' read -ra _suite_parts <<< "$SUITES"
for _raw in "${_suite_parts[@]}"; do
  _p="${_raw// /}"
  _p="${_p,,}"
  case "$_p" in
    math)
      SUITE_NAMES+=(aime2025 minif2f proofnet putnam)
      ;;
    math-rest|math_rest|mathrest)
      SUITE_NAMES+=(minif2f proofnet putnam)
      ;;
    physics)
      SUITE_NAMES+=(phyx seephys physunibench)
      ;;
    aime|aime2025)
      SUITE_NAMES+=(aime2025)
      ;;
    phyx-af|phyxaf|phyx_af)
      SUITE_NAMES+=(phyx-af)
      ;;
    minif2f|proofnet|putnam|phyx|phyx_test_mini|seephys|physunibench)
      SUITE_NAMES+=("$_p")
      ;;
    "")
      ;;
    *)
      echo "unknown SUITES entry: $_raw (want aime|math|math-rest|physics|phyx-af|minif2f|proofnet|putnam|phyx|seephys|physunibench)" >&2
      exit 1
      ;;
  esac
done
declare -A _seen=()
declare -a _unique=()
for _n in "${SUITE_NAMES[@]}"; do
  if [[ -z "${_seen[$_n]:-}" ]]; then
    _seen[$_n]=1
    _unique+=("$_n")
  fi
done
SUITE_NAMES=("${_unique[@]}")
if [[ ${#SUITE_NAMES[@]} -eq 0 ]]; then
  echo "SUITES resolved to empty list: $SUITES" >&2
  exit 1
fi

for _name in "${SUITE_NAMES[@]}"; do
  _extra=$(suite_extra "$_name" || true)
  # shellcheck disable=SC2086
  run_one "$_name" "$(suite_file "$_name")" $_extra
done

echo "done. results in $OUT_DIR"
