#!/usr/bin/env bash
# Evaluate the deployed ReForm SFT model on official test sets.
#
# Default is the full official suites (not leak-filtered heldout):
#   MODEL_URL=http://<POD_IP>:8000/v1 bash scripts/eval_all_testsets.sh
#   SUITES=math MODEL_URL=... bash scripts/eval_all_testsets.sh
#   SUITES=math,physics MODEL_URL=... bash scripts/eval_all_testsets.sh
#   SPLIT=heldout MODEL_URL=... bash scripts/eval_all_testsets.sh
set -euo pipefail

ROOT=/mnt/afs_reason/chendongyang/code
TRAIN=$ROOT/llm4math/ReForm/data/train
EVAL_DIR=$ROOT/llm4math/sft/eval
SETS=$EVAL_DIR/sets
OUT_DIR=${OUT_DIR:-$EVAL_DIR/runs/$(date +%Y%m%d_%H%M%S)}
LOG_DIR=$EVAL_DIR/logs
WORKERS=${WORKERS:-32}
mkdir -p "$OUT_DIR" "$LOG_DIR"

MODEL_URL=${MODEL_URL:?set MODEL_URL e.g. http://10.x.x.x:8000/v1}
MODEL_NAME=${MODEL_NAME:-qwen35-9b-reform-sft}
SEARCH_URL=${SEARCH_URL:-http://10.120.5.252:8000}
VERIFY_URL=${VERIFY_URL:-http://10.120.1.3:8001}
VERIFY_KEY=${VERIFY_KEY:-reform-lean-key}
export VERIFY_URL VERIFY_KEY
# Kimina holds one REPL per import set and a cold `import Mathlib` costs ~5min.
# Exceeding the number of *warm* REPLs makes Kimina spawn a cold one, which is
# far slower than just queueing, so keep this at the warm count. Warm first:
#   N=<VERIFY_MAX_CONCURRENCY> bash llm4math/kimina-lean-server/warm_repls.sh
export VERIFY_MAX_CONCURRENCY="${VERIFY_MAX_CONCURRENCY:-1}"
export VERIFY_RETRIES="${VERIFY_RETRIES:-8}"
export VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-900}"
export VERIFY_CANONICAL_HEADER="${VERIFY_CANONICAL_HEADER:-1}"
if [[ -f $TRAIN/poc_phyx/.tokenhub.env ]]; then
  # shellcheck disable=SC1091
  source "$TRAIN/poc_phyx/.tokenhub.env"
fi
MAX_TOKENS_AUTO=${MAX_TOKENS_AUTO:-24576}
MAX_TOKENS_CONSISTENCY=${MAX_TOKENS_CONSISTENCY:-4096}
# qwen3.7-plus spends 8k-17k reasoning chars per verdict without changing it.
export CRITIC_NO_THINKING="${CRITIC_NO_THINKING:-1}"
# Physlib retrieval on pure math items drags the model into `import Physlib`.
export PHYSSEARCH_ON_MATH="${PHYSSEARCH_ON_MATH:-0}"
CRITIC_MODEL=${CRITIC_MODEL:-qwen3.7-plus}
CRITIC_BASE_URL=${CRITIC_BASE_URL:-https://tokenhub.sensetime.com/v1}
CRITIC_API_KEY=${CRITIC_API_KEY:-${TOKENHUB_API_KEY:-}}
if [[ -z "$CRITIC_API_KEY" ]]; then
  echo "set CRITIC_API_KEY or TOKENHUB_API_KEY for the TokenHub judge" >&2
  exit 1
fi
PY=${PY:-/mnt/afs_reason/chendongyang/conda/leansearch/bin/python}
SUITES=${SUITES:-math,physics}
SPLIT=${SPLIT:-all}
MAX_ITEMS=${MAX_ITEMS:-}
if [[ "$SPLIT" != "all" && "$SPLIT" != "heldout" ]]; then
  echo "SPLIT must be all or heldout, got: $SPLIT" >&2
  exit 1
fi

if [[ -f $TRAIN/poc_phyx/.serper.env ]]; then
  # shellcheck disable=SC1091
  source "$TRAIN/poc_phyx/.serper.env"
fi

cd "$TRAIN"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTHONUNBUFFERED=1

run_one() {
  local name="$1" input="$2" extra="${3:-}"
  local out="$OUT_DIR/${name}.jsonl"
  local log="$OUT_DIR/${name}.log"
  echo "==== $name ===="
  echo "IN=$input OUT=$out"
  local args=(
    "$PY" run_collect_sft_data.py
    --input "$input"
    --output "$out"
    --max-rounds 3
    --max-tool-calls 3
    --use-lean-verify
    --joint-retrieval
    --leansearch-url "$SEARCH_URL"
    --physsearch-url "$SEARCH_URL"
    --auto-model "$MODEL_NAME"
    --auto-base-url "$MODEL_URL"
    --auto-api-key EMPTY
    --critic-model "$CRITIC_MODEL"
    --critic-base-url "$CRITIC_BASE_URL"
    --critic-api-key "$CRITIC_API_KEY"
    --workers "$WORKERS"
    --max-tokens-auto "$MAX_TOKENS_AUTO"
    --max-tokens-consistency "$MAX_TOKENS_CONSISTENCY"
  )
  if [[ -n "${SERPER_API_KEY:-}" ]]; then
    args+=(--use-web-search)
  fi
  if [[ -n "$MAX_ITEMS" ]]; then
    args+=(--max-items "$MAX_ITEMS")
  fi
  # shellcheck disable=SC2086
  "${args[@]}" $extra >>"$log" 2>&1 || true
  "$PY" - <<PY
import json
from pathlib import Path
from collections import Counter
p = Path("$out")
n = ok = syn = infra = 0
kinds = Counter()
if p.exists():
    for line in p.open():
        r = json.loads(line)
        n += 1
        ok += int(bool(r.get("success")))
        syn += int(bool(r.get("syntax_pass")))
        if r.get("verify_error"):
            infra += 1
            kinds[r["verify_error"].split(":")[0][:40]] += 1
# Items whose Lean verify never completed are excluded from the denominator:
# they measure the Lean service, not the model.
den = n - infra
print(f"[$name] n={n} success={ok} syntax_pass={syn} verify_error={infra} "
      f"rate={ok/den if den else 0:.3f} (over {den} verified)")
if kinds:
    print(f"[$name] verify_error kinds: {dict(kinds)}")
PY
}

echo "MODEL_URL=$MODEL_URL SEARCH=$SEARCH_URL VERIFY=$VERIFY_URL SUITES=$SUITES SPLIT=$SPLIT WORKERS=$WORKERS"
echo "CRITIC=$CRITIC_MODEL @ $CRITIC_BASE_URL no_thinking=$CRITIC_NO_THINKING"
echo "MAX_TOKENS auto=$MAX_TOKENS_AUTO critic=$MAX_TOKENS_CONSISTENCY physsearch_on_math=$PHYSSEARCH_ON_MATH"
echo "OUT_DIR=$OUT_DIR"

if [[ "$SUITES" == "aime" || "$SUITES" == *aime2025* ]]; then
  run_one aime2025 "$SETS/math_aime2025_${SPLIT}.jsonl"
elif [[ "$SUITES" == *math* ]]; then
  run_one aime2025 "$SETS/math_aime2025_${SPLIT}.jsonl"
  run_one minif2f "$SETS/math_minif2f_testset_${SPLIT}.jsonl"
  run_one proofnet "$SETS/math_proofnet_testset_${SPLIT}.jsonl"
  run_one putnam "$SETS/math_putnam_v4_${SPLIT}.jsonl"
fi

if [[ "$SUITES" == *physics* ]]; then
  run_one phyx_test_mini "$SETS/physics_phyx_test_mini_${SPLIT}.jsonl" "--include-image"
  run_one seephys "$SETS/physics_seephys_${SPLIT}.jsonl" "--include-image"
  run_one physunibench "$SETS/physics_physunibench_${SPLIT}.jsonl" "--include-image"
fi

echo "done. results in $OUT_DIR"
