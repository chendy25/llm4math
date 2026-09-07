#!/usr/bin/env bash
# Wait for v5 SFT to actually finish, then merge, serve, and run full eval.
# ACP SUCCEEDED alone is not enough (v4's first job succeeded after a crash).
set -euo pipefail

ROOT=/mnt/afs_reason/chendongyang/code
SFT=$ROOT/llm4math/sft
SCRIPTS=$SFT/scripts
LORA_ROOT=$SFT/output/qwen35_9b_reform_lora_v5
MERGED=$SFT/output/qwen35_9b_reform_merged_v5/checkpoint-312
JOB_ID=${JOB_ID:-pt-q6v3zpxz}
WORKSPACE=${WORKSPACE:-rl}
SEARCH_URL=${SEARCH_URL:-http://10.120.1.119:8000}
VERIFY_URL=${VERIFY_URL:-http://10.120.1.3:8001}
STATE_FILE=${STATE_FILE:-$SFT/logs/wait_v5_sft_state.txt}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
mkdir -p "$SFT/logs"

job_line() {
  sco acp jobs list --workspace-name "$WORKSPACE" --user-name chendongyang --page-size 50 \
    | awk -v id="$JOB_ID" 'index($0, id) {print; exit}'
}

echo "[$(date)] wait SFT $JOB_ID + checkpoint-312"
ADAPTERS=""
i=0
while true; do
  i=$((i + 1))
  ckpt=$(find "$LORA_ROOT" -mindepth 2 -maxdepth 2 -type d -name checkpoint-312 2>/dev/null | sort | tail -1 || true)
  st=$(job_line || true)
  if (( i % 5 == 1 )); then
    echo "[$(date)] poll#$i ckpt=${ckpt:-none}"
    echo "  $st"
  fi
  if [[ -n "$ckpt" && -f "$ckpt/adapter_model.safetensors" && -f "$ckpt/adapter_config.json" ]]; then
    if echo "$st" | grep -q SUCCEEDED; then
      ADAPTERS=$ckpt
      echo "[$(date)] train done: $ADAPTERS"
      break
    fi
    if echo "$st" | grep -qE 'FAILED|CANCELLED'; then
      echo "[$(date)] job died after writing ckpt; abort" >&2
      echo "$st" >&2
      exit 1
    fi
  fi
  if echo "$st" | grep -qE 'FAILED|CANCELLED' && [[ -z "$ckpt" ]]; then
    echo "[$(date)] SFT failed without checkpoint-312" >&2
    exit 1
  fi
  sleep 60
done
echo "$ADAPTERS" > "$STATE_FILE"
sleep 30

echo "[$(date)] submit merge ADAPTERS=$ADAPTERS"
ADAPTERS="$ADAPTERS" OUTPUT_DIR="$MERGED" EXPERIMENT_NAME=reform_qwen35_9b_merge_v5 \
  bash "$SCRIPTS/srun_cci_merge_qwen35_9b_reform.sh"

for i in $(seq 1 90); do
  if [[ -f "$MERGED/config.json" && -f "$MERGED/tokenizer.json" && -f "$MERGED/preprocessor_config.json" ]] \
     && compgen -G "$MERGED/model-*.safetensors" > /dev/null; then
    echo "[$(date)] merge files present"
    break
  fi
  if (( i % 6 == 1 )); then
    echo "[$(date)] wait merge files #$i"
  fi
  if (( i == 90 )); then
    echo "[$(date)] merge timed out" >&2
    exit 1
  fi
  sleep 20
done
sleep 20

MERGE_APP=$(sco cci apps list --workspace-name "$WORKSPACE" --keyword reform_qwen35_9b_merge_v5 2>/dev/null \
  | awk -F'[()]' '/app-[a-z0-9]+/ {print $2; exit}' || true)
echo "[$(date)] merge app=${MERGE_APP:-unknown}"
if [[ -n "${MERGE_APP:-}" ]]; then
  sco cci apps delete "$MERGE_APP" --workspace-name "$WORKSPACE" || true
fi

echo "[$(date)] submit LightLLM v5"
SOURCE_MODEL="$MERGED" MODEL_NAME=qwen35-9b-reform-sft-v5 \
  EXPERIMENT_NAME=reform_qwen35_9b_lightllm_v5 \
  bash "$SCRIPTS/srun_cci_lightllm_qwen35_9b_reform.sh"

LLM_LOG=$(ls -t "$SFT"/logs/cci_lightllm_reform_qwen35_9b_lightllm_v5_*.log 2>/dev/null | head -1 || true)
echo "[$(date)] lightllm log=${LLM_LOG:-none}"
IP=""
for i in $(seq 1 180); do
  if [[ -n "${LLM_LOG:-}" && -f "$LLM_LOG" ]]; then
    IP=$(rg -a -o 'ips=[0-9.]+' "$LLM_LOG" 2>/dev/null | tail -1 | cut -d= -f2 || true)
  fi
  if [[ -n "$IP" ]] && curl -sf --max-time 5 "http://${IP}:8000/v1/models" | grep -q 'qwen35-9b-reform-sft-v5'; then
    echo "[$(date)] LightLLM ready http://$IP:8000/v1"
    break
  fi
  if (( i % 6 == 0 )); then
    echo "[$(date)] wait LightLLM #$i ip=${IP:-none}"
  fi
  sleep 20
done
if [[ -z "$IP" ]] || ! curl -sf --max-time 5 "http://${IP}:8000/v1/models" | grep -q 'qwen35-9b-reform-sft-v5'; then
  echo "[$(date)] LightLLM not ready" >&2
  exit 1
fi

curl -sf --max-time 5 "${SEARCH_URL}/health" >/dev/null || { echo "search down $SEARCH_URL" >&2; exit 1; }
curl -sf --max-time 5 "${VERIFY_URL}/health" >/dev/null || { echo "kimina down $VERIFY_URL" >&2; exit 1; }

export MODEL_URL="http://${IP}:8000/v1"
export MODEL_NAME=qwen35-9b-reform-sft-v5
export SEARCH_URL VERIFY_URL
export VERIFY_KEY=reform-lean-key
export WORKERS=16 SPLIT=all EVAL_PROMPT_PATCH=0
export ROLLOUT_TOKENIZER="$MERGED/tokenizer.json"

echo "[$(date)] eval prompt check (must match train system):"
python3 - <<'PY'
import sys
sys.path.insert(0, "/mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train")
from prompts import format_system_prompt, format_user_prompt
s = format_system_prompt("physics")
u = format_user_prompt("demo problem", "physics")
assert s.startswith("You are an autoformalization assistant.")
assert "# Tools" in s
assert "Not Lean names" not in s
assert "MUST" not in s
assert "# Tools" not in u
print(s)
print("--- user ---")
print(u)
PY

echo "[$(date)] start v5 full eval"
cd "$SFT"
bash "$SCRIPTS/eval_v5_rollout_sequence.sh"
echo "[$(date)] v5 pipeline finished"
