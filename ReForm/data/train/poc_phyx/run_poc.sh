#!/usr/bin/env bash
set -euo pipefail
cd /mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export OPENAI_BASE_URL="https://tokenhub.sensetime.com/v1"
export OPENAI_API_KEY="sk-3iRWHXSB3D6TkQtckXQgV3vSRtGLYn3kafhRyZ8HryCOJyG6"
export PYTHONUNBUFFERED=1
PY="${PY:-/opt/conda/bin/python3}"
# prefer leansearch if openai imports quickly
if /mnt/afs_reason/chendongyang/conda/leansearch/bin/python -c "import openai" 2>/dev/null; then
  PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
fi
echo "Using PY=$PY"
$PY -c "import openai; print('openai', openai.__version__)"
$PY run_collect_sft_data.py \
  --input /mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train/poc_phyx/phyx_poc_em1.jsonl \
  --output /mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train/poc_phyx/sft_phyx_physsearch_poc.jsonl \
  --max-items 1 \
  --max-rounds 1 \
  --max-tool-calls 3 \
  --use-physsearch \
  --physsearch-url http://127.0.0.1:8000 \
  --auto-model gpt-4o-mini \
  --auto-base-url "https://tokenhub.sensetime.com/v1" \
  --auto-api-key "sk-3iRWHXSB3D6TkQtckXQgV3vSRtGLYn3kafhRyZ8HryCOJyG6" \
  --critic-model gpt-4o-mini \
  --critic-base-url "https://tokenhub.sensetime.com/v1" \
  --critic-api-key "sk-3iRWHXSB3D6TkQtckXQgV3vSRtGLYn3kafhRyZ8HryCOJyG6" \
  2>&1 | tee /mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train/poc_phyx/run_poc.log
echo EXIT:$?
