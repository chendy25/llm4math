#!/bin/bash

# Launch SGLang server for ReForm data collection
# This model serves both autoformalization and semantic consistency checking

model_path=/mnt/afs_rl/yinzixin/models/Qwen/Qwen3-30B-A3B-Thinking-2507 

# Check if model path exists
if [ ! -d "$model_path" ]; then
    echo "Error: Model path does not exist: $model_path"
    exit 1
fi

echo "Starting SGLang server with model: $model_path"

python -m sglang.launch_server \
    --model-path ${model_path} \
    --tp 1 \
    --trust-remote-code \
    --context-length 32768 \
    --max-running-requests 64 \
    --port 8000 \
    --host 0.0.0.0

