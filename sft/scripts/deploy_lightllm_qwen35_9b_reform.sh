#!/usr/bin/env bash
# LightLLM serve: merged ReForm Qwen3.5-9B (multimodal).
# Do NOT prepend conda/ms-swift or conda/opd to PATH.
set -euo pipefail

export LOADWORKER=18
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1

SOURCE_MODEL=${SOURCE_MODEL:-/mnt/afs_reason/chendongyang/code/llm4math/sft/output/qwen35_9b_reform_merged_v3/checkpoint-312}
MODEL_NAME=${MODEL_NAME:-qwen35-9b-reform-sft-v3}
MODEL_PATH=${MODEL_PATH:-/root/${MODEL_NAME}}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-/mnt/afs_agents/hongjiawei/tools/lightllm_qwen35/chat-template_multimodal.jinja}
PORT=${PORT:-8000}
TP=${TP:-1}
VISUAL_DP=${VISUAL_DP:-1}

echo "============================================================"
echo "  LightLLM ReForm Qwen3.5-9B"
echo "  Source: $SOURCE_MODEL"
echo "  Local:  $MODEL_PATH"
echo "  Name:   $MODEL_NAME  tp=$TP visual_dp=$VISUAL_DP"
echo "  hostname=$(hostname) ips=$(hostname -I 2>/dev/null || true)"
echo "============================================================"
ip -4 addr show 2>/dev/null | awk '/inet /{print "[deploy] inet",$2}' || true

if [ ! -f "${SOURCE_MODEL}/config.json" ]; then
    echo "Missing config.json under ${SOURCE_MODEL}" >&2
    exit 1
fi
if ! compgen -G "${SOURCE_MODEL}/*.safetensors" > /dev/null; then
    echo "No *.safetensors under ${SOURCE_MODEL} (need merged HF)" >&2
    exit 1
fi
if [ ! -f "${SOURCE_MODEL}/preprocessor_config.json" ]; then
    echo "Missing preprocessor_config.json under ${SOURCE_MODEL}" >&2
    exit 1
fi
if [ ! -f "${CHAT_TEMPLATE}" ]; then
    echo "Missing chat template: ${CHAT_TEMPLATE}" >&2
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    mkdir -p "${MODEL_PATH}"
    ls "${SOURCE_MODEL}" | xargs -I {} -P 32 cp "${SOURCE_MODEL}/{}" "${MODEL_PATH}/{}"
else
    echo "Directory already exists: ${MODEL_PATH}"
fi

python -m lightllm.server.api_server \
    --model_dir "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --tp "${TP}" \
    --visual_dp "${VISUAL_DP}" \
    --graph_max_batch_size 64 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --nccl_port 28765 \
    --mem_fraction 0.75 \
    --max_image_token_count 4096 \
    --max_image_pixels 3686400 \
    --batch_max_tokens 8192 \
    --quant_type triton-fp8w8a8-pertensor \
    --chat_template "${CHAT_TEMPLATE}"

sleep infinity
