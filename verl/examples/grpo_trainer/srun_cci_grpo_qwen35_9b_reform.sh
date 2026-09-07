#!/bin/bash
# ACP: native verl GRPO on ReForm SFT v2 · 8×H800
#
#   bash examples/grpo_trainer/srun_cci_grpo_qwen35_9b_reform.sh
set -euo pipefail

ROOT=/mnt/afs_reason/chendongyang/code
VERL_ROOT=$ROOT/llm4math/verl
SCRIPT=$VERL_ROOT/examples/grpo_trainer/run_qwen35_9b_reform_fsdp.sh

NNODES=${NNODES:-1}
GPU_PER_NODE=${GPU_PER_NODE:-8}
PARTITION=${PARTITION:-toolcall}
WORKSPACE=${WORKSPACE:-rl}
name=${JOB_NAME:-reform_qwen35_9b_grpo_v2}

CONTAINER=registry.cn-tj-01.sensecore.cn/ccr-devsfttj/verl:cu129_lightllm_sandbox_megatron0.14.0_vllm0.13.0R3_torch2.9.0_fa3_te2.5.0_0211
MOUNT=019eca36-8694-7f33-b861-c1aafe34074d:/mnt/afs_reason
spec_name=${WORKER_SPEC:-N4lS.Iu.I80.${GPU_PER_NODE}}

LOG_DIR=$ROOT/llm4math/sft/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

echo "Job name:  $name"
echo "GPUs/node: $GPU_PER_NODE"
echo "Partition: $PARTITION"
echo "Log:       $LOG_FILE"
echo "VERIFY_URL defaults to CCI Kimina http://10.120.5.33:8001"

sco acp jobs create \
    --workspace-name "$WORKSPACE" \
    --aec2-name "$PARTITION" \
    --container-image-url "$CONTAINER" \
    --storage-mount "$MOUNT" \
    --training-framework pytorch \
    --worker-spec "$spec_name" \
    -N ${NNODES} \
    -j "$name" \
    --priority HIGH \
    --command="unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; export GPU_PER_NODE=${GPU_PER_NODE} NNODES=${NNODES} VERIFY_URL=http://10.120.5.33:8001 SEARCH_URL=http://10.120.1.119:8000 && bash $SCRIPT 2>&1 | tee $LOG_FILE"
