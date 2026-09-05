#!/bin/bash
# Submit ReForm Qwen3.5-9B SFT-LoRA (tool-prompt mix v3).
#
#   bash scripts/srun_sft_qwen35_9b_lora.sh
set -euo pipefail

ROOT=/mnt/afs_reason/chendongyang/code
NNODES=${NNODES:-1}
SMOKE=${SMOKE:-0}
PARTITION=${PARTITION:-agentic-rl}
WORKSPACE=${WORKSPACE:-rl}

if [ "$SMOKE" = "1" ]; then
    GPU_PER_NODE=${GPU_PER_NODE:-1}
    SCRIPT=$ROOT/llm4math/sft/scripts/sft_qwen35_9b_lora.sh
    name="reform_qwen35_9b_sft_lora_v3_smoke"
    DATASET=$ROOT/llm4math/sft/data/train_smoke_tools.jsonl
    OUTPUT_DIR=$ROOT/llm4math/sft/output/qwen35_9b_reform_lora_v3_smoke
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
else
    GPU_PER_NODE=${GPU_PER_NODE:-4}
    SCRIPT=$ROOT/llm4math/sft/scripts/sft_qwen35_9b_lora.sh
    name=${JOB_NAME:-reform_qwen35_9b_sft_lora_v3}
    DATASET=${DATASET:-$ROOT/llm4math/sft/data/train_mix_tools.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/llm4math/sft/output/qwen35_9b_reform_lora_v3}
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
fi

CONTAINER=registry.cn-tj-01.sensecore.cn/ccr-devsfttj/verl:cu129_lightllm_sandbox_megatron0.14.0_vllm0.13.0R3_torch2.9.0_fa3_te2.5.0_0211
MOUNT=019eca36-8694-7f33-b861-c1aafe34074d:/mnt/afs_reason
spec_name=${WORKER_SPEC:-N4lS.Iu.I80.${GPU_PER_NODE}.16c256g}

LOG_DIR=$ROOT/llm4math/sft/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

echo "Job name:   $name"
echo "Nodes:      $NNODES"
echo "GPUs/node:  $GPU_PER_NODE"
echo "Partition:  $PARTITION"
echo "Dataset:    $DATASET"
echo "Output:     $OUTPUT_DIR"
echo "Log file:   $LOG_FILE"

sco acp jobs create \
    --workspace-name  "$WORKSPACE" \
    --aec2-name "$PARTITION" \
    --container-image-url "$CONTAINER" \
    --storage-mount "$MOUNT" \
    --training-framework pytorch \
    --worker-spec "$spec_name" \
    -N ${NNODES} \
    -j "$name" \
    --priority HIGH \
    --command="bash $ROOT/easel/sft/fix_conda_env_permissions.sh /mnt/afs_reason/chendongyang/conda/ms-swift && export GPU_PER_NODE=${GPU_PER_NODE} NNODES=${NNODES} DATASET=${DATASET} OUTPUT_DIR=${OUTPUT_DIR} VIT_LAST_N=${VIT_LAST_N:-4} ENABLE_VIT=${ENABLE_VIT:-1} SAVE_STEPS=${SAVE_STEPS:-50} GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS} && bash $SCRIPT 2>&1 | tee $LOG_FILE"
