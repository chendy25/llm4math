#!/bin/bash
set -euo pipefail

# ReForm SFT-LoRA on local Qwen3.5-9B (ms-swift). v5 = system-turn tool schema.
ROOT=/mnt/afs_reason/chendongyang/code
MODEL_PATH=${MODEL_PATH:-$ROOT/data/models/Qwen3.5-9B}
DATASET=${DATASET:-$ROOT/llm4math/sft/data/train_mix_tools_v5.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/llm4math/sft/output/qwen35_9b_reform_lora_v5}

NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
VIT_LR=${VIT_LR:-1e-5}
ALIGNER_LR=${ALIGNER_LR:-1e-5}
VIT_LAST_N=${VIT_LAST_N:-4}
ENABLE_VIT=${ENABLE_VIT:-1}
INSTALL_DEPS=${INSTALL_DEPS:-0}
MAX_LENGTH=${MAX_LENGTH:-16384}
LOSS_SCALE=${LOSS_SCALE:-ignore_tool_response}
ADD_NON_THINKING_PREFIX=${ADD_NON_THINKING_PREFIX:-false}
EVAL_STEPS=${EVAL_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-50}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
ADAPTERS=${ADAPTERS:-}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
GPU_PER_NODE=${GPU_PER_NODE:-4}
NNODES=${NNODES:-1}

SWIFT=${SWIFT:-/mnt/afs_reason/chendongyang/conda/ms-swift/bin/swift}
CONDA_PY=${CONDA_PY:-/mnt/afs_reason/chendongyang/conda/ms-swift/bin/python}

if [ ! -x "$SWIFT" ]; then
    echo "swift missing: $SWIFT" >&2
    exit 1
fi

# Re-apply local loss_scale plugin (AFS has wiped other.py before).
PATCH_DIR=$ROOT/llm4math/sft/patches/ms-swift
MS_SWIFT=$ROOT/ms-swift
if [ -f "$PATCH_DIR/swift/loss_scale/other.py" ]; then
    mkdir -p "$MS_SWIFT/swift/loss_scale/config"
    cp -f "$PATCH_DIR/swift/loss_scale/other.py" "$MS_SWIFT/swift/loss_scale/other.py"
    cp -f "$PATCH_DIR/swift/loss_scale/config/ignore_tool_response.json" \
        "$MS_SWIFT/swift/loss_scale/config/ignore_tool_response.json"
    cp -f "$PATCH_DIR/swift/loss_scale/mapping.py" "$MS_SWIFT/swift/loss_scale/mapping.py"
    echo "[sft] applied ms-swift loss_scale patch from $PATCH_DIR"
fi
export PYTHONPATH=$MS_SWIFT${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$OUTPUT_DIR"
export NPROC_PER_NODE=${GPU_PER_NODE}
export NNODES
export MASTER_PORT=${MASTER_PORT:-29501}

VIT_FLAGS=()
if [ "$ENABLE_VIT" = "1" ]; then
    # Last 4 ViT blocks + merger, same as v2 (blocks.23-26).
    VIT_FLAGS+=(
        --freeze_vit true
        --freeze_aligner true
        --modules_to_save "blocks.23 blocks.24 blocks.25 blocks.26 merger"
        --vit_lr "$VIT_LR"
        --aligner_lr "$ALIGNER_LR"
    )
fi

RESUME_FLAGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    RESUME_FLAGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
ADAPTER_FLAGS=()
if [ -n "$ADAPTERS" ]; then
    ADAPTER_FLAGS+=(--adapters "$ADAPTERS")
fi
MAX_STEPS_FLAGS=()
if [ -n "${MAX_STEPS:-}" ]; then
    MAX_STEPS_FLAGS+=(--max_steps "$MAX_STEPS")
fi

echo "[sft] dataset=$DATASET"
echo "[sft] output=$OUTPUT_DIR"
echo "[sft] gpus=$GPU_PER_NODE nnodes=$NNODES"

"$SWIFT" sft \
    --model "$MODEL_PATH" \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --max_length "$MAX_LENGTH" \
    --truncation_strategy delete \
    --loss_scale "$LOSS_SCALE" \
    --add_non_thinking_prefix "$ADD_NON_THINKING_PREFIX" \
    --split_dataset_ratio 0.01 \
    --eval_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 3 \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --dataloader_num_workers 8 \
    --use_hf false \
    "${VIT_FLAGS[@]}" \
    "${RESUME_FLAGS[@]}" \
    "${ADAPTER_FLAGS[@]}" \
    "${MAX_STEPS_FLAGS[@]}"
