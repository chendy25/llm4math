#!/bin/bash
# CCI: Qwen3.5-9B ReForm merged HF · agentic-rl · 1×H800 · LightLLM FP8
#
#   bash scripts/srun_cci_lightllm_qwen35_9b_reform.sh
set -euo pipefail

ROOT=/mnt/afs_reason/chendongyang/code
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_PER_NODE=${GPU_PER_NODE:-1}
CPU=${CPU:-14}
MEMORY=${MEMORY:-240GiB}
REPLICAS=1
PORT=${PORT:-8000}

PARTITION="${PARTITION:-agentic-rl}"
WORKSPACE=${WORKSPACE:-rl}
AVAILABLE_ZONE=cn-tj-01a
VPC_ID=019e25be-ec63-7831-98f8-3993ee8be3ab
WORKER_SPEC="N4lS.Iu.I80.${GPU_PER_NODE}"

CONTAINER=registry.cn-tj-01.sensecore.cn/ccr-tj/lightllm:qwen35-27b-889e0d9-cu128

AFS_REASON=019eca36-8694-7f33-b861-c1aafe34074d:/mnt/afs_reason
AFS_SHARE=019eee5e-9743-71ca-892e-3c1c2df74fd3:/mnt/afs_share
AFS_AGENTS=019ed8e0-f4f8-71f7-830c-f8a1e667a9ec:/mnt/afs_agents
AFS_RL=019eab19-b6af-7651-978e-f14eb4b81855:/mnt/afs_rl

DEPLOY="$SCRIPT_DIR/deploy_lightllm_qwen35_9b_reform.sh"
SOURCE_MODEL=${SOURCE_MODEL:-$ROOT/llm4math/sft/output/qwen35_9b_reform_merged_v5/checkpoint-312}
MODEL_NAME=${MODEL_NAME:-qwen35-9b-reform-sft-v5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-reform_qwen35_9b_lightllm_v5}
LOG_DIR="$ROOT/llm4math/sft/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cci_lightllm_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

CONFIG_FILE="$SCRIPT_DIR/cci_${EXPERIMENT_NAME}.generated.yaml"
CMD="mkdir -p /mnt/afs && ln -sfn /mnt/afs_share/users/chendongyang /mnt/afs/chendongyang && set -o pipefail && export SOURCE_MODEL=${SOURCE_MODEL} MODEL_NAME=${MODEL_NAME} && bash ${DEPLOY} 2>&1 | tee ${LOG_FILE}"

cat > "$CONFIG_FILE" <<EOF
display_name: ${EXPERIMENT_NAME}
resource_pool:
  name: ${PARTITION}
  available_zone: ${AVAILABLE_ZONE}
  vpc_id: ${VPC_ID}
replicas: ${REPLICAS}
template:
  containers:
    - name: lightllm
      image_path: ${CONTAINER}
      resource_request:
        nvidia.com/gpu: "${GPU_PER_NODE}"
        cpu: "${CPU}"
        memory: ${MEMORY}
      volume_mounts:
        - type: PV_AFS
          id: ${AFS_REASON%%:*}
          mount_path: ${AFS_REASON#*:}
          subdir: /
          zone: ${AVAILABLE_ZONE}
          display_name: afs_reason
        - type: PV_AFS
          id: ${AFS_SHARE%%:*}
          mount_path: ${AFS_SHARE#*:}
          subdir: /
          zone: ${AVAILABLE_ZONE}
          display_name: afs_share
        - type: PV_AFS
          id: ${AFS_AGENTS%%:*}
          mount_path: ${AFS_AGENTS#*:}
          subdir: /
          zone: ${AVAILABLE_ZONE}
          display_name: afs_agents
        - type: PV_AFS
          id: ${AFS_RL%%:*}
          mount_path: ${AFS_RL#*:}
          subdir: /
          zone: ${AVAILABLE_ZONE}
          display_name: afs_rl
      command:
        - bash
        - -c
        - "${CMD}"
  resource_spec:
    name: ${WORKER_SPEC}
scheduling:
  priority: HIGH
  quota_type: RESERVED
  instance_affinity: POD_AFFINITY
EOF

echo "CCI config: $CONFIG_FILE"
echo "Source:     $SOURCE_MODEL"
echo "Name:       $MODEL_NAME"
echo "Log:        $LOG_FILE"
sco cci apps create \
    --workspace-name "$WORKSPACE" \
    --config "$CONFIG_FILE" \
    --ports "$PORT"

echo "submitted $EXPERIMENT_NAME"
echo "  sco cci apps list --workspace-name $WORKSPACE --keyword $EXPERIMENT_NAME"
echo "after READY: MODEL_URL=http://<POD_IP>:${PORT}/v1 bash $SCRIPT_DIR/eval_all_testsets.sh"
