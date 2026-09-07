#!/bin/bash
# =====================================================================
# CCI 部署：LeanSearch-v2（Qwen3-Embedding-8B + Qwen3-Reranker-8B）
# 分区: code · 2×H800 · uvicorn（不是 vLLM）
# =====================================================================
# 用法:
#   bash scripts/srun_cci_leansearch.sh
#
# 起来后:
#   sco cci apps list --workspace-name rl --keyword leansearch_dual_corpus
#   把新 POD IP 写回 SEARCH_URL=http://<POD_IP>:8000
# =====================================================================

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_PER_NODE=2
CPU=28
MEMORY=480GiB
REPLICAS=1
PORT=8000

PARTITION="${PARTITION:-code}"
WORKSPACE=rl
AVAILABLE_ZONE=cn-tj-01a
VPC_ID=019e25be-ec63-7831-98f8-3993ee8be3ab
WORKER_SPEC="N4lS.Iu.I80.${GPU_PER_NODE}"

CONTAINER=registry.cn-tj-01.sensecore.cn/ccr-devsfttj/verl:cu129_lightllm_sandbox_megatron0.14.0_vllm0.13.0R3_torch2.9.0_fa3_te2.5.0_0211

AFS_REASON=019eca36-8694-7f33-b861-c1aafe34074d:/mnt/afs_reason
AFS_SHARE=019eee5e-9743-71ca-892e-3c1c2df74fd3:/mnt/afs_share
AFS_AGENTS=019ed8e0-f4f8-71f7-830c-f8a1e667a9ec:/mnt/afs_agents
AFS_RL=019eab19-b6af-7651-978e-f14eb4b81855:/mnt/afs_rl

DEPLOY="$SCRIPT_DIR/deploy_leansearch.sh"

EXPERIMENT_NAME=leansearch_dual_corpus
LOG_DIR="$REPO/data/cuvs/physlib/cci_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cci_$(date +%Y%m%d_%H%M%S).log"
echo "日志将输出到: $LOG_FILE"

CONFIG_FILE="$SCRIPT_DIR/cci_${EXPERIMENT_NAME}.generated.yaml"
CMD="mkdir -p /mnt/afs && ln -sfn /mnt/afs_share/users/chendongyang /mnt/afs/chendongyang && set -o pipefail && bash ${DEPLOY} 2>&1 | tee ${LOG_FILE}"

cat > "$CONFIG_FILE" <<EOF
display_name: ${EXPERIMENT_NAME}
resource_pool:
  name: ${PARTITION}
  available_zone: ${AVAILABLE_ZONE}
  vpc_id: ${VPC_ID}
replicas: ${REPLICAS}
template:
  containers:
    - name: leansearch
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
sco cci apps create \
    --workspace-name "$WORKSPACE" \
    --config "$CONFIG_FILE" \
    --ports "$PORT"

echo "提交完成: $EXPERIMENT_NAME"
echo "  sco cci apps list --workspace-name $WORKSPACE --keyword $EXPERIMENT_NAME"
echo "日志: $LOG_FILE"
echo "起来后把 SEARCH_URL 改成 http://<POD_IP>:8000"
