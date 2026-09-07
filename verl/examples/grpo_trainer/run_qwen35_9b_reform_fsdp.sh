#!/usr/bin/env bash
# Native verl GRPO on ReForm SFT v2.
# Generate uses tool_agent: Hermes parses <tool_call>, executes leansearch /
# physsearch / web_search, splices <tool_response>. Prompt stays v2 (no catalog).
# Reward = 0.5*compile + 0.5*semantic (BOTH=1). Kimina = CCI.
#
#   bash examples/grpo_trainer/run_qwen35_9b_reform_fsdp.sh
#   bash examples/grpo_trainer/srun_cci_grpo_qwen35_9b_reform.sh
set -xeuo pipefail

VERL_ROOT=/mnt/afs_reason/chendongyang/code/llm4math/verl
REFORM_TRAIN=${REFORM_TRAIN:-/mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train}
DATA_DIR=${DATA_DIR:-$VERL_ROOT/data/reform_rl}
MODEL_PATH=${MODEL_PATH:-/mnt/afs_reason/chendongyang/code/llm4math/sft/output/qwen35_9b_reform_merged_v2/checkpoint-312}
TOKENHUB_ENV=${TOKENHUB_ENV:-$REFORM_TRAIN/poc_phyx/.tokenhub.env}
SERPER_ENV=${SERPER_ENV:-$REFORM_TRAIN/poc_phyx/.serper.env}

if [[ -f "$TOKENHUB_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$TOKENHUB_ENV"
  set +a
fi
if [[ -f "$SERPER_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$SERPER_ENV"
  set +a
fi

export CRITIC_MODEL=${CRITIC_MODEL:-qwen3.7-plus}
export CRITIC_BASE_URL=${CRITIC_BASE_URL:-https://tokenhub.sensetime.com/v1}
export CRITICLEAN_BASE_URL=${CRITICLEAN_BASE_URL:-$CRITIC_BASE_URL}
export CRITIC_API_KEY=${CRITIC_API_KEY:-${TOKENHUB_API_KEY:-}}
export CRITICLEAN_API_KEY=${CRITICLEAN_API_KEY:-$CRITIC_API_KEY}
export CRITIC_NO_THINKING=${CRITIC_NO_THINKING:-1}

export SEARCH_URL=${SEARCH_URL:-http://10.120.1.119:8000}
export VERIFY_URL=${VERIFY_URL:-http://10.120.5.33:8001}
export VERIFY_KEY=${VERIFY_KEY:-reform-lean-key}
export VERIFY_MAX_CONCURRENCY=${VERIFY_MAX_CONCURRENCY:-4}
export VERIFY_TIMEOUT=${VERIFY_TIMEOUT:-900}
export VERIFY_CANONICAL_HEADER=${VERIFY_CANONICAL_HEADER:-1}
export REFORM_TRAIN
export PYTHONPATH="${VERL_ROOT}:${REFORM_TRAIN}:${PYTHONPATH:-}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

cd "$VERL_ROOT"
export VERL_USE_UV=0

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
INFER_BACKEND=${INFER_BACKEND:-vllm}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_9b_reform_grpo_v2}
PROJECT_NAME=${PROJECT_NAME:-reform_grpo}
CKPT_DIR=${CKPT_DIR:-$VERL_ROOT/checkpoints/reform_grpo/${EXPERIMENT_NAME}}

TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val.parquet}
TOOL_CFG=${TOOL_CFG:-$VERL_ROOT/examples/grpo_trainer/reform_tool_config.yaml}
AGENT_CFG=${AGENT_CFG:-$VERL_ROOT/examples/grpo_trainer/reform_agent_loop.yaml}
REWARD_FILE=${REWARD_FILE:-$VERL_ROOT/verl/utils/reward_score/reform_rollout.py}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=${TRAIN_BATCH_SIZE:-16} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH:-2048} \
  data.max_response_length=${MAX_RESPONSE_LENGTH:-8192} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.image_key=images \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name="$INFER_BACKEND" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.5} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N:-8} \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1536 \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CFG" \
  actor_rollout_ref.rollout.agent.default_agent_loop=reform_tool_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_CFG" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="$REWARD_FILE" \
  reward.custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.save_freq=${SAVE_FREQ:-10} \
  trainer.test_freq=${TEST_FREQ:-10} \
  trainer.total_epochs=${TOTAL_EPOCHS:-2} \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}
