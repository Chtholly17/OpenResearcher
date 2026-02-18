#!/bin/bash
# Launch multi-turn GRPO training for OpenResearcher
#
# Prerequisites:
#   1. Search service running: bash scripts/start_search_service.sh bm25 8090
#   2. Data preprocessed: python verl_rl/preprocess_openresearcher.py \
#        --hf_dataset OpenResearcher/OpenResearcher-Dataset --hf_subset seed_42 \
#        --local_save_dir ~/data/openresearcher
#   3. verl installed: pip install verl (or cloned at ../verl)
#
# Usage:
#   # Default: 8 GPUs, TP=4 for rollout
#   bash verl_rl/run_grpo_training.sh
#
#   # Custom GPU count
#   N_GPUS=4 bash verl_rl/run_grpo_training.sh
#
#   # Override any config via command line
#   bash verl_rl/run_grpo_training.sh trainer.total_epochs=3 algorithm.grpo_n_generations=8

set -x
ulimit -n 65535

# ---- Configuration ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${VERL_DIR:-$(cd "$PROJECT_DIR/../verl" && pwd)}"
CONFIG_PATH="$PROJECT_DIR/verl_rl/config"

N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"

# Model: 30B-A3B MoE with TP=4 for rollout (2 server groups)
TP_SIZE="${TP_SIZE:-4}"

# Data paths
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/openresearcher/train.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/openresearcher/test.parquet}"

# Search service URL (must be running)
SEARCH_SERVICE_URL="${SEARCH_SERVICE_URL:-http://127.0.0.1:8090}"

# Tool, interaction, and reward config paths (absolute)
TOOL_CONFIG="$CONFIG_PATH/tool_config/openresearcher_tool_config.yaml"
INTERACTION_CONFIG="$CONFIG_PATH/interaction_config/openresearcher_interaction_config.yaml"
REWARD_MODULE="$PROJECT_DIR/verl_rl/reward/openresearcher_reward.py"

# Experiment naming
TIMESTAMP=$(date '+%m%d-%H%M')
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_multiturn_${TIMESTAMP}}"

# ---- Validate prerequisites ----
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Run: python verl_rl/preprocess_openresearcher.py --hf_dataset OpenResearcher/OpenResearcher-Dataset --hf_subset seed_42 --local_save_dir ~/data/openresearcher"
    exit 1
fi

if [ ! -f "$TOOL_CONFIG" ]; then
    echo "ERROR: Tool config not found at $TOOL_CONFIG"
    exit 1
fi

# Check search service is reachable
if ! curl -s --max-time 5 "$SEARCH_SERVICE_URL" > /dev/null 2>&1; then
    echo "WARNING: Search service at $SEARCH_SERVICE_URL may not be running."
    echo "Rollouts will fail if the service is unavailable."
    echo "Start it with: bash scripts/start_search_service.sh bm25 8090"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "=========================================="
echo "OpenResearcher Multi-Turn GRPO Training"
echo "=========================================="
echo "Project dir:    $PROJECT_DIR"
echo "verl dir:       $VERL_DIR"
echo "Config path:    $CONFIG_PATH"
echo "Train data:     $TRAIN_DATA"
echo "Val data:       $VAL_DATA"
echo "GPUs:           $N_GPUS (TP=$TP_SIZE)"
echo "Nodes:          $NNODES"
echo "Search service: $SEARCH_SERVICE_URL"
echo "Experiment:     $EXPERIMENT_NAME"
echo "=========================================="

# ---- Launch training ----
# Add project dir to PYTHONPATH so Ray workers can import verl_rl tools/reward
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Disable Ray's OOM killer (false positive with 1TB+ RAM machines)
export RAY_memory_monitor_refresh_ms=0

# Help PyTorch manage fragmented GPU memory (avoids OOM on the 30B MoE model)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Working directory must be the verl root for hydra config resolution
cd "$VERL_DIR"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='openresearcher_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=OpenResearcher/OpenResearcher-30B-A3B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG" \
    custom_reward_function.path="$REWARD_MODULE" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.project_name='openresearcher_rl' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    "$@"
