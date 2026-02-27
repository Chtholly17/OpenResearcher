#!/bin/bash
# Launch multi-turn GRPO training for OpenResearcher — Qwen3-8B
#
# Key differences from Nemotron (30B-A3B) version:
#   - Model: Qwen/Qwen3-8B (8B dense vs 30B MoE)
#   - Tool format: hermes (standard JSON) vs nemotron (XML)
#   - System prompt: <answer> tags as primary format (see --model_type qwen3)
#   - No trust_remote_code required
#   - Fits in 2× A100-80GB for rollout (TP=2), or even 1× for small context
#   - No FSDP offload needed (8B model fits in 80GB comfortably)
#   - Full 3-policy GRPO with KL loss (ref policy fits easily)
#
# Prerequisites:
#   1. Generate Qwen3-format training data:
#        python verl_rl/preprocess_openresearcher.py \
#          --hf_dataset PahaII/openresearcher-training-data \
#          --model_type qwen3 \
#          --local_save_dir data/qwen3
#      (The existing Nemotron parquets use a different system prompt and won't work)
#   2. Dense search service running on port 8090
#   3. conda activate openresearcher
#
# Usage:
#   bash verl_rl/run_grpo_training_qwen3.sh
#
#   # Override data / experiment name:
#   TRAIN_DATA=data/qwen3/train_curriculum_500.parquet \
#   EXPERIMENT_NAME=grpo_qwen3_500 \
#   bash verl_rl/run_grpo_training_qwen3.sh

set -x
ulimit -n 65535

# ---- Configuration ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${VERL_DIR:-$(cd "$PROJECT_DIR/../verl" && pwd)}"
CONFIG_PATH="$PROJECT_DIR/verl_rl/config"

N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"

# TP=2: Qwen3-8B fits in a single 80GB GPU, TP=2 gives 4 server groups on 8 GPUs
TP_SIZE="${TP_SIZE:-2}"

# Data paths — must be generated with --model_type qwen3
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/qwen3/train.parquet}"
VAL_DATA="${VAL_DATA:-$PROJECT_DIR/data/qwen3/test.parquet}"

SEARCH_SERVICE_URL="${SEARCH_SERVICE_URL:-http://127.0.0.1:8090}"

TOOL_CONFIG="$CONFIG_PATH/tool_config/openresearcher_tool_config.yaml"
INTERACTION_CONFIG="$CONFIG_PATH/interaction_config/openresearcher_interaction_config.yaml"
REWARD_MODULE="$PROJECT_DIR/verl_rl/reward/openresearcher_reward.py"

TIMESTAMP=$(date '+%m%d-%H%M')
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3_${TIMESTAMP}}"

# ---- Download data if not present ----
HF_DATA_REPO="PahaII/openresearcher-training-data"
DATA_DIR="$PROJECT_DIR/data"
QWEN3_DATA_DIR="$PROJECT_DIR/data/qwen3"
DATA_FILES=(
    "train_curriculum_500.parquet"
    "train_curriculum_1k.parquet"
    "train_curriculum_2k.parquet"
    "train_no_passrate1.parquet"
    "test_20.parquet"
)

# Download source parquets if needed, then regenerate with qwen3 prompts
missing_source=()
for f in "${DATA_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        missing_source+=("$f")
    fi
done

if [ ${#missing_source[@]} -gt 0 ]; then
    echo "Downloading missing source data files from $HF_DATA_REPO..."
    mkdir -p "$DATA_DIR"
    for f in "${missing_source[@]}"; do
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$HF_DATA_REPO', filename='$f', repo_type='dataset', local_dir='$DATA_DIR')
" || { echo "ERROR: Failed to download $f"; exit 1; }
    done
fi

# ---- Validate prerequisites ----
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo ""
    echo "Generate Qwen3 training data first:"
    echo "  python verl_rl/preprocess_openresearcher.py \\"
    echo "    --hf_dataset PahaII/openresearcher-training-data \\"
    echo "    --model_type qwen3 \\"
    echo "    --local_save_dir $QWEN3_DATA_DIR"
    echo ""
    echo "Or point TRAIN_DATA to an existing qwen3-format parquet."
    exit 1
fi

if [ ! -f "$TOOL_CONFIG" ]; then
    echo "ERROR: Tool config not found at $TOOL_CONFIG"
    exit 1
fi

if ! curl -s --max-time 5 "$SEARCH_SERVICE_URL" > /dev/null 2>&1; then
    echo "WARNING: Search service at $SEARCH_SERVICE_URL may not be running."
    echo "Start it with: CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_search_service.sh dense 8090"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "=========================================="
echo "OpenResearcher GRPO Training (Qwen3-8B)"
echo "=========================================="
echo "Project dir:    $PROJECT_DIR"
echo "verl dir:       $VERL_DIR"
echo "Train data:     $TRAIN_DATA"
echo "Val data:       $VAL_DATA"
echo "GPUs:           $N_GPUS (TP=$TP_SIZE)"
echo "Search service: $SEARCH_SERVICE_URL"
echo "Experiment:     $EXPERIMENT_NAME"
echo "=========================================="

# ---- Apply monkey patches ----
# Patch: verl tool schemas (extra="allow" + array types)
VERL_SCHEMAS=$(python3 -c "import verl.tools.schemas; print(verl.tools.schemas.__file__)" 2>/dev/null)
if [ -n "$VERL_SCHEMAS" ] && ! grep -q 'extra="allow"' "$VERL_SCHEMAS"; then
    echo "Applying verl tool schemas patch to: $VERL_SCHEMAS"
    cp "$PROJECT_DIR/verl_rl/patches/verl_tool_schemas.py" "$VERL_SCHEMAS"
fi

# ---- Launch training ----
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export RAY_memory_monitor_refresh_ms=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export REWARD_DEBUG_LOG="${REWARD_DEBUG_LOG:-/tmp/reward_debug_qwen3.jsonl}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd "$VERL_DIR"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='openresearcher_multiturn_grpo_qwen3' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=28672 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=500 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=500 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG" \
    custom_reward_function.path="$REWARD_MODULE" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.project_name='openresearcher_rl' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.total_epochs=2 \
    trainer.val_before_train=False \
    trainer.test_freq=50 \
    trainer.logger='["console","wandb"]' \
    "$@"
