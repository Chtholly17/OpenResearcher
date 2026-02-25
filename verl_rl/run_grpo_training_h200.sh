#!/bin/bash
# Launch multi-turn GRPO training for OpenResearcher — H200 (141GB) optimized
#
# Key differences from A100 version:
#   - TP=2 → 4 vLLM server groups (2x rollout parallelism)
#   - 64K context (vs 16K on A100) — fits 30B lm_head logits in 141GB
#   - No FSDP param/optimizer offload — everything fits on-GPU
#   - Full 3-policy GRPO with KL loss (ref policy fits in memory)
#   - Larger micro-batches for faster forward/backward
#   - train_batch_size=8 for better gradient signal
#
# Usage:
#   bash verl_rl/run_grpo_training_h200.sh
#
#   # Override defaults:
#   TRAIN_DATA=data/train_curriculum_1k.parquet \
#   EXPERIMENT_NAME=grpo_h200_1k \
#   bash verl_rl/run_grpo_training_h200.sh

set -x
ulimit -n 65535

unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
unset NVIDIA_TF32_OVERRIDE
unset NCCL_TF32_OVERRIDE
export TORCH_FP32_PRECISION=tf32

# ---- Configuration ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# VERL_DIR="${VERL_DIR:-$(cd "$PROJECT_DIR/../verl" && pwd)}"
echo "VERL_DIR: $VERL_DIR"
CONFIG_PATH="$PROJECT_DIR/verl_rl/config"

N_GPUS="${N_GPUS:-4}"
NNODES="${NNODES:-1}"

# TP=2: 30B-A3B fits in 2×141GB → 4 server groups (vs 2 on A100 with TP=4)
TP_SIZE="${TP_SIZE:-2}"

# Data paths
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/train_curriculum_500.parquet}"
VAL_DATA="${VAL_DATA:-$PROJECT_DIR/data/test_20.parquet}"

# Search service URL (must be running)
SEARCH_SERVICE_URL="${SEARCH_SERVICE_URL:-http://127.0.0.1:8090}"

# Tool, interaction, and reward config paths (absolute)
TOOL_CONFIG="$CONFIG_PATH/tool_config/openresearcher_tool_config.yaml"
INTERACTION_CONFIG="$CONFIG_PATH/interaction_config/openresearcher_interaction_config.yaml"
REWARD_MODULE="$PROJECT_DIR/verl_rl/reward/openresearcher_reward.py"

# Experiment naming
TIMESTAMP=$(date '+%m%d-%H%M')
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_h200_${TIMESTAMP}}"

# ---- Download data if not present ----
HF_DATA_REPO="PahaII/openresearcher-training-data"
DATA_DIR="$PROJECT_DIR/data"
DATA_FILES=(
    "train_curriculum_500.parquet"
    "train_curriculum_1k.parquet"
    "train_curriculum_2k.parquet"
    "train_no_passrate1.parquet"
    "test_20.parquet"
)

missing_files=()
for f in "${DATA_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        missing_files+=("$f")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "Downloading missing data files from $HF_DATA_REPO..."
    mkdir -p "$DATA_DIR"
    for f in "${missing_files[@]}"; do
        echo "  Downloading $f..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$HF_DATA_REPO', filename='$f', repo_type='dataset', local_dir='$DATA_DIR')
" || { echo "ERROR: Failed to download $f"; exit 1; }
    done
    echo "Download complete."
fi

# ---- Validate prerequisites ----
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Available files in $DATA_DIR:"
    ls -1 "$DATA_DIR"/*.parquet 2>/dev/null || echo "  (none)"
    echo "Set TRAIN_DATA to one of the above, e.g.:"
    echo "  TRAIN_DATA=$DATA_DIR/train_curriculum_500.parquet bash verl_rl/run_grpo_training_h200.sh"
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
echo "OpenResearcher GRPO Training (H200)"
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

# ---- Apply monkey patches ----
# Patch 1: NemotronH flash attention (only if model is in HF cache)
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
NEMOTRON_GLOB="$HF_CACHE_DIR/modules/transformers_modules/OpenResearcher/OpenResearcher_hyphen_30B_hyphen_A3B/*/modeling_nemotron_h.py"
for f in $NEMOTRON_GLOB; do
    if [ -f "$f" ]; then
        if ! grep -q '_supports_flash_attn_2' "$f"; then
            echo "Applying NemotronH flash attention patch to: $f"
            cp "$PROJECT_DIR/verl_rl/patches/modeling_nemotron_h.py" "$f"
        fi
    fi
done

# Patch 2: mamba-ssm graceful import
MAMBA_INIT=$(python3 -c "import mamba_ssm; print(mamba_ssm.__file__)" 2>/dev/null)
if [ -n "$MAMBA_INIT" ] && ! grep -q 'except ImportError' "$MAMBA_INIT"; then
    echo "Applying mamba-ssm graceful import patch to: $MAMBA_INIT"
    cp "$PROJECT_DIR/verl_rl/patches/mamba_ssm__init__.py" "$MAMBA_INIT"
fi

# Patch 3: verl tool schemas (extra="allow" + array types)
VERL_SCHEMAS=$(python3 -c "import verl.tools.schemas; print(verl.tools.schemas.__file__)" 2>/dev/null)
if [ -n "$VERL_SCHEMAS" ] && ! grep -q 'extra="allow"' "$VERL_SCHEMAS"; then
    echo "Applying verl tool schemas patch to: $VERL_SCHEMAS"
    cp "$PROJECT_DIR/verl_rl/patches/verl_tool_schemas.py" "$VERL_SCHEMAS"
fi


export PYTHONPATH="$PROJECT_DIR/verl_rl/torch_hooks:$PROJECT_DIR:${PYTHONPATH:-}"
export TORCH_HOOKS_VERBOSE=1

# ---- Launch training ----
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export RAY_memory_monitor_refresh_ms=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export REWARD_DEBUG_LOG="${REWARD_DEBUG_LOG:-/tmp/reward_debug.jsonl}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# cd "$VERL_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

"$PYTHON" -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='openresearcher_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=98304 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=OpenResearcher/OpenResearcher-30B-A3B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=102400 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
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
