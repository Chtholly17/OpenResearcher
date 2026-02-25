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

# ---- TF32 API consistency (avoid "TF32 API mixing" errors with vLLM/FSDP) ----
# If TF32 errors persist, try: rm -rf ~/.cache/vllm/torch_compile_cache
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
unset NVIDIA_TF32_OVERRIDE
unset NCCL_TF32_OVERRIDE
export TORCH_FP32_PRECISION=tf32

# ---- Configuration ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${VERL_DIR:-$(cd "$PROJECT_DIR/../verl" && pwd)}"
CONFIG_PATH="$PROJECT_DIR/verl_rl/config"

N_GPUS="${N_GPUS:-4}"
NNODES="${NNODES:-1}"

# Model: 30B-A3B MoE with TP=4 for rollout (2 server groups)
TP_SIZE="${TP_SIZE:-4}"

# Data paths
TRAIN_DATA="/fsx-shared/juncheng/data/openresearcher/train.parquet"
VAL_DATA="/fsx-shared/juncheng/data/openresearcher/test.parquet"

# Search service URL (must be running)
SEARCH_SERVICE_URL="${SEARCH_SERVICE_URL:-http://127.0.0.1:8090}"

# Tool, interaction, and reward config paths (absolute)
TOOL_CONFIG="$CONFIG_PATH/tool_config/openresearcher_tool_config.yaml"
INTERACTION_CONFIG="$CONFIG_PATH/interaction_config/openresearcher_interaction_config.yaml"
REWARD_MODULE="$PROJECT_DIR/verl_rl/reward/openresearcher_reward.py"

# Experiment naming
TIMESTAMP=$(date '+%m%d-%H%M')
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_multiturn_${TIMESTAMP}}"

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
    echo "  TRAIN_DATA=$DATA_DIR/train_curriculum_500.parquet bash verl_rl/run_grpo_training.sh"
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

# ---- Launch training ----
# Add torch_hooks first (TF32 consistency via sitecustomize.py), then project dir for verl_rl
export PYTHONPATH="$PROJECT_DIR/verl_rl/torch_hooks:$PROJECT_DIR:${PYTHONPATH:-}"
export TORCH_HOOKS_VERBOSE=1

# Disable Ray's OOM killer (false positive with 1TB+ RAM machines)
export RAY_memory_monitor_refresh_ms=0

# Help PyTorch manage fragmented GPU memory (avoids OOM on the 30B MoE model)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Enable reward function debug logging (first 20 samples)
export REWARD_DEBUG_LOG="${REWARD_DEBUG_LOG:-/tmp/reward_debug.jsonl}"

# HF cache — use writable home directory (model already cached here)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Working directory must be the verl root for hydra config resolution
cd "$VERL_DIR"

# Use venv python if available (ensures correct verl from site-packages)
PYTHON="${PROJECT_DIR}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

CUDA_VISIBLE_DEVICES=0,1,2,3  "$PYTHON" -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='openresearcher_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=126976 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=OpenResearcher/OpenResearcher-30B-A3B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_model_len=131072 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=500 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=500 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG" \
    custom_reward_function.path="$REWARD_MODULE" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.rollout_correction.bypass_mode=True \
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
