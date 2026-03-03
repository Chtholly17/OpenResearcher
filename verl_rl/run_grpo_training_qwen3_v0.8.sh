#!/bin/bash
# GRPO training script for Qwen3-8B — v0.8 config (A100 80GB)
#
# Key design decisions:
#
#   Data: qwen3_highpass (Nemotron pass_rate ≥ 0.875, 1874 examples)
#     - 43% per-rollout correct rate → near-zero dead-gradient steps
#     - System prompt mandates browser.search (no memory-recall shortcuts)
#     - Reward v0.5: correct+searched → length bonus [0.8,1.2],
#                   correct+no_search → 0.3, wrong+searched → 0.1
#
#   lr=1e-5       (1e-6 froze policy; 1e-5 learns but must resume after ~75 steps)
#   clip_grad=0.5 (prevents entropy spike that collapsed runs at step ~65-90)
#   bs=16, n=8    (10.4 useful questions/step; σ≈0.07 vs σ≈0.15 for bs=4)
#   TP=2          (A100-80GB: 2 GPUs per vLLM group → 2 server groups on 4 GPUs)
#   resume_mode=auto  (resumes from latest checkpoint automatically)
#
# Hardware layout (8× A100-80GB):
#   GPUs 0-1  — dense search service (Qwen3-Embedding-8B, ~15 GB each)
#   GPUs 4-7  — training (FSDP + vLLM colocated)
#
# Prerequisites:
#   1. Dense search service on GPUs 0-1, port 8090:
#        bash verl_rl/prepare_highpass_data.sh  # once, to create data/qwen3_highpass/
#        CUDA_VISIBLE_DEVICES=0,1 GPU_IDS=0,1 \
#          SEARCHER_TYPE=dense \
#          DENSE_INDEX_PATH="$PROJECT_DIR/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl" \
#          DENSE_MODEL_NAME="Qwen/Qwen3-Embedding-8B" \
#          LUCENE_EXTRA_DIR="$PROJECT_DIR/tevatron" \
#          CORPUS_PARQUET_PATH="$PROJECT_DIR/Tevatron/browsecomp-plus-corpus/data/*.parquet" \
#          python -m uvicorn scripts.deploy_search_service:app --host 0.0.0.0 --port 8090
#   2. Training data at data/qwen3_highpass/:
#        bash verl_rl/prepare_highpass_data.sh
#   3. conda activate openresearcher (or set PYTHON env var)
#
# Usage:
#   bash verl_rl/run_grpo_training_qwen3_v0.8.sh
#
#   # Resume from a specific checkpoint
#   RESUME_FROM=checkpoints/openresearcher_rl/grpo_qwen3_v0.8_highpass/global_step_75 \
#   bash verl_rl/run_grpo_training_qwen3_v0.8.sh
#
#   # Override GPUs / experiment name
#   CUDA_VISIBLE_DEVICES=2,3,4,5 N_GPUS=4 \
#   EXPERIMENT_NAME=grpo_v0.8_run2 \
#   bash verl_rl/run_grpo_training_qwen3_v0.8.sh

set -euo pipefail
ulimit -n 65535

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="${PYTHON:-/opt/dlami/nvme/hqhd-miniconda3/envs/openresearcher/bin/python}"

# A100 layout: GPUs 4-7 for training, GPUs 0-1 for search service
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
N_GPUS="${N_GPUS:-4}"
TP_SIZE="${TP_SIZE:-2}"   # 2 GPUs per vLLM group → 2 server groups on 4 GPUs

TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/qwen3_highpass/train.parquet}"
VAL_DATA="${VAL_DATA:-$PROJECT_DIR/data/qwen3_highpass/test.parquet}"
SEARCH_SERVICE_URL="${SEARCH_SERVICE_URL:-http://127.0.0.1:8090}"

TIMESTAMP=$(date '+%m%d-%H%M')
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3_v0.8_highpass}"

TOOL_CONFIG="$PROJECT_DIR/verl_rl/config/tool_config/openresearcher_tool_config.yaml"
INTERACTION_CONFIG="$PROJECT_DIR/verl_rl/config/interaction_config/openresearcher_interaction_config.yaml"
REWARD_MODULE="$PROJECT_DIR/verl_rl/reward/openresearcher_reward.py"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Run: bash verl_rl/prepare_highpass_data.sh"
    exit 1
fi
if ! curl -s --max-time 5 "$SEARCH_SERVICE_URL" > /dev/null 2>&1; then
    echo "WARNING: Search service at $SEARCH_SERVICE_URL may not be running."
    read -p "Continue anyway? [y/N] " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

echo "=========================================="
echo "OpenResearcher GRPO Training — v0.8 (A100)"
echo "=========================================="
echo "Project dir:    $PROJECT_DIR"
echo "GPUs:           $CUDA_VISIBLE_DEVICES  (N=$N_GPUS, TP=$TP_SIZE)"
echo "Train data:     $TRAIN_DATA"
echo "Val data:       $VAL_DATA"
echo "Search service: $SEARCH_SERVICE_URL"
echo "Experiment:     $EXPERIMENT_NAME"
echo "lr=1e-5  clip_grad=0.5  bs=16  TP=2  max_response=12288"
echo "=========================================="

# ── Apply verl tool schemas patch (idempotent) ────────────────────────────────
VERL_SCHEMAS=$("$PYTHON" -c "import verl.tools.schemas; print(verl.tools.schemas.__file__)" 2>/dev/null || true)
if [ -n "$VERL_SCHEMAS" ] && ! grep -q 'extra="allow"' "$VERL_SCHEMAS" 2>/dev/null; then
    echo "Applying verl tool schemas patch..."
    cp "$PROJECT_DIR/verl_rl/patches/verl_tool_schemas.py" "$VERL_SCHEMAS"
fi

# ── Environment ───────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export RAY_memory_monitor_refresh_ms=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export REWARD_DEBUG_LOG="${REWARD_DEBUG_LOG:-/tmp/reward_debug_v0.8.jsonl}"

# ── Launch ────────────────────────────────────────────────────────────────────
"$PYTHON" -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/verl_rl/config" \
    --config-name='openresearcher_multiturn_grpo_qwen3' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=12288 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.clip_grad=0.5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=16384 \
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
    trainer.nnodes=1 \
    trainer.total_epochs=2 \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.logger='["console","wandb"]' \
    "$@"
