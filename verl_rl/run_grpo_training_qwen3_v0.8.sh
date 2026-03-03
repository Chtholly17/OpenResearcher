#!/bin/bash
# GRPO training script for Qwen3-8B — v0.8 config
#
# Key design decisions vs earlier versions:
#
#   Data: qwen3_highpass (Nemotron pass_rate 0.875–1.0, 1874 examples)
#     - 43% per-rollout correct rate vs 12% on mid-difficulty data
#     - System prompt mandates browser.search (fixes memory-recall problem)
#     - Reward penalises 0-search correct answers (0.3 vs 1.0)
#
#   lr=1e-5  (not 1e-6 which froze the policy, not higher which diverges)
#   clip_grad=0.5  (prevents the gradient spike that collapsed v0.6 at step 65)
#   train_batch_size=16, ppo_mini_batch_size=16
#     - With ~43% correct rate and n=8: P(0 correct in group) = 0.57^8 = 1%
#     - 16 questions → ~10.4 useful questions per step (vs 2.6 with bs=4)
#     - Reduces step-level reward variance from σ≈0.15 (bs=4) to σ≈0.07 (bs=16)
#
# Prerequisites:
#   1. Dense search service running on GPUs 0-1, port 8090:
#        CUDA_VISIBLE_DEVICES=0,1 GPU_IDS=0,1 \
#          SEARCHER_TYPE=dense \
#          DENSE_INDEX_PATH="$PROJECT/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl" \
#          DENSE_MODEL_NAME="Qwen/Qwen3-Embedding-8B" \
#          LUCENE_EXTRA_DIR="$PROJECT/tevatron" \
#          CORPUS_PARQUET_PATH="$PROJECT/Tevatron/browsecomp-plus-corpus/data/*.parquet" \
#          python -m uvicorn scripts.deploy_search_service:app --host 0.0.0.0 --port 8090
#   2. Training data at data/qwen3_highpass/ (regenerate if system prompt changed):
#        python -c "
#          from datasets import load_dataset, Dataset
#          import sys; sys.path.insert(0, '.')
#          sys.path.insert(0, 'verl_rl')
#          from preprocess_openresearcher import make_verl_record
#          import random, os
#          random.seed(42)
#          ds = load_dataset('Chtholly17/OR_reject_sampling', split='train')
#          hp = [r for r in ds if r['pass_rate'] >= 0.875]
#          random.shuffle(hp)
#          Dataset.from_list([make_verl_record(str(r['qid']),r['question'],r['correct_answer'],'train',i,'qwen3') for i,r in enumerate(hp[20:])]).to_parquet('data/qwen3_highpass/train.parquet')
#          Dataset.from_list([make_verl_record(str(r['qid']),r['question'],r['correct_answer'],'test',i,'qwen3') for i,r in enumerate(hp[:20])]).to_parquet('data/qwen3_highpass/test.parquet')
#        "
#   3. conda activate openresearcher (or use full python path)
#
# Usage:
#   # Default: GPUs 4-7 for training, search service on 0-1
#   bash verl_rl/run_grpo_training_qwen3_v0.8.sh
#
#   # Override GPU assignment
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash verl_rl/run_grpo_training_qwen3_v0.8.sh
#
#   # Override experiment name and data
#   EXPERIMENT_NAME=grpo_v0.8_run2 \
#   TRAIN_DATA=data/qwen3_highpass/train.parquet \
#   bash verl_rl/run_grpo_training_qwen3_v0.8.sh

set -euo pipefail
ulimit -n 65535

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/dlami/nvme/hqhd-miniconda3/envs/openresearcher/bin/python}"

# GPUs: training uses 4 GPUs with TP=2 (2 vLLM server groups)
# Search service should be on the remaining GPUs (e.g. 0-1)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
N_GPUS="${N_GPUS:-4}"
TP_SIZE="${TP_SIZE:-2}"

# Data
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/qwen3_highpass/train.parquet}"
VAL_DATA="${VAL_DATA:-$PROJECT_DIR/data/qwen3_highpass/test.parquet}"
SEARCH_SERVICE_URL="${SEARCH_SERVICE_URL:-http://127.0.0.1:8090}"

TIMESTAMP=$(date '+%m%d-%H%M')
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3_v0.8_${TIMESTAMP}}"

TOOL_CONFIG="$PROJECT_DIR/verl_rl/config/tool_config/openresearcher_tool_config.yaml"
INTERACTION_CONFIG="$PROJECT_DIR/verl_rl/config/interaction_config/openresearcher_interaction_config.yaml"
REWARD_MODULE="$PROJECT_DIR/verl_rl/reward/openresearcher_reward.py"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "See script header for generation instructions."
    exit 1
fi
if ! curl -s --max-time 5 "$SEARCH_SERVICE_URL" > /dev/null 2>&1; then
    echo "WARNING: Search service at $SEARCH_SERVICE_URL may not be running."
    read -p "Continue anyway? [y/N] " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

echo "=========================================="
echo "OpenResearcher GRPO Training — v0.8"
echo "=========================================="
echo "Project dir:    $PROJECT_DIR"
echo "GPUs:           $CUDA_VISIBLE_DEVICES  (N=$N_GPUS, TP=$TP_SIZE)"
echo "Train data:     $TRAIN_DATA"
echo "Val data:       $VAL_DATA"
echo "Search service: $SEARCH_SERVICE_URL"
echo "Experiment:     $EXPERIMENT_NAME"
echo "lr=1e-5  clip_grad=0.5  bs=16  max_response=12288"
echo "=========================================="

# ── Apply verl tool schemas patch (idempotent) ────────────────────────────────
VERL_SCHEMAS=$("$PYTHON" -c "import verl.tools.schemas; print(verl.tools.schemas.__file__)" 2>/dev/null)
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
    trainer.val_before_train=False \
    trainer.test_freq=25 \
    trainer.logger='["console","wandb"]' \
    "$@"
