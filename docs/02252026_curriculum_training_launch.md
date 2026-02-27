# 02/25/2026 - Curriculum Training Launch: Pass-Rate-Based Data Selection & Full Run Config

Continuation of 02/24/2026 session. Goal: launch a proper full GRPO training run with n=8, wandb logging, and curriculum-based data selection.

## Rejection Sampling Analysis

Analyzed pass rate distribution from `Chtholly17/OR_reject_sampling` dataset (`evaluated_bedrock_batch.jsonl`). This contains 6,102 questions evaluated with 8 binary judges each, producing discrete pass rates (0/8 to 8/8).

### Pass Rate Distribution

| Pass Rate | Count |
|-----------|-------|
| 0.000 | 2,878 (47.2%) |
| 0.125 | 331 |
| 0.250 | 229 |
| 0.375 | 203 |
| 0.500 | 145 |
| 0.625 | 175 |
| 0.750 | 247 |
| 0.875 | 313 |
| 1.000 | 1,581 (25.9%) |

Key observation: 73% of data is either pass_rate=0 (too hard) or pass_rate=1 (too easy). Only 1,643 examples (26.9%) are in the mid-range (0 < pass_rate < 1). Of these, 1,557 overlap with the training set.

## Curriculum Data Selection Strategy

Sorted training examples by `|pass_rate - 0.5|` (distance to mid-difficulty), then selected outward from center. This prioritizes questions where the model sometimes succeeds and sometimes fails — the most informative examples for RL.

Created 4 curriculum datasets in `data/`:

| Dataset | Size | Pass Rate Composition | Purpose |
|---------|------|-----------------------|---------|
| `train_curriculum_500.parquet` | 500 | 0.375–0.625 (core mid-range) | Stage 1 training |
| `train_curriculum_1k.parquet` | 1,000 | 0.125–0.875 (full mid-range) | Stage 2 |
| `train_curriculum_2k.parquet` | 2,000 | 0.000–0.875 (mid + some hard) | Stage 3 |
| `train_no_passrate1.parquet` | 4,289 | All except pass_rate==1.0 | Final stage |

The 2k set includes 443 pass_rate=0 examples because only 1,557 mid-range examples exist in the training set.

## Config Changes for Full Run

### `run_grpo_training.sh`

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `rollout.n` | 2 | **8** | Standard GRPO group size for advantage estimation |
| `trainer.logger` | `["console"]` | **`["console","wandb"]`** | Enable experiment tracking |
| `data.train_files` / `data.val_files` | `$HOME/data/openresearcher/` | **`$PROJECT_DIR/data/`** | Moved data into project dir |

`ppo_mini_batch_size` kept at 4 (must be <= `train_batch_size`; verl iterates over prompts, not total rollouts).

### `_session_state.py` — Fixed Placeholder Echoing

Removed `<your answer>` and `<your best answer>` template text from budget warning messages. The model was literally echoing these placeholders instead of producing real answers.

Before:
```
"Exact Answer: <your answer>\n"
"Exact Answer: <your best answer>\n"
```

After: Generic instructions without placeholder examples:
```
"Use the submit_answer tool or write 'Exact Answer:' followed by your answer."
"Based on everything you have gathered, write your final answer immediately."
```

## Data Hosting

Curriculum datasets and test splits uploaded to HuggingFace: `PahaII/openresearcher-training-data`

Files:
- `train_curriculum_500.parquet`
- `train_curriculum_1k.parquet`
- `train_curriculum_2k.parquet`
- `train_no_passrate1.parquet`
- `test_20.parquet`

The launch script (`run_grpo_training.sh`) auto-downloads missing files from this repo on startup.

## Launch Issues & Fixes

1. **`ppo_mini_batch_size=32` validation error**: Initially set mini_batch to 32 (train_batch_size × n) thinking it iterates over rollouts. verl validates `train_batch_size >= ppo_mini_batch_size` — it iterates over prompts. Reverted to 4.

2. **HF_HOME empty string**: `$HOME` resolved to empty in nohup context, giving `HF_HOME=/.cache/huggingface`. Fixed by passing explicit path `/home/hqhardy/.cache/huggingface`.

3. **wandb not logged in**: Initial launch failed with `wandb.errors.UsageError: No API key configured`. Fixed by running `wandb login` first.

## Additional OOM Issues & Fixes

4. **OOM at `_compute_old_log_prob`**: After validation completed, the first training step OOM'd recomputing log probs on 128K sequences with the FSDP actor. Reducing `log_prob_micro_batch_size_per_gpu` from 2 to 1 didn't help — a single 128K sequence is too large for the FSDP forward pass on A100-80GB.

   **Fix**: Enabled `algorithm.rollout_correction.bypass_mode=True` which skips recomputing old log probs entirely and uses the log probs from vLLM rollout directly (2-policy GRPO: pi_rollout, pi_theta instead of 3-policy: pi_rollout, pi_old, pi_theta).

5. **vLLM WorkerProc init failure**: Stale GPU memory from crashed runs caused vLLM engines to fail on restart. Must do thorough cleanup: `ray stop --force` + kill all GPU processes + wait for memory to fully free before relaunching.

## Current Training Run

**Run**: `grpo_curriculum_500`
**Wandb**: https://wandb.ai/tuisaac/openresearcher_rl/runs/ergda3t3
**Config**:
- Training data: `train_curriculum_500.parquet` (500 mid-pass-rate examples)
- Validation data: `test_20.parquet` (20 examples, n=1)
- n=8 (GRPO group size), val_kwargs.n=1
- train_batch_size=4 → 32 rollouts per step
- Total steps: 250 (500/4 × 2 epochs)
- 128K context, max_num_seqs=256, TP=4
- Budget: 200 soft warning / 300 hard cutoff
- val_before_train=False, test_freq=50
- bypass_mode=True (skip old log prob recomputation)
- Forced answer prefix still active in tool_agent_loop.py

## Files Modified

- `verl_rl/run_grpo_training.sh` — n=8, wandb logging, project-local data paths
- `verl_rl/tools/_session_state.py` — Removed placeholder text from budget messages
- `.gitignore` — Already had `data/` entry (no change needed)

## Files Created

- `data/train_curriculum_500.parquet` — 500 examples, pass_rate 0.375–0.625
- `data/train_curriculum_1k.parquet` — 1,000 examples, pass_rate 0.125–0.875
- `data/train_curriculum_2k.parquet` — 2,000 examples, pass_rate 0.000–0.875
- `data/train_no_passrate1.parquet` — 4,289 examples, all except pass_rate=1.0

## H200 Migration: Hyperparameter Tuning Guide

H200 has 141 GB HBM3e per GPU (vs A100's 80 GB HBM2e) — ~1.76x more memory and ~1.43x higher bandwidth (4.8 TB/s vs 3.35 TB/s). This unlocks several parameters that were constrained on A100.

### High Priority — Biggest Impact

| Parameter | A100 Value | H200 Recommended | Rationale |
|-----------|-----------|-----------------|-----------|
| `rollout.tensor_model_parallel_size` | 4 | **2** | 30B-A3B fits in 2×141GB. TP=2 gives **4 vLLM server groups** instead of 2, doubling rollout parallelism — the single biggest speedup. |
| `rollout.max_model_len` | 131072 (128K) | **196608–262144** (192K–256K) | Longer context = more room for tool responses + model tokens in `max_response_length`. 192K+ was OOM/too-slow on A100. |
| `data.max_response_length` | 126976 | **match max_model_len - 4096** | Token budget for model + tool responses. Increase proportionally with max_model_len. |
| `algorithm.rollout_correction.bypass_mode` | True (forced by OOM) | **Remove / False** | With more memory, `_compute_old_log_prob` may succeed. Decoupled 3-policy mode (pi_rollout, pi_old, pi_theta) gives more stable training. Test first. |

### Medium Priority — Better Throughput

| Parameter | A100 Value | H200 Recommended | Rationale |
|-----------|-----------|-----------------|-----------|
| `actor.fsdp_config.param_offload` | True | **False** | 141GB may fit the full 30B model without CPU offload. Removes offload latency. Try without first; re-enable if OOM. |
| `actor.fsdp_config.optimizer_offload` | True | **False** | Same — optimizer states fit in HBM3e. Significant speedup for actor update steps. |
| `rollout.log_prob_micro_batch_size_per_gpu` | 1 | **2–4** | Was reduced from 2→1 due to OOM. H200 can handle larger micro-batches for log prob computation. |
| `actor.ppo_micro_batch_size_per_gpu` | 1 | **2** | Larger micro-batch for actor gradient computation. |
| `data.train_batch_size` | 4 | **8** | More prompts per step = more diverse training signal. Requires `ppo_mini_batch_size` adjustment (keep equal to train_batch_size). |
| `rollout.max_num_seqs` | 256 | **256–512** | More concurrent sequences in vLLM. Monitor GPU util — increase if GPUs are underutilized during rollouts. |

### Lower Priority — Fine Tuning

| Parameter | A100 Value | H200 Recommended | Rationale |
|-----------|-----------|-----------------|-----------|
| `rollout.gpu_memory_utilization` | 0.9 | **0.9** | Keep as-is initially. Only increase if vLLM needs more KV cache. |
| `rollout.n` | 8 | **8–16** | More rollouts per prompt improves GRPO advantage estimation. n=16 doubles compute but improves gradient quality. Try after other params are stable. |
| `ref.log_prob_micro_batch_size_per_gpu` | 1 | **2–4** | Same reasoning as rollout log_prob micro-batch. |

### Migration Checklist

1. **Start with TP=2** — this is the single biggest win (4 server groups → 2x parallel rollouts)
2. **Try disabling FSDP offloading** (both param and optimizer) — if it fits, training steps will be much faster
3. **Increase max_model_len to 192K** and max_response_length accordingly
4. **Remove bypass_mode** and test if old_log_prob computation fits in memory
5. **Increase micro-batch sizes** (log_prob and actor) incrementally
6. **If all above work**, consider increasing train_batch_size to 8 and n to 16
7. **Monitor**: Use wandb to compare step time, reward curves, and GPU utilization against A100 baseline

### Key Constraint Reminder

`max_response_length` is the total token budget for BOTH model-generated tokens AND tool response tokens in multi-turn rollouts. The SFT model needs 30K–120K+ tokens for a full trajectory. Increasing this is the most direct way to improve answer rates, and H200's larger memory makes this feasible.

## Continued: A100 OOM Cascade & Context Reduction

The `grpo_curriculum_500` run (128K context) crashed with further OOM errors during the actor backward pass. The root cause is the lm_head logits tensor: `seq_len × vocab_size(131072) × 4 bytes (fp32)`.

### OOM Fix Chain

6. **Ref policy OOM at `_compute_ref_log_prob`**: `torch.OutOfMemoryError: Tried to allocate 48.19 GiB` at lm_head forward pass.

   **Fix**: Set `actor_rollout_ref.actor.use_kl_loss=False` (removed kl_loss_coef and kl_loss_type). With both `use_kl_in_reward=False` and `use_kl_loss=False`, `need_reference_policy()` returns False, so the ref model is never loaded.

7. **Actor backward OOM at 128K, 64K, and 32K contexts**: The lm_head produces a logits tensor of `seq_len × 131072 × 4B`:
   - 128K → 64 GiB
   - 64K → 32 GiB
   - 32K → 16 GiB

   With FSDP base ~60 GiB + vLLM ~4 GiB on A100-80GB, only ~16 GiB remains. All three context sizes OOM'd.

   **Fix**: Reduced to 16K context (`max_model_len=16384`, `max_response_length=12288`). The 8 GiB logits tensor fits.

### Successful A100 Training at 16K

**Run**: `grpo_curriculum_500_16k`
- 16K context, 8× A100-80GB, TP=4
- Step time: ~165s after warmup
- Score mean 0.008–0.080 (most rollouts hitting 12K token cap)
- clip_ratio rising to 59%

### Final A100 `run_grpo_training.sh` Parameters

```
data.max_response_length=12288
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.rollout.tensor_model_parallel_size=4
actor_rollout_ref.rollout.max_model_len=16384
actor_rollout_ref.rollout.calculate_log_probs=True
algorithm.rollout_correction.bypass_mode=True
```

## Dense Retrieval Migration (BM25 → Qwen3-Embedding-8B)

Switched from BM25 search to dense retrieval to improve search quality for RL rollouts.

### Dense Retrieval Service Setup

- **Model**: `Qwen/Qwen3-Embedding-8B` (8B param embedding model, ~15 GB per GPU in fp16)
- **Index**: FAISS FlatIP with 100,195 documents, sharded into 4 pickle files (~1.6 GB total) at `Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/`
- **Corpus**: 100,195 documents in parquet at `Tevatron/browsecomp-plus-corpus/data/`
- **Server**: FastAPI/uvicorn on port 8090, endpoints: `POST /search`, `POST /get_content`
- **Multi-GPU**: `GPU_IDS` env var supports comma-separated IDs; loads a model instance per GPU

### Launch Command (Dense Retrieval)

```bash
CUDA_VISIBLE_DEVICES=0,1 GPU_IDS=0,1 HF_HOME=$HOME/.cache/huggingface \
  SEARCHER_TYPE=dense \
  DENSE_INDEX_PATH="$PROJECT/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl" \
  DENSE_MODEL_NAME="Qwen/Qwen3-Embedding-8B" \
  LUCENE_EXTRA_DIR="$PROJECT/tevatron" \
  CORPUS_PARQUET_PATH="$PROJECT/Tevatron/browsecomp-plus-corpus/data/*.parquet" \
  python -m uvicorn scripts.deploy_search_service:app --host 0.0.0.0 --port 8090
```

### Issues Encountered

- **Broken `.venv` symlink**: `.venv` pointed to `/opt/dlami/nvme/miniconda3/envs/openresearcher` (wrong path). Fixed symlink to `/opt/dlami/nvme/hqhardy-miniconda3/envs/openresearcher`. The conda env has no `bin/activate` script, so `start_search_service.sh` (which calls `source .venv/bin/activate`) can't be used directly — must launch via the conda python binary.
- **HF_HOME permission error**: Default `$HOME` resolved to `/home/efs/hardychen` in some contexts, causing `PermissionError` when downloading Qwen3-Embedding-8B. Must pass `HF_HOME` explicitly.
- **Port conflict**: Failed previous launch attempts left port 8090 bound. Fixed with `lsof -ti :8090 | xargs kill -9`.

## 4-GPU Training Attempt on A100

Attempted training on 4 A100-80GB GPUs (GPUs 4-7) with TP=2 while dense retrieval ran on GPUs 0-1.

**Result**: OOM at `actor_rollout_update_actor` — `Tried to allocate 6.41 GiB` with only 6.29 GiB free. Per-GPU breakdown:
- FSDP actor process: 69.78 GiB (vs ~60 GiB with 8 GPUs)
- vLLM colocated worker: 3.21 GiB
- Total: ~73 GiB, leaving only ~6 GiB — not enough for lm_head logits (6.4+ GiB at 16K)

**Root cause**: 4-way FSDP means each GPU holds 2x the gradient shards (30B × 4B / 4 = 30 GB) vs 8-way (15 GB). This 15 GB increase eats the entire headroom that was available on 8 GPUs.

**Conclusion**: 30B-A3B cannot train on 4× A100-80GB. Minimum is 6 GPUs on A100, or 4× H200-141GB with optimizer offload.

## H200 Script: 4-GPU Training Layout (`run_grpo_training_h200.sh`)

Updated for 2 retrieval + 4 training GPU layout on H200.

### Memory Budget (per H200, 4-way FSDP, optimizer offload)

| Component | Size |
|-----------|------|
| Model params (bf16) | 15 GB (30B × 2B / 4) |
| Gradients (fp32) | 30 GB (30B × 4B / 4) |
| Optimizer (Adam) | offloaded to CPU |
| lm_head logits (64K) | 32 GB (64K × 131072 × 4B) |
| **Remaining for activations** | **~64 GB** |

### Key Parameter Differences from 8-GPU H200 Plan

| Parameter | 8-GPU Plan | 4-GPU Final | Rationale |
|-----------|-----------|-------------|-----------|
| `N_GPUS` | 8 | **4** | 2 GPUs for dense retrieval |
| `CUDA_VISIBLE_DEVICES` | (all) | **2,3,4,5** | Skip retrieval GPUs 0-1 |
| `train_batch_size` | 8 | **4** | Halved with fewer GPUs |
| `ppo_mini_batch_size` | 8 | **4** | Must equal train_batch_size |
| `ppo_micro_batch_size_per_gpu` | 2 | **1** | Conservative for 4-way FSDP |
| `optimizer_offload` | False | **True** | 30B/4 = 60 GB optimizer states won't fit |
| `max_model_len` | 102400 | **65536** | 100K logits (51 GB) too large; 64K (32 GB) fits |
| `max_response_length` | 98304 | **61440** | max_model_len - 4096 |
| `log_prob_micro_batch_size` | 4 | **2** | Memory safety |
| `ref.param_offload` | False | **True** | Save memory during ref forward |

KL loss remains enabled (`use_kl_loss=True`, `kl_loss_coef=0.001`) — 3-policy GRPO with ref model. bypass_mode removed.

## Files Modified (Continued)

- `verl_rl/run_grpo_training.sh` — Added `calculate_log_probs=True`, disabled KL loss, reduced context to 16K, fixed HF_HOME
- `verl_rl/run_grpo_training_h200.sh` — Rewrote for 4-GPU layout: `CUDA_VISIBLE_DEVICES=2,3,4,5`, optimizer offload, 64K context, KL loss enabled
- `.venv` symlink — Fixed to point to correct conda env path

## Next Steps

1. Set up H200 environment (conda env, verl, mamba-ssm, model cache)
2. Launch dense retrieval on H200 GPUs 0-1
3. Launch training with `run_grpo_training_h200.sh` on GPUs 2-5
4. If 64K fits, try increasing to 96K (`max_model_len=98304`, `max_response_length=94208`)
5. If optimizer offload is too slow, try disabling it with reduced context (32K)
6. Monitor reward curves vs A100 16K baseline — 64K context should significantly improve answer rates
