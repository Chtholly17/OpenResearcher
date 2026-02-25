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

## Next Steps

1. Monitor `grpo_curriculum_500` run on wandb for reward curves and answer rates
2. Once stage 1 converges, switch to `train_curriculum_1k` for stage 2
3. Progressively expand to 2k and full non-trivial set
4. Consider whether forced answer prefix helps or hurts — no correct answers have come from it so far
5. Migrate to H200 using the tuning guide above — prioritize TP=2 and disabling FSDP offload
