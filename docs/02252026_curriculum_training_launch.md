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

## Launch Issues & Fixes

1. **`ppo_mini_batch_size=32` validation error**: Initially set mini_batch to 32 (train_batch_size × n) thinking it iterates over rollouts. verl validates `train_batch_size >= ppo_mini_batch_size` — it iterates over prompts. Reverted to 4.

2. **HF_HOME empty string**: `$HOME` resolved to empty in nohup context, giving `HF_HOME=/.cache/huggingface`. Fixed by passing explicit path `/home/hqhardy/.cache/huggingface`.

3. **wandb not logged in**: Initial launch failed with `wandb.errors.UsageError: No API key configured`. Fixed by running `wandb login` first.

## Current Training Run

**Run**: `grpo_curriculum_500`
**Wandb**: https://wandb.ai/tuisaac/openresearcher_rl/runs/o7y7ulpy
**Config**:
- Training data: `train_curriculum_500.parquet` (500 mid-pass-rate examples)
- Validation data: `test.parquet`
- n=8 (GRPO group size)
- train_batch_size=4 → 32 rollouts per step
- Total steps: 625 (500/4 × 5 epochs)
- 128K context, max_num_seqs=256, TP=4
- Budget: 200 soft warning / 300 hard cutoff
- val_before_train=True
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

## Next Steps

1. Monitor `grpo_curriculum_500` run on wandb for reward curves and answer rates
2. Once stage 1 converges, switch to `train_curriculum_1k` for stage 2
3. Progressively expand to 2k and full non-trivial set
4. Consider whether forced answer prefix helps or hurts — no correct answers have come from it so far
