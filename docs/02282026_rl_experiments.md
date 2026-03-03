# RL Training Experiments Log

All runs use Qwen3-8B on 4× A100-80GB (GPUs 4-7), with dense retrieval service on GPUs 0-1.
Base config: `verl_rl/config/openresearcher_multiturn_grpo_qwen3.yaml`

---

## Common Fixed Parameters (all runs)

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-8B` |
| Tool format | `hermes` (JSON tool calls) |
| GPUs | 4× A100-80GB (CUDA_VISIBLE_DEVICES=4,5,6,7) |
| Search service | Dense (Qwen3-Embedding-8B) on GPUs 0-1, port 8090 |
| TP size | 2 |
| train_batch_size | 4 |
| n (GRPO rollouts) | 8 |
| max_prompt_length | 4096 |
| ppo_mini_batch_size | 4 |
| ppo_micro_batch_size_per_gpu | 1 |
| kl_loss | True, coef=0.001, type=low_var_kl |
| param_offload | False |
| optimizer_offload | False |
| model_dtype | bf16 |
| gpu_memory_utilization | 0.6 |
| max_model_len | 16384 |
| max_num_seqs | 256 |
| log_prob_micro_batch_size_per_gpu | 4 |
| max_assistant_turns | 500 |
| max_user_turns | 500 |
| max_tool_response_length | 1024 |
| total_epochs | 2 (250 steps on 500-sample data) |
| val_before_train | False |
| test_freq | 50 |
| logger | console + wandb |

---

## Experiment Runs

### debug (pre-v0.1)
- **wandb**: `grpo_qwen3_debug` — https://wandb.ai/tuisaac/openresearcher_rl/runs/sb6nerab
- **Data**: `data/qwen3/train_curriculum_500.parquet` (500 examples, Nemotron pass_rate 0.375–0.625)
- **max_response_length**: 28672 → **OOM at actor update** (logits = 17.4 GB). Reduced to 12288.
- **Outcome**: Crashed at OOM, ~3 steps completed.

---

### v0.1
- **wandb**: `grpo_qwen3_v0.1` — https://wandb.ai/tuisaac/openresearcher_rl/runs/4camav6r
- **Data**: `data/qwen3/train_curriculum_500.parquet`
- **max_response_length**: 12288
- **Reward function**: v0.1 — binary (correct=1.0, wrong_explicit=0.1, no_answer=0.0). No efficiency scaling.
- **Interaction**: Original — on wrong answer, returns "try again" feedback, does NOT terminate.
- **Key diffs from debug**: `max_response_length` reduced to 12288; wandb enabled; `ppo_micro_batch_size_per_gpu=1`
- **Result (180 steps)**: Reward flat at ~0.18 avg. 11% correct rate. 36% no-correct steps.
  - **Issue found**: Efficiency-scaled wrong-answer reward (0.1–0.3) caused near-uniform rewards in no-correct batches → zero gradient on ~36% of steps.
  - **Issue found**: Reward function read FIRST `<answer>` tag — retry correct answers silently discarded.
  - **Issue found**: 61% of wrong trajectories hit 12K token cap; retry feedback burned context.

---

### v0.2
- **wandb**: `grpo_qwen3_v0.2` — https://wandb.ai/tuisaac/openresearcher_rl/runs/vjajnkdh
- **Data**: `data/qwen3/train_curriculum_500.parquet`
- **max_response_length**: 12288
- **Reward function**: v0.2 — same binary rewards as v0.1. Changed `extract_answer` to use **last** `<answer>` tag.
- **Interaction**: **Terminate on any explicit answer** (correct or wrong). Empty response, no retry feedback.
- **Result (250 steps, full run)**: Reward flat at ~0.19 avg. Final val acc = 32.5% (20 samples, greedy).
  - **Good**: Token cap eliminated (clip_ratio → 0). Step time ~200s (↓ from 270s). Trajectories shorter.
  - **Issue found**: Model collapsed to 7.3 turns/step by step 250 — learned to submit immediately after 1–2 searches to exploit 0.1 format reward cheaply. Terminate-on-submit removed all pressure to search.
  - Checkpoint saved: `checkpoints/openresearcher_rl/grpo_qwen3_v0.2/global_step_250/`

---

### v0.3 (curriculum_500)
- **wandb**: `grpo_qwen3_v0.3` — https://wandb.ai/tuisaac/openresearcher_rl/runs/xc3qcok9
- **Data**: `data/qwen3/train_curriculum_500.parquet`
- **max_response_length**: 12288
- **Reward function**: v0.3 — added `MIN_SEARCH_CALLS=5` gate: wrong explicit answer earns 0.1 only if ≥5 `browser.search` calls in trajectory; else 0.0. Correct always earns 1.0.
- **Interaction**: On wrong answer: **do NOT terminate**, return empty response. On correct: terminate.
- **Result (112 steps, killed early)**: Reward flat at ~0.15 avg. Turn count restored to 300–500 (collapse fixed). BUT: 61% of wrong-answer samples hit the lazy gate (< 5 searches), even with 100–500 turns — model was doing deep browsing via `browser.open`/`browser.find` with only 2 search queries. Gate was penalizing legitimate deep-browsing trajectories.
  - **Root cause confirmed**: Fundamental question difficulty mismatch. `train_curriculum_500` was selected for Nemotron (50% pass rate), but Qwen3 base model gets only ~12% correct on these. ~30% of steps have zero correct rollouts → zero gradient.

---

### v0.3-highpass

- **wandb**: `grpo_qwen3_v0.3_highpass` — https://wandb.ai/tuisaac/openresearcher_rl/runs/96ike2x9
- **Data**: `data/qwen3_highpass/train.parquet` — **1,874 examples** from `Chtholly17/OR_reject_sampling` filtered to Nemotron **pass_rate ≥ 0.875** (≥7/8 judges correct). Includes pass_rate=0.875 (313) and pass_rate=1.0 (1,581).
- **max_response_length**: 12288
- **Reward function**: v0.3 (MIN_SEARCH_CALLS=5 gate, last-answer extraction)
- **Interaction**: v0.3 (silent non-termination on wrong, terminate on correct)
- **Result (23 steps, killed early)**: Correct rate jumped to **43%** (was 12%). No-correct steps: **0%** (was 30%). Score mean 0.39–0.42 (was 0.15–0.19). Steps 22–23 reached 0.756 and 0.806 — clear upward signal.
  - **Issue found**: `MIN_SEARCH_CALLS=5` gate was wrong for this easy dataset. Correct answers only need 1.6 searches on average, so requiring 5 for the format reward penalised 43% of wrong-answer samples making comparable effort. Gate removed in v0.4.

---

### v0.4-highpass ← **CURRENT RUN**
- **wandb**: `grpo_qwen3_v0.4_highpass` — https://wandb.ai/tuisaac/openresearcher_rl/runs/w1f4y9sb
- **Data**: `data/qwen3_highpass/train.parquet` (1,874 examples, pass_rate ≥ 0.875)
- **max_response_length**: 12288
- **Reward function**: v0.4 — binary rewards (correct=1.0, wrong_explicit=0.1, no_answer=0.0). Min-search gate removed.
- **Interaction**: v0.3 (silent non-termination on wrong, terminate on correct)

---

## Reward Function Versions

| Version | Correct | Wrong explicit | No answer | Notes |
|---------|---------|---------------|-----------|-------|
| pre-v0.1 | 0.8–1.2 (eff) | 0.1–0.3 (eff) | 0.0 | Efficiency scaling caused uniform wrong rewards |
| v0.1 | 1.0 | 0.1 (fixed) | 0.0 | Fixed binary rewards |
| v0.2 | 1.0 | 0.1 (fixed) | 0.0 | + last `<answer>` tag extraction |
| **v0.3** | 1.0 | 0.1 if ≥5 searches, else 0.0 | 0.0 | + min-search gate |

## Interaction Versions

| Version | On wrong answer | On correct answer |
|---------|-----------------|-------------------|
| original | Return "try again" text, do NOT terminate | Terminate |
| v0.2 | Terminate immediately, empty response | Terminate |
| **v0.3** | Do NOT terminate, empty response | Terminate |

## Environment Notes

- `.venv` → `/opt/dlami/nvme/hqhd-miniconda3/envs/openresearcher` (conda env, Python 3.12)
- Use `python -m pip` (pip shebang is stale)
- Monkey patches applied: verl tool schemas, verl fsdp_workers, mamba_ssm __init__, NemotronH modeling
- causal_conv1d 1.6.0 and mamba_ssm 2.3.0 built from source at `/tmp/causal-conv1d` and `/tmp/mamba`
