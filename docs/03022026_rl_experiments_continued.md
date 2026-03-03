# RL Training Experiments — 2026-03-02 (Continued)

Continuation of `02282026_rl_experiments.md`. Previous runs (v0.1–v0.5-bs4) documented there.

---

## Summary of Runs Today

### v0.5-bs16
- **wandb**: `grpo_qwen3_v0.5_bs16` — https://wandb.ai/tuisaac/openresearcher_rl/runs/0hvfm7tl
- **Data**: `data/qwen3_mid/train.parquet` (547 examples, Nemotron pass_rate 0.5–0.8)
- **Key changes vs v0.5-bs4**: `train_batch_size=16`, `ppo_mini_batch_size=16`
- **Reward**: v0.5 (correct+searched→length bonus [0.8,1.2], correct+no_search→0.3, wrong+searched→0.1, wrong+no_search→0.0)
- **Interaction**: v0.3 (silent non-termination on wrong, terminate on correct)
- **lr**: 1e-6, `clip_grad=1.0`
- **Result (16 steps)**: Reward still zigzag (~0.13–0.36). `pg_clipfrac=0.0` throughout.
  **Root cause found**: KL divergence = 0.002–0.004 per step → policy barely moving. With lr=1e-6 and grad_norm≈0.3, weight update per step ≈ 3e-7 (essentially zero). Policy never escapes initial weights.

---

### v0.6 (lr=1e-5)
- **wandb**: `grpo_qwen3_v0.6_lr1e5` — https://wandb.ai/tuisaac/openresearcher_rl/runs/6cryalac
- **Data**: `data/qwen3_mid/train.parquet`
- **Key change**: `lr: 1e-6 → 1e-5` (10×), `train_batch_size=16`
- **Reward/Interaction**: same as v0.5-bs16
- **Result (68 steps, full run)**:
  - Steps 1–48: **genuine learning** — score mean improved 0.22 → 0.34, correct rate 11.9% → 17.5%
  - Steps 49–64: entropy rising (0.05 → 0.27), instability signs
  - Steps 65–68: **catastrophic divergence** — entropy exploded 0.06 → 11.2, score collapsed to 0.002, KL_loss jumped from 0.001 → 0.65
  - Trigger: grad_norm spike to 2.97 at step 66
  - **Best checkpoint**: `global_step_51` (before divergence)
  - **Lesson**: lr=1e-5 too aggressive — policy learned initially then diverged late

---

### v0.7 (lr=3e-6, clip_grad=0.5) ← **CURRENT RUN**
- **wandb**: `grpo_qwen3_v0.7_lr3e6` — https://wandb.ai/tuisaac/openresearcher_rl/runs/jlmiqsax
- **Data**: `data/qwen3_mid/train.parquet`
- **Key changes**: `lr=3e-6` (3× higher than v0.5, 3× lower than v0.6), `clip_grad=0.5` (tighter gradient clipping to prevent the step-66-style spike)
- **Reward/Interaction**: same as v0.5-bs16
- **Hypothesis**: lr=3e-6 should move the policy meaningfully (unlike v0.5 which was frozen) while clip_grad=0.5 prevents the gradient spike that triggered v0.6's collapse

---

## Key Learnings from Today

### 1. Learning Rate was the core problem all along
All runs v0.1–v0.5 used lr=1e-6 → weight change per step ≈ 3e-7 → policy essentially frozen. `pg_clipfrac=0.0` throughout was the diagnostic signal (policy ratio never left [0.8, 1.2]).

### 2. The Goldilocks zone: lr=3e-6 to 1e-5
- lr=1e-6: too slow, no policy movement
- lr=1e-5: too fast, diverges after ~60 steps
- lr=3e-6: target zone — fast enough for learning, slow enough to avoid divergence

### 3. Gradient clipping matters for stability
v0.6 collapsed when grad_norm hit 2.97 at step 66. With clip_grad=0.5 (vs default 1.0), any spike is halved before the optimizer step.

### 4. Correct rate DID improve in v0.6 stable phase
11.9% → 17.5% (+47% relative) before divergence. Proves the RL pipeline CAN learn when lr is right.

### 5. Data quality: pass_rate 0.5–0.8 is the right zone
- pass_rate ≥ 0.875 (v0.4): 43% correct but memory-recall, model doesn't need to search
- pass_rate 0.375–0.625 (v0.1–v0.3): 12% correct, too hard, 30% dead-gradient steps
- pass_rate 0.5–0.8 (v0.5+): 12–17% correct, model must search, correct rate improving

### 6. System prompt fix was necessary
Before the fix, 29% of correct answers used 0 searches (pure memory recall). After the explicit "MUST use browser.search" instruction, 92% of correct answers use searches.

---

## Reward Function Versions (v0.5 is current)

| Version | Correct+searched | Correct+no_search | Wrong+searched | Wrong+no_search |
|---------|-----------------|-------------------|----------------|-----------------|
| v0.1–v0.4 | 1.0 | 1.0 | 0.1 | 0.1 |
| v0.5 | **0.8+0.4×eff (length bonus)** | **0.3** | 0.1 | **0.0** |

`eff = max(0, 1 - num_turns/500)` → short correct answers (10 turns) get 1.19, long ones (500 turns) get 0.8.

## Current Data Files

| File | Size | Pass rate | Purpose |
|------|------|-----------|---------|
| `data/qwen3_mid/train.parquet` | 547 | 0.5–0.8 | Current training data |
| `data/qwen3_mid/test.parquet` | 20 | 0.5–0.8 | Validation |
| `data/qwen3_highpass/train.parquet` | 1874 | ≥0.875 | Too easy (memory recall) |
| `data/qwen3/train_curriculum_500.parquet` | 500 | 0.375–0.625 | Too hard for Qwen3 |

## System Prompt (v0.5+)

```
You are a research assistant. To answer questions accurately you MUST use the browser tools — do not answer from memory alone.

Research process:
1. Call browser.search with a precise query to find relevant sources
2. Call browser.open to read the most promising pages
3. Call browser.find to locate specific facts within a page
4. Submit a concise answer with submit_answer once you are confident
```

## Interaction (v0.3, current)

- Wrong answer → do NOT terminate, return empty response (model continues searching)
- Correct answer → terminate immediately
- Last `<answer>` tag used for reward (fixes retry-correct bug from v0.1)
