# Multi-Turn Agentic RL for OpenResearcher

**Date:** 2026-02-16
**Status:** Implementation ready for integration testing

## Overview

This document summarizes the implementation of multi-turn tool-integrated GRPO (Group Relative Policy Optimization) training for OpenResearcher, built on top of the [verl](https://github.com/volcengine/verl) framework. The goal is to further improve the SFT-trained OpenResearcher-30B-A3B model by optimizing its search strategy through reinforcement learning with verifiable rewards.

## Architecture

```
SFT checkpoint (OpenResearcher-30B-A3B)
    |
    v
verl GRPO trainer
    |
    +-- Data: (question, ground_truth) pairs in parquet format
    |
    +-- Rollout: SGLang multi-turn with live browser tools
    |   |-- browser.search -> search service (BM25/Dense)
    |   |-- browser.open   -> document retrieval service
    |   |-- browser.find   -> in-page pattern search service
    |
    +-- Reward: Custom reward function extracts <answer> tags
    |          from trajectory, compares against ground truth
    |
    +-- Interaction: Provides feedback on submitted answers,
                    allowing multi-turn refinement
```

## File Structure

```
verl_rl/
  __init__.py
  preprocess_openresearcher.py          # SFT data -> verl parquet conversion
  config/
    openresearcher_multiturn_grpo.yaml   # Main GRPO training config
    tool_config/
      openresearcher_tool_config.yaml    # Browser tool definitions
    interaction_config/
      openresearcher_interaction_config.yaml  # Interaction handler config
  tools/
    __init__.py
    browser_search_tool.py               # Search tool (async HTTP to search service)
    browser_open_tool.py                 # Open/read tool (document retrieval)
    browser_find_tool.py                 # In-page find tool (pattern matching)
    research_reward_tool.py              # [Retained for reference, not used in training]
  interactions/
    __init__.py
    openresearcher_interaction.py        # Trajectory evaluation + feedback
  reward/
    __init__.py
    openresearcher_reward.py             # Custom reward function for verl
```

## Key Design Decisions

### 1. SFT Data to RL Data Conversion

The SFT dataset contains 96K full multi-turn trajectories with 100+ turns each. For RL training, we **discard the trajectories entirely** and keep only the (question, answer) pairs. The RL agent generates its own trajectories during rollout by interacting with live browser tools.

```
SFT record: {question, answer, messages: [100+ turns]}
                                    |
                              DROPPED
                                    |
RL record:  {question, answer} -> parquet -> verl
```

The SFT trajectories served their purpose: training the initial policy that knows how to call browser tools in valid syntax. RL optimizes _which_ queries to search, _which_ results to open, and _when_ to submit an answer.

### 2. System Prompt Consistency

The preprocessing script uses the **same** `DEVELOPER_CONTENT` system prompt from `data_utils.py` that the SFT model was trained on. An earlier version used a custom prompt, which would cause distribution mismatch between the SFT policy and the RL prompt format.

### 3. No `research_reward` Tool

The SFT model was never trained to call a `research_reward` tool. It uses `<answer>...</answer>` tags natively to submit answers. Instead of adding an artificial tool, the reward is computed at two levels:

- **Trajectory-level (GRPO):** verl's reward manager calls `compute_score()` from `verl_rl/reward/openresearcher_reward.py`, which extracts `<answer>` tags and compares against ground truth. This is the primary signal for GRPO advantage computation.

- **Turn-level (interaction):** The interaction handler (`OpenResearcherInteraction`) evaluates each assistant turn. If an `<answer>` tag is found and correct, it terminates the rollout. If incorrect, it provides feedback text ("Your answer appears to be incorrect...") allowing the model to refine.

### 4. Custom Reward Function Registration

verl's `default_compute_score()` dispatches based on `data_source` via a hardcoded if-elif chain. Our `data_source` ("OpenResearcher/OpenResearcher") is not registered there. Instead, we use verl's `custom_reward_function` config:

```yaml
reward:
  custom_reward_function:
    path: verl_rl/reward/openresearcher_reward.py
    name: compute_score
```

The function signature matches verl's calling convention: `compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float`.

### 5. Tool Call Format

OpenResearcher-30B-A3B uses `<tool_call>{"name": "browser.search", "arguments": {...}}</tool_call>` format (hermes), matching verl's `HermesToolParser`. The config sets `format: hermes`.

### 6. Stateful Browser Tools

The browser tools (search/open/find) are **stateful**: `browser.open` references search result IDs from prior `browser.search` calls, and `browser.find` operates on the currently open page. The tool implementations forward requests to the search service with `instance_id` for session tracking. The search service must maintain per-trajectory state during RL rollouts.

## Issues Found and Fixed During Audit

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | Critical | Reward function not in verl's dispatch table | Added `custom_reward_function` config pointing to our reward module |
| 2 | Critical | Missing multi_turn config fields (`max_user_turns`, `max_parallel_calls`, `max_tool_response_length`, `tool_response_truncate_side`, `format`) | Added all required fields to YAML |
| 3 | Critical | System prompt mismatch (custom vs. SFT's DEVELOPER_CONTENT) | Changed to use `DEVELOPER_CONTENT` from `data_utils.py` |
| 4 | Medium | `research_reward` tool never seen by SFT model | Removed from tool config; reward computed via custom_reward_function + interaction |
| 5 | Medium | `browser.find` was a placeholder returning dummy text | Rewrote to forward to search service `/find` endpoint |
| 6 | Medium | `browser.open` API format mismatch | Fixed to use `/open` endpoint with proper payload |
| 7 | Minor | `interaction_config_path` missing from YAML | Added to multi_turn config |
| 8 | Minor | `max_tool_response_length` default (256) too small for search results | Set to 2048 |
| 9 | Medium | `compute_score` signature didn't match verl's calling convention | Fixed to accept `data_source`, `solution_str`, `ground_truth`, `extra_info`, `**kwargs` |

## Usage

### Step 1: Preprocess Data

```bash
# From SFT dataset
python verl_rl/preprocess_openresearcher.py \
    --hf_dataset OpenResearcher/OpenResearcher-Dataset \
    --hf_subset seed_42 \
    --local_save_dir ~/data/openresearcher

# From evaluation benchmarks (for validation)
python verl_rl/preprocess_openresearcher.py \
    --eval_dataset browsecomp \
    --local_save_dir ~/data/openresearcher_eval
```

### Step 2: Start Search Service

The search service must be running during RL training (same as during inference):

```bash
bash scripts/start_search_service.sh bm25 8090
```

### Step 3: Run GRPO Training

```bash
# Default: 8 GPUs, TP=2 for the 30B-A3B MoE model
bash verl_rl/run_grpo_training.sh

# Custom GPU count
N_GPUS=4 bash verl_rl/run_grpo_training.sh

# Override any config via command line
bash verl_rl/run_grpo_training.sh \
    trainer.total_epochs=3 \
    algorithm.grpo_n_generations=8 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15
```

The launch script (`verl_rl/run_grpo_training.sh`):
- Validates prerequisites (data files exist, search service reachable)
- Sets correct TP=2 for the 30B MoE model
- Configures FSDP with ref model offloading for memory efficiency
- Uses absolute paths for tool/interaction configs
- Supports environment variable overrides (`N_GPUS`, `TP_SIZE`, `SEARCH_SERVICE_URL`, etc.)

## Remaining Work (Not Yet Implemented)

1. **Search service session management:** The search service (backend.py) currently doesn't maintain per-trajectory state for open/find operations. It needs a session layer that maps `instance_id` to browser state (opened pages, cursor positions). This is the main infrastructure gap.

2. **Launch script:** A script that orchestrates search service startup, SGLang server launch, and verl training in the correct order.

3. **Small-scale validation:** Run on a small subset of questions with `max_assistant_turns: 10` and `grpo_n_generations: 4` on a single node to verify the pipeline works end-to-end before scaling up.

4. **Reward shaping experiments:** Minimal heuristics for step-level rewards (penalize duplicate queries, empty searches) to improve credit assignment in long trajectories.

5. **Curriculum design:** Start RL on easier questions (shorter answers, more common topics) before progressing to harder ones to avoid reward sparsity early in training.
