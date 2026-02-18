# Multi-Turn Agentic RL for OpenResearcher

**Date:** 2026-02-16
**Status:** Pipeline validated end-to-end; reward function config fix pending final test run

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
    +-- Rollout: vLLM multi-turn with live browser tools
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
  run_grpo_training.sh                  # Launch script with env var overrides
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
  patches/
    modeling_nemotron_h.py               # Patched NemotronH model (flash attention support)
    mamba_ssm__init__.py                 # Patched mamba-ssm __init__ (graceful import)
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

verl's `default_compute_score()` dispatches based on `data_source` via a hardcoded if-elif chain. Our `data_source` ("OpenResearcher/OpenResearcher") is not registered there. Instead, we use verl's `custom_reward_function` config as a **top-level** key (not nested under `reward`):

```yaml
# Top-level key in the YAML config (NOT under reward:)
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

## Issues Found and Fixed During Integration Testing (Feb 16)

### Dependency Issues

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | mamba-ssm CUDA ABI mismatch crashes import | Pre-built wheel compiled against different torch/CUDA version | Built from source with `TORCH_CUDA_ARCH_LIST="8.0"` (see patches section) |
| 2 | causal-conv1d undefined symbol errors | Same ABI mismatch as mamba-ssm | Built from source with `TORCH_CUDA_ARCH_LIST="8.0"` |
| 3 | flash-attn not installed | Missing package | `pip install flash-attn --no-build-isolation` |

### SGLang Backend Issues (Resolved by Switching to vLLM)

| # | Issue | Details |
|---|-------|---------|
| 1 | NCCL deadlock with TP=2 | With 4 SGLang replicas on 8 GPUs, one replica consistently gets stuck during NCCL init (99% SM utilization, no memory progress). Systematic, not transient. |
| 2 | OOM during server init | SGLang server killed by OOM during `AgentLoopManager.__init__() -> self.sleep()` even with `gpu_memory_utilization=0.5` and `param_offload=True` |

**Resolution:** Switched to vLLM backend. verl's multi-turn `AgentLoopManager` is engine-agnostic (sits above the rollout replica layer). Both vLLM and SGLang are registered backends:
```python
# verl/workers/rollout/replica.py
RolloutReplicaRegistry.register("vllm", _load_vllm)
RolloutReplicaRegistry.register("sglang", _load_sglang)
```

### NemotronH Model Issues

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `NemotronHForCausalLM does not support Flash Attention 2.0` | `NemotronHPreTrainedModel` missing `_supports_flash_attn_2 = True` flag despite having `NemotronHFlashAttention2` implementation | Patched model class (see patches section below) |
| 2 | mamba-ssm import crash prevents model config loading | CUDA extension load fails with undefined symbols | Patched `mamba_ssm/__init__.py` with try/except (see patches section) |

### Config Issues

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `NotImplementedError: Reward function is not implemented for data_source='OpenResearcher/OpenResearcher'` | `custom_reward_function` was nested under `reward:` but verl expects it as a **top-level** config key | Moved to top-level in YAML |

## Required Patches

Two source files need patching for NemotronH to work correctly with flash attention and verl. Patched copies are saved in `verl_rl/patches/`.

### Patch 1: NemotronH Flash Attention Support

**File:** `modeling_nemotron_h.py`
**Locations to patch:**
- HuggingFace cache: `~/.cache/huggingface/modules/transformers_modules/OpenResearcher/OpenResearcher_hyphen_30B_hyphen_A3B/<commit_hash>/modeling_nemotron_h.py`
- Site-packages (if installed): `<site-packages>/modeling_nemotron_h.py`

**What to change:** In the `NemotronHPreTrainedModel` class, add two flags:

```python
# BEFORE (around line 1208-1212):
class NemotronHPreTrainedModel(PreTrainedModel):
    config_class = NemotronHConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["NemotronHBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True

# AFTER:
class NemotronHPreTrainedModel(PreTrainedModel):
    config_class = NemotronHConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["NemotronHBlock"]
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _is_stateful = True
```

**Why:** The model code already defines `NemotronHFlashAttention2` and `NemotronHSdpaAttention` in `NEMOTRONH_ATTENTION_CLASSES`, but `transformers` checks the `_supports_flash_attn_2` / `_supports_sdpa` class flags before allowing these attention backends. Without the flags, transformers rejects flash attention and forces eager (O(n^2)) attention.

**Apply automatically:**
```bash
cp verl_rl/patches/modeling_nemotron_h.py \
   ~/.cache/huggingface/modules/transformers_modules/OpenResearcher/OpenResearcher_hyphen_30B_hyphen_A3B/*/modeling_nemotron_h.py
```

### Patch 2: mamba-ssm Graceful Import

**File:** `<site-packages>/mamba_ssm/__init__.py`

**What to change:** Wrap CUDA extension imports in try/except so that when mamba-ssm is compiled against a different torch version, transformers can still load the model config without crashing.

```python
# BEFORE:
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
# ...

# AFTER:
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    selective_scan_fn = None
    mamba_inner_fn = None
    Mamba = None
    Mamba2 = None
    MambaLMHeadModel = None
```

**Apply automatically:**
```bash
MAMBA_INIT=$(python -c "import mamba_ssm; print(mamba_ssm.__file__)")
cp verl_rl/patches/mamba_ssm__init__.py "$MAMBA_INIT"
```

## Validated Working Environment

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.9.1+cu128 | |
| transformers | 4.57.6 | |
| vllm | 0.13.0 | Recommended rollout backend |
| sglang | 0.5.6 | Has NCCL/OOM issues with NemotronH; not recommended |
| verl | 0.7.0 | |
| mamba-ssm | 2.3.0 | Must build from source (see below) |
| causal-conv1d | 1.6.0 | Must build from source (see below) |
| flash-attn | 2.8.3 | `pip install flash-attn --no-build-isolation` |

### Build-from-source dependencies

```bash
# causal-conv1d (required by mamba-ssm)
cd /tmp && git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d && TORCH_CUDA_ARCH_LIST="8.0" pip install -e . --no-build-isolation

# mamba-ssm
cd /tmp && git clone https://github.com/state-spaces/mamba.git
cd mamba && TORCH_CUDA_ARCH_LIST="8.0" pip install -e . --no-build-isolation

# flash-attn
pip install flash-attn --no-build-isolation
```

Adjust `TORCH_CUDA_ARCH_LIST` for your GPU architecture (8.0 = A100, 9.0 = H100/H200).

## Working Launch Command (vLLM + Flash Attention)

```bash
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/verl_rl/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/openresearcher_tool_config.yaml"
INTERACTION_CONFIG="$CONFIG_PATH/interaction_config/openresearcher_interaction_config.yaml"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='openresearcher_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=OpenResearcher/OpenResearcher-30B-A3B \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
```

**Pipeline status as of Feb 16:**
- Model loading with flash attention: PASS
- FSDP initialization (8 workers): PASS
- vLLM server initialization (TP=2): PASS
- Rollout generation started: PASS
- Reward function: Config fixed (moved `custom_reward_function` to top-level), pending test run

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

### Step 3: Apply Patches

```bash
# Patch NemotronH flash attention support
cp verl_rl/patches/modeling_nemotron_h.py \
   ~/.cache/huggingface/modules/transformers_modules/OpenResearcher/OpenResearcher_hyphen_30B_hyphen_A3B/*/modeling_nemotron_h.py

# Patch mamba-ssm graceful import
MAMBA_INIT=$(python -c "import mamba_ssm; print(mamba_ssm.__file__)")
cp verl_rl/patches/mamba_ssm__init__.py "$MAMBA_INIT"
```

### Step 4: Run GRPO Training

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

## Future Plans

1. **Complete end-to-end validation:** Run the full pipeline with the reward function fix to confirm training loop completes. The last remaining issue (reward config) is fixed but untested.

2. **Search service session management:** The search service (backend.py) currently doesn't maintain per-trajectory state for open/find operations. It needs a session layer that maps `instance_id` to browser state (opened pages, cursor positions). This is the main infrastructure gap.

3. **Small-scale validation:** Run on a small subset of questions with `max_assistant_turns: 10` and `grpo_n_generations: 4` on a single node to verify reward signals are meaningful before scaling up.

4. **Reward shaping experiments:** Minimal heuristics for step-level rewards (penalize duplicate queries, empty searches) to improve credit assignment in long trajectories.

5. **Curriculum design:** Start RL on easier questions (shorter answers, more common topics) before progressing to harder ones to avoid reward sparsity early in training.

6. **SGLang debugging:** Investigate the NCCL deadlock and OOM issues with NemotronH on SGLang. SGLang may offer better performance than vLLM for certain workloads once the initialization issues are resolved.

7. **Multi-node scaling:** Test on 2+ nodes with the current vLLM backend. The verl config supports `nnodes > 1` but this hasn't been validated.

8. **Upstream patches:** Submit the `_supports_flash_attn_2` / `_supports_sdpa` fix to the NemotronH model repo so future users don't hit the same issue.
