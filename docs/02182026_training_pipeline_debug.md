# 02/18/2026 - Training Pipeline Debug: OOM Fix & Reward Signal

Continuation of 02/17/2026 debugging session. Started with the TODO list from the previous changelog.

## Resolved: num_turns=2 (Agent Loop Config)

**Root cause**: verl's default `default_agent_loop` is `"single_turn_agent"` (defined in `verl/workers/config/rollout.py:73`). `SingleTurnAgentLoop` generates one response and terminates — it never calls the tool parser or executes tool calls.

**Fix**: Added to `run_grpo_training.sh`:
```
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

This switches to `ToolAgentLoop` which:
1. Generates a response
2. Calls `NemotronToolParser.extract_tool_calls()` to find tool calls
3. Executes tool calls via the tool config
4. Appends tool responses to context
5. Repeats until no more tool calls, answer submitted, or budget exhausted

**Result**: num_turns jumped from 2 (all trajectories) to min=2, max=12, mean=7.8-11.5. The NemotronToolParser processes 1526/1527 tool calls successfully.

## Resolved: OOM During Adam Optimizer Step

### Problem

Every training attempt OOM'd at the same point — `torch.optim.adam._multi_tensor_adam`:
```
torch.OutOfMemoryError: Tried to allocate 620.00 MiB.
GPU 0: 72.94 GiB allocated by PyTorch, 4.04 GiB by vLLM process.
Total: ~77 GiB > 79.25 GiB (A100-80GB)
```

### What Didn't Work

1. **`param_offload=True` + `optimizer_offload=True` (FSDP v1)**: These flags only move data to CPU *between phases* (rollout ↔ training). During the optimizer step itself, everything is loaded back to GPU. The 72.94GB number was unchanged.

2. **`grad_offload=True` (FSDP v1)**: Same — no effect on peak memory during optimizer step.

3. **`PYTORCH_ALLOC_CONF=expandable_segments:True`**: Helps with fragmentation but doesn't reduce peak usage. The 620MB gap isn't fragmentation.

4. **TP=2 with `gpu_memory_utilization=0.3`**: vLLM couldn't allocate KV cache — model weights per GPU doubled (~15GB with TP=2), leaving insufficient room at 0.3 utilization.

5. **FSDP v2 (`strategy=fsdp2`) with `offload_policy=True`**: The `CPUOffloadPolicy` is supposed to offload per-parameter during the optimizer step. But verl's hybrid engine manually loads everything back to GPU before training (`offload_fsdp_model_to_gpu()`), defeating the policy. Peak memory: still 72.94GB.

### What Worked: `model_dtype=bf16`

**Root cause analysis**: The 72.94GB breakdown with `model_dtype=fp32` (default):
- Master weights (fp32, FSDP-sharded): 30B × 4 bytes / 8 GPUs = **15 GB**
- Adam optimizer states (fp32, 2 per param): 30B × 4 × 2 / 8 = **30 GB**
- Gradients: ~7.5-15 GB
- Activations + buffers: remainder

**Fix**: Set `actor_rollout_ref.actor.fsdp_config.model_dtype=bf16`

This stores master weights in bf16 instead of fp32:
- Master weights (bf16): 30B × 2 / 8 = **7.5 GB** (saved 7.5 GB)
- Optimizer states also created in bf16 (PyTorch creates states matching param dtype)
- Peak memory dropped from **72.94 GB → 39.18 GB**

The 7.5GB savings far exceeds the 620MB needed. Training runs stably with no OOM.

## Resolved: Reward = 0.0 (Cold Start Problem)

### Problem

During validation, the model never submits an `<answer>` tag. It exhausts its entire token budget (`max_response_length=2048`) on tool calls (search → open → find → repeat). Since `compute_score()` only returns non-zero reward when it finds `<answer>` in the trajectory, every sample gets 0.0. With all-zero rewards, GRPO advantages are all zero → no gradient signal → no learning.

### Analysis

- `solution_str` passed to `compute_score()` is the full decoded trajectory (all response tokens including tool calls and tool responses)
- `max_response_length` is the *total* token budget across all turns (not per-turn)
- With ~80-100 tokens per turn (think + tool call) and mean ~8 turns, the model uses ~800 tokens on generation, plus tool responses consuming the rest
- The model has no incentive to stop researching and submit an answer

### Fix: Format Reward

Modified `verl_rl/reward/openresearcher_reward.py` to add a format reward component:

| Condition | Reward |
|-----------|--------|
| Correct answer (`<answer>` matches ground truth) | **1.0** |
| Wrong answer (submitted but incorrect) | **0.1** |
| No answer submitted (budget exhausted on tool calls) | **0.0** |

The 0.1 format reward creates gradient signal: trajectories that submit answers (even wrong ones) get positive advantage vs. trajectories that don't. This teaches the model to stop researching and actually answer within the token budget.

### Result

- Step 23: first non-zero `critic/score/max: 0.1` (model submitted an answer)
- Step 24: `critic/score/mean: 0.025` (2/8 samples submitted answers)
- Validation reward improved from 0.0 → 0.00197 (some correct answers appearing)

## Also Changed: max_assistant_turns

Reduced from 30 to 15 in `openresearcher_multiturn_grpo.yaml`. With 30 turns, the model had too much room to endlessly search. Reducing to 15 applies light pressure to submit answers sooner while still allowing meaningful research trajectories.

## Current Training Status

Training is running stably on 8× A100-80GB:
- **Memory**: 39.18 GB peak per GPU (stable, no OOM)
- **Throughput**: ~55 tokens/s, ~50s per step
- **num_turns**: min=2, max=12, mean=7.5-9.25
- **Reward signal**: sparse but non-zero (format reward working)
- **Process**: PID on the machine, log at `/tmp/grpo_bf16_run.log`

## Files Changed (This Session)

| File | Change | Description |
|------|--------|-------------|
| `verl_rl/run_grpo_training.sh` | Modified | Added `default_agent_loop=tool_agent`, `model_dtype=bf16`, `param_offload=True`, `optimizer_offload=True`; TP=4, `gpu_memory_utilization=0.5`, `max_model_len=8192`, `max_response_length=2048` |
| `verl_rl/reward/openresearcher_reward.py` | Modified | Added format reward (0.1 for submitting any answer, 1.0 for correct) |
| `verl_rl/config/openresearcher_multiturn_grpo.yaml` | Modified | `max_assistant_turns: 30` → `15` |

## Key Learnings

1. **verl's FSDP v1 offload flags (`param_offload`, `optimizer_offload`) only work between phases**, not during the optimizer step. Don't expect them to reduce peak training memory.

2. **FSDP v2 `offload_policy` is defeated by verl's manual GPU loading** in the hybrid engine. The policy tries to manage memory dynamically, but verl explicitly loads everything to GPU before training.

3. **`model_dtype` is the real lever for GPU memory on large models.** Switching from fp32 to bf16 master weights halved the parameter + optimizer memory (72.94 → 39.18 GB). Trade-off: potential training instability, but many recent RL papers use bf16 successfully.

4. **Zero reward is a cold-start problem, not just a model problem.** Without format rewards, there's literally no gradient signal. The model can't learn to submit answers if it never gets credit for trying.

5. **`default_agent_loop` defaults to `single_turn_agent`** — always set it explicitly for multi-turn tool-calling workloads.

## Required Monkey Patches (External Libraries)

Two files in `verl_rl/patches/` must be copied into installed packages before training. These are **not** runtime patches — they replace files on disk in `~/.cache` and `site-packages`.

1. **`verl_rl/patches/modeling_nemotron_h.py`** → replaces `~/.cache/huggingface/modules/transformers_modules/OpenResearcher/OpenResearcher_hyphen_30B_hyphen_A3B/*/modeling_nemotron_h.py`
   - Adds `_supports_flash_attn_2 = True` to the model class (line 1212)
   - Without this, the model falls back to O(n^2) eager attention instead of FlashAttention2
   - The upstream model code implements FlashAttention2 but is missing this flag

2. **`verl_rl/patches/mamba_ssm__init__.py`** → replaces `mamba_ssm/__init__.py` in the installed mamba-ssm package
   - Wraps CUDA extension imports in `try/except` to handle ABI mismatches
   - Without this, `import mamba_ssm` crashes if compiled against a different torch version

**Not a monkey patch**: The custom `NemotronToolParser` (`verl_rl/parsers/nemotron_tool_parser.py`) registers into verl's plugin registry at runtime via `@ToolParser.register("nemotron")`. It does not modify verl's source code.

## TODO for Next Session

1. **Monitor training progress** — check if reward/acc trends upward over more steps. Watch for training instability from bf16 master weights.

2. **Remove debug prints** from `verl_rl/parsers/nemotron_tool_parser.py` — they're verbose and no longer needed.

3. **Consider increasing `max_response_length`** — currently 2048, which limits research depth. The model hits the budget cap frequently (`response_length/clip_ratio: 0.25-0.625`). More budget would let it research more before submitting.

4. **Tune format reward weight** — 0.1 may be too low or too high. If the model learns to always submit garbage answers early (to get 0.1), it won't learn to research. If too low, the signal is too sparse. Monitor behavior.

5. **Investigate `pg_clipfrac=0.0`** — this means no clipping is happening, which could indicate the KL penalty is too strong or the policy isn't changing. May need to adjust `kl_loss_coef` or `clip_ratio`.
