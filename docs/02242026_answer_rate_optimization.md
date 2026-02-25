# 02/24/2026 - Answer Rate Optimization: Budget Control & Forced Answer Prefix

Continuation of 02/22/2026 session. Goal: achieve 80%+ answer rate during validation rollouts so GRPO training has meaningful reward signal.

## Starting Point

From prior sessions, the core bottleneck was identified: `max_response_length` is the total token budget for BOTH model-generated tokens AND tool response tokens. The SFT model needs 50-150+ tool call rounds (median 67, p90 220) to naturally produce answers, consuming 30K-120K+ tokens.

Best prior result: **v18** with 128K context, 500+500 turns → 46.2% answer rate (partial), 30.8% correct.

## Key Discovery: Budget Tracking Was Broken

The tool call budget system (soft warning at N calls, hard cutoff at M calls) used `instance_id` for tracking. But verl's `_call_tool()` creates a **new** `instance_id` per tool call via `tool.create()`. This meant `total_tool_calls` counter reset every call — the budget never accumulated.

**Fix**: Changed all browser tools (search/open/find) to use `agent_data.request_id` (the trajectory-level ID passed via kwargs) instead of the per-call `instance_id`:
```python
agent_data = kwargs.get("agent_data")
traj_id = getattr(agent_data, "request_id", instance_id) if agent_data else instance_id
increment_tool_calls(traj_id)
```

## Forced Answer Prefix Injection

When tool budget is exhausted, the model receives "budget exhausted" as a tool response but keeps calling tools (the SFT model was trained to always use tools). This creates a wasteful loop: model calls tool → gets "exhausted" message → calls another tool → repeat until turn limit.

**Solution**: Patched verl's `_handle_processing_tools_state()` in `tool_agent_loop.py` to detect `budget_exhausted` metadata from tool responses. When detected, inject a partial assistant message as a forced generation prefix:

```python
if budget_exhausted:
    answer_prefix = (
        "<think>\nMy research budget is exhausted. Based on all the "
        "information I have gathered, the answer is:\n</think>\n\n"
        "Exact Answer: "
    )
    prefix_ids = await self.apply_chat_template(
        [{"role": "assistant", "content": answer_prefix}], ...
    )
    # Strip EOS so model continues generating after prefix
    if prefix_ids[-1] == self.tokenizer.eos_token_id:
        prefix_ids = prefix_ids[:-1]
    agent_data.prompt_ids += prefix_ids
    agent_data.response_mask += [1] * len(prefix_ids)  # counts as model output
```

This forces the model to complete "Exact Answer: <actual answer>" instead of calling more tools.

### First attempt issue
Initial prefix was just `"Exact Answer: "` — the model completed it with the template placeholder `"<your answer>"` from the system prompt. Fixed by adding a `<think>` reasoning block before it to prime the model to recall its research findings.

## Efficiency-Scaled Reward Function

Added `_compute_efficiency()` to reward shorter, more efficient trajectories:

```
Correct + efficient   → up to 1.2  (base 1.0 + efficiency bonus 0.2)
Correct + inefficient → down to 0.8
Wrong + efficient     → 0.3
Wrong + inefficient   → down to 0.1
No answer             → 0.0
```

Efficiency = 0.5 × turn_efficiency + 0.5 × length_efficiency (both linear decay to 0 at ceiling).

## Budget Warning Messages

Two-stage system in `_session_state.py`:
1. **SOFT_WARNING_AFTER** (200 tool calls): Append warning to every tool response urging the model to wrap up
2. **MAX_TOOL_CALLS** (300): Hard cutoff — refuse all tool calls, return budget exhausted message → triggers forced answer prefix

## Version Progression

| Version | max_resp | model_len | max_num_seqs | turns | Budget | Answer% | Correct% | Notes |
|---------|----------|-----------|-------------|-------|--------|---------|----------|-------|
| v18 (prior) | 126976 | 131072 | 256 | 500 | None | 46.2%* | 30.8%* | No budget system |
| v19 | 258048 | 262144 | 256 | 500 | 200/300 | OOM | - | 256K too large |
| v19b | 258048 | 262144 | 32 | 500 | 200/300 | OOM | - | Still OOM at warmup |
| v20 | 192512 | 196608 | 32 | 500 | 200/300 | 1/1* | - | Way too slow (1 result in 25min) |
| v21 | 126976 | 131072 | 256 | 500 | 200/300 | 40%* | 30%* | Budget broken (instance_id bug) |
| v22 | 126976 | 131072 | 256 | 200 | 100/150 | 7.3% | 1.3% | Budget too tight |
| v23 | 126976 | 131072 | 256 | 300 | 150/250 | 9.4% | 2.4% | Budget still too tight |
| v24 | 159744 | 163840 | 128 | 500 | 200/300 | 100%* | 67%* | Too slow (3 results only) |
| v25 | 126976 | 131072 | 256 | 500 | 200/300 (fixed) | 100%* | 67%* | Budget fix working, 6 results |
| v25-dbg | 126976 | 131072 | 256 | 500 | 200/300 (fixed) | 100% | 0% | Forced prefix gave "<your answer>" |
| v25b-dbg | 126976 | 131072 | 256 | 500 | 200/300 (fixed+think) | pending | - | Improved prefix with <think> block |

*partial results

## Key Insights

1. **Token budget is the #1 bottleneck**: `max_response_length` must cover ALL tool responses + model tokens. SFT model needs 30K-120K+ tokens for a full trajectory.

2. **Budget tracking must use trajectory-level ID**: verl creates fresh instance_ids per tool call. Use `agent_data.request_id` for persistent per-trajectory state.

3. **Forced prefix > text instructions**: When budget is exhausted, telling the model "please answer now" doesn't work — it keeps calling tools. Injecting answer prefix tokens directly into the generation forces completion.

4. **Context size vs speed tradeoff**: 160K+ context is too slow with TP=4 (only 2 vLLM server groups). 128K is the practical maximum for reasonable iteration speed.

5. **GRPO n=2 is too low**: Standard GRPO uses n=8+ for reliable advantage estimates. Current n=2 was set for speed but hurts training signal quality.

6. **Inference vs RL gap**: `deploy_agent.py` uses max_rounds=200, NO token limit, untruncated tool responses. RL has a fixed `max_response_length` budget shared between model and tools — fundamentally harder.

## Files Modified

- `verl_rl/run_grpo_training.sh` — Parameter tuning across many versions
- `verl_rl/reward/openresearcher_reward.py` — Efficiency-scaled reward, increased debug log limit to 500
- `verl_rl/tools/_session_state.py` — Two-stage budget (soft warning + hard cutoff), fixed messaging
- `verl_rl/tools/browser_search_tool.py` — Budget tracking with traj_id, soft warning injection
- `verl_rl/tools/browser_open_tool.py` — Budget tracking with traj_id, soft warning injection
- `verl_rl/tools/browser_find_tool.py` — Budget tracking with traj_id, soft warning injection
- `/efs/tuhq/codes/verl/verl/experimental/agent_loop/tool_agent_loop.py` — Forced answer prefix injection on budget exhaustion

## Current Status

v25b debug run (4 prompts, n=2 = 8 rollouts) in progress with:
- 128K context, max_num_seqs=256
- Fixed budget tracking (traj_id)
- Forced answer prefix with `<think>` reasoning block
- Waiting for results to verify the prefix produces real answers (not template placeholders)

## Next Steps

1. Verify v25b debug produces real answers (not "<your answer>" placeholders)
2. If working, launch full training run
3. Consider increasing `n` from 2 to 4-8 for better GRPO advantage estimation
4. Consider adding `HF_HOME` to run script permanently (was needed for model download permissions)
