# 02/17/2026 - Chat Template Debugging for Nemotron RL Training

## Problem

Loading the SFT-ed model `OpenResearcher/OpenResearcher-30B-A3B` into multi-turn RL (GRPO via verl), the initial validation shows:
- reward/mean = 0.0, acc/mean = 0.0
- **num_turns = 2 (min=max=mean)** — meaning every trajectory is 1 prompt + 1 assistant generation, then terminated. No tool calls are ever executed.

## Root Cause #1: Tool Call Format Mismatch (FIXED)

verl's built-in `HermesToolParser` (`verl/experimental/agent_loop/tool_parser.py:97`) expects **JSON** inside `<tool_call>` tags:
```
<tool_call>{"name": "browser.search", "arguments": {"query": "..."}}</tool_call>
```

But the Nemotron model's chat template generates **XML-style** tool calls:
```xml
<tool_call>
<function=browser.search>
<parameter=query>
some query
</parameter>
</function>
</tool_call>
```

The `json.loads()` fails silently (caught by `except`), dropping all tool calls.

### Fix Applied

Created a custom `NemotronToolParser` registered as `"nemotron"` in verl's `ToolParser` registry, without modifying the verl package:

- **`verl_rl/parsers/__init__.py`** — empty package init
- **`verl_rl/parsers/nemotron_tool_parser.py`** — custom parser that:
  - Regex-extracts `<tool_call>...</tool_call>` blocks
  - Parses `<function=name><parameter=key>value</parameter></function>` XML format
  - Converts to `FunctionCall` objects with JSON-encoded arguments
  - Registers itself via `@ToolParser.register("nemotron")`
- **`verl_rl/__init__.py`** — imports the parser module as a side-effect, so it registers before `ToolParser.get_tool_parser()` is called (verl loads `verl_rl.tools.*` from config, which triggers `verl_rl/__init__.py`)
- **`verl_rl/config/openresearcher_multiturn_grpo.yaml`** — changed `format: hermes` to `format: nemotron`

### Verification

- Unit tests pass: the parser correctly extracts function names and arguments from actual model output.
- Registration chain confirmed: importing `verl_rl.tools.browser_search_tool` (as verl does) triggers `verl_rl/__init__.py` which adds `"nemotron"` to `ToolParser._registry`.
- The RL training config dump shows `'format': 'nemotron'` and workers start without errors.

## Confirmed: Model Generates Correct Tool Calls

Ran standalone vLLM generation with the model. For "What is the population of Tokyo?", the model:
1. Generates `<think>...</think>` reasoning (763 chars, ~150 tokens)
2. Produces a proper `<tool_call>` with `browser.search`
3. Total output: 232 tokens, finish_reason=stop (hits `<|im_end|>`)
4. The nemotron parser correctly extracts the tool call from this output

## Unresolved Issue: num_turns Still = 2

Despite the parser fix and format config change, RL validation still shows `num_turns=2`. This means the tool parser's `extract_tool_calls()` is either:
1. **Never being called** — some termination condition fires before line 262 of `tool_agent_loop.py`
2. **Called but returns empty** — the decoded text from `response_ids` doesn't contain `<tool_call>` tags for some reason

Debug prints (`print(..., file=sys.stderr, flush=True)`) were added to the parser but the training OOM'd before we could capture them.

### Possible Explanations (to investigate tomorrow)

1. **vLLM server returns different token IDs than standalone vLLM** — the generation through verl's `server_manager.generate()` might truncate, pad, or modify the output tokens.

2. **EOS token handling** — `eos_token_id=11` (`<|im_end|>`). The model generates `<think>...</think><tool_call>...</tool_call><|im_end|>`. If vLLM strips the EOS from `output.token_ids`, the decoded text should still have `<tool_call>`. But if something else is wrong with the decoding...

3. **`response_length` check** — `len(agent_data.response_mask) >= self.response_length` (4096). After 232 tokens this shouldn't fire. But verify this with logging.

4. **max_model_len vs response_length** — `max_model_len=4096` is the total context window. With prompt ~950 tokens, max new tokens = ~3146. But verl might compute `max_tokens` differently for the vLLM generate call, potentially limiting generation.

## Issue #2: OOM During Training

After validation, the first training step OOM'd:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 620.00 MiB.
GPU 0 has a total capacity of 79.25 GiB of which 521.38 MiB is free.
```

This happens during the optimizer step (Adam `exp_avg_sq_sqrt`). The 30B-A3B MoE model with TP=4 uses ~29GB per GPU for model weights, plus vLLM KV cache at `gpu_memory_utilization=0.5`. When FSDP tries to run the optimizer with `optimizer_offload=True` but `param_offload=False`, GPU 0 runs out of memory.

### Possible fixes
- Increase `gpu_memory_utilization` to give more room to training (but less to vLLM)
- Enable `param_offload=True` for the actor (CPU offload model params during training)
- Reduce `ppo_micro_batch_size_per_gpu` from 1 to... well it's already 1
- Reduce `grpo_n_generations` from 2 to 1 (less memory for rollout storage)
- Use `max_model_len` smaller than 4096 for rollout

## Files Changed

| File | Status | Description |
|------|--------|-------------|
| `verl_rl/__init__.py` | Modified | Imports nemotron parser for registration |
| `verl_rl/parsers/__init__.py` | New | Package init |
| `verl_rl/parsers/nemotron_tool_parser.py` | New | Custom XML-format tool parser (has debug prints) |
| `verl_rl/config/openresearcher_multiturn_grpo.yaml` | Modified | `format: hermes` → `format: nemotron` |
| `verl_rl/debug_generation.py` | New | Standalone vLLM generation test script (can delete) |

## TODO for Next Session

1. **Debug the num_turns=2 issue**: Run training with debug prints in the parser. The prints go to `sys.stderr` so check `worker*.err` files in `/tmp/ray/session_latest/logs/`. If the parser is never called, the issue is in verl's agent loop before tool extraction. If called with empty tool calls, the issue is in how vLLM returns tokens to verl.

2. **Alternative debug approach**: Write a minimal script that simulates what the agent loop does — load tokenizer, apply chat template, generate via vLLM HTTP server (like verl's `server_manager.generate()`), then decode and parse. This isolates whether the issue is in generation or parsing.

3. **Fix OOM**: Address memory pressure. Most likely need `param_offload=True` for actor, or lower `gpu_memory_utilization` from 0.5 to 0.4.

4. **Remove debug prints** from `nemotron_tool_parser.py` once the issue is resolved.

5. **Consider `enable_thinking`**: The model spends ~150 tokens on `<think>` per turn. With 30 max turns and 4096 response budget, that's ~4500 tokens just for thinking. May want to set `apply_chat_template_kwargs: {enable_thinking: false}` in the data config, or increase `max_response_length`.
