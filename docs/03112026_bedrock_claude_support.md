# Bedrock Claude Support - Debugging & Modifications

**Date:** 2026-03-11
**Model:** `global.anthropic.claude-sonnet-4-6` via AWS Bedrock
**Dataset:** `OpenResearcher/OpenResearcher-Dataset@seed_42:train`
**Search Backend:** Serper

## Overview

This document records the debugging process and code modifications required to run the OpenResearcher pipeline with AWS Bedrock Claude models and the Serper search engine.

## Run Command

```bash
bash run_agent.sh results/bedrock_test bedrock 1 \
  "OpenResearcher/OpenResearcher-Dataset@seed_42:train" \
  serper "global.anthropic.claude-sonnet-4-6"
```

The `run_agent.sh` script detects `BASE_PORT=bedrock` and launches in Bedrock mode with `--use_bedrock` flag.

---

## Bug 1: Tool Name Validation Error

### Symptom

All requests failed immediately on the first Bedrock API call:

```
botocore.errorfactory.ValidationException: An error occurred (ValidationException)
when calling the InvokeModel operation:
tools.0.custom.name: String should match pattern '^[a-zA-Z0-9_-]{1,128}$'
```

### Root Cause

The tool definitions in `TOOL_CONTENT` (defined in `data_utils.py`) use dotted names: `browser.search`, `browser.open`, `browser.find`. The Bedrock Anthropic API requires tool names to match the regex `^[a-zA-Z0-9_-]{1,128}$` — dots are not allowed.

### Fix (in `utils/bedrock_generator.py`)

Three methods were modified to translate tool names at the Bedrock API boundary:

1. **`_convert_tools_to_anthropic()`** — Replace dots with underscores when sending tool definitions:
   ```python
   name = func["name"].replace(".", "_")  # "browser.search" -> "browser_search"
   ```

2. **`_convert_messages_to_anthropic()`** — Replace dots with underscores in assistant tool_use blocks sent back to Bedrock:
   ```python
   bedrock_name = function_name.replace(".", "_")
   ```

3. **`_convert_response_to_openai()`** — Convert underscored names back to dotted format in Bedrock responses, so the rest of the pipeline works unchanged:
   ```python
   if raw_name.startswith("browser_"):
       raw_name = "browser." + raw_name[len("browser_"):]
   ```

This approach keeps the translation entirely within the Bedrock generator — no changes needed to `deploy_agent.py` or `data_utils.py` tool definitions.

---

## Bug 2: Assistant Message Prefill Error

### Symptom

Some questions (e.g., qid=5, qid=9) failed after several successful rounds:

```
botocore.errorfactory.ValidationException: An error occurred (ValidationException)
when calling the InvokeModel operation:
This model does not support assistant message prefill.
The conversation must end with a user message.
```

### Root Cause

When the model responded with text only (no tool calls) and the text did not trigger any answer termination check (no `<answer>`, `Exact Answer:`, etc.), the loop continued to the next round. At that point, the conversation history ended with an assistant message. Bedrock Claude rejects this — it requires the conversation to always end with a user message.

### Fix (two parts)

**Part A — Claude-specific system prompt** (`data_utils.py` + `deploy_agent.py`):

A new system prompt `DEVELOPER_CONTENT_CLAUDE` was created in `data_utils.py` that explicitly instructs the model to format its final answer as:

```
Explanation: {explanation with citations}
Exact Answer: {succinct answer}
Confidence: {0%-100%}
```

In `deploy_agent.py` `run_one_native()`, the system prompt is now selected based on model type:

```python
if 'bedrock' in generator.__class__.__name__.lower() or \
   (hasattr(generator, 'model_id') and 'claude' in getattr(generator, 'model_id', '').lower()):
    base_prompt = DEVELOPER_CONTENT_CLAUDE
else:
    base_prompt = DEVELOPER_CONTENT
```

This ensures Claude always produces output that triggers the `"exact answer:" in content_lower and "confidence:" in content_lower` termination check, avoiding the scenario where the model outputs text without a recognized answer format.

**Part B — Consecutive same-role message merging** (`utils/bedrock_generator.py`):

Added a merging step at the end of `_convert_messages_to_anthropic()` to handle any edge case where consecutive same-role messages appear (Anthropic API requires strict alternating user/assistant turns):

```python
merged = []
for msg in anthropic_messages:
    if merged and merged[-1]["role"] == msg["role"]:
        merged[-1]["content"].extend(msg["content"])
    else:
        merged.append(msg)
return system_prompt, merged
```

**Part C — Graceful termination on unexpected state** (`deploy_agent.py`):

When no tool calls and no answer is detected, the conversation now ends gracefully with a `break` instead of continuing into an invalid state:

```python
# No tool calls and no answer detected — this is an unexpected state
print(f"[qid={qid}] Round {round_num}: No tool calls, no answer detected — ending conversation", flush=True)
break
```

---

## Improvement: Error Logging

### Change (in `deploy_agent.py` `process_item` / writer loop)

Failed trajectories are no longer written to the main shard file. Instead:

- **Success trajectories** -> `node_{rank}_shard_{idx}.jsonl` (as before)
- **Failed trajectories** -> `error_log_node_{rank}_shard_{idx}.jsonl` (new)

Error log entries contain:
```json
{
  "qid": 5,
  "error": "traceback...",
  "attempts": 5,
  "timestamp": "2026-03-11T19:30:00.000000"
}
```

This keeps the main shard clean for downstream evaluation.

---

## Improvement: Debug Logging

### Change (in `deploy_agent.py` `run_one_native()`)

Added `print()` (not `vprint()`) statements at key points so progress is always visible regardless of `--verbose` flag:

- **Round start**: `[qid=X] Round N/200 (Native API) | msgs_so_far=M`
- **Model response**: `[qid=X] Round N MODEL RESPONSE: content_len=L, tool_calls=C` + content preview
- **Tool calls**: `[qid=X] Round N TOOL_CALL[i]: function_name(args_preview)`
- **Tool results**: `[qid=X] Round N TOOL_RESULT[i]: len=L, preview=...`
- **Tool errors**: `[qid=X] Round N TOOL_ERROR[i]: error_msg`
- **Termination**: `[qid=X] Round N: Found <answer> tag - DONE`
- **Completion**: `[qid=X] Finished after N rounds, total messages=M`

---

## Files Modified

| File | Changes |
|------|---------|
| `utils/bedrock_generator.py` | Tool name dot-to-underscore translation; consecutive same-role message merging |
| `deploy_agent.py` | Claude-specific system prompt selection; debug logging; error log separation; graceful termination on no-tool-no-answer state |
| `data_utils.py` | Added `DEVELOPER_CONTENT_CLAUDE` system prompt (done by user) |

## Validation

After all fixes, ran the pipeline with 8 concurrent requests. First 22 questions all completed successfully (`status=success`) with 0 errors. Previously failing qid=5 and qid=9 both succeeded.
