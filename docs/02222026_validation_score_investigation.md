# 02/22/2026 - Validation Score Investigation: Response Budget & Prompt Mismatch

Continuation of 02/18/2026 session. The training pipeline ran stably but initial validation accuracy was ~0.2% — far too low for a pre-trained search agent. This session investigated the root cause.

## Problem Statement

- `val_before_train=True` validation showed **acc=0.197%**, reward=0.00197
- The model was making tool calls (mean 7.87 turns) but almost never submitting `<answer>` tags
- Validation loss was nearly flat — no signal even from a model that should already be competent at search tasks

## Investigation: Three Root Causes Found

### 1. CRITICAL: `max_response_length=2048` — response budget too short

The run script overrode `data.max_response_length` from the YAML default of 8192 down to **2048** (done during OOM debugging on 02/18 as a memory workaround that was never reverted).

In multi-turn mode, `response_length` covers **everything** after the initial prompt:
- `<think>` reasoning tokens (Nemotron defaults to `enable_thinking=True`)
- Tool call XML (`<tool_call><function=...>...</function></tool_call>`)
- Tool response text (mask=0, up to 2048 chars ≈ 500-800 tokens each)
- The model's final `<answer>` submission

**Token budget analysis per turn:**
- Thinking: ~100-300 tokens (model generates `<think>...</think>` before each action)
- Tool call: ~50-100 tokens
- Tool response: ~500-800 tokens (truncated to 2048 chars)
- Total per turn: ~650-1200 tokens

With a 2048 total budget, the model can afford **1-2 tool calls** before the budget is exhausted. It never reaches the answer submission stage.

### 2. HIGH: Tool schema mismatch between SFT and RL

The model sees different tool definitions during RL than during SFT training.

**Root cause**: verl's Pydantic schema (`verl/tools/schemas.py`):
```python
class OpenAIFunctionPropertySchema(BaseModel):
    type: str                          # Only str, not str | list[str]
    description: str | None = None
    enum: list[str] | None = None      # No 'default' field
```

This silently drops fields during `model_validate()`:

| Field | SFT/Inference (TOOL_CONTENT) | RL (tool_config_path via Pydantic) |
|-------|-----|-----|
| `default` values | Present (e.g., `<default>10</default>`) | **Dropped** |
| `browser.open.id` type | `["integer", "string"]` → `<type>["integer", "string"]</type>` | **Validation error** (str expected) |
| `browser.open` params | 6 params (incl. `view_source`, `source`) | 4 params |

The Nemotron chat template's `render_extra_keys` Jinja macro renders extra fields like `default` as XML tags. When these are missing, the rendered `<tools>` section differs from what the model was trained on.

### 3. MEDIUM: `max_model_len=8192` too small for increased response budget

With `max_prompt_length=4096` and `max_response_length=6144`, worst-case total is 10240 tokens. vLLM rejects any sequence exceeding `max_model_len`. Typical prompt is ~1300 tokens so typical total is ~7444, but safety margin needed.

## Fixes Applied

### run_grpo_training.sh — Token budget increases

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `data.max_prompt_length` | 2048 | **4096** | Room for system prompt + tool definitions (~950 tokens) + long questions |
| `data.max_response_length` | 2048 | **6144** | Room for ~5-6 tool call rounds with thinking + final answer |
| `max_model_len` | 8192 | **12288** | Must exceed prompt + response total; uses paged attention so KV cache impact is moderate |

### verl/tools/schemas.py — Pydantic schema patch (new monkey patch)

Changed `OpenAIFunctionPropertySchema`:
- `type: str` → `type: str | list[str]` (supports JSON Schema array types)
- Added `model_config = ConfigDict(extra="allow")` to both `PropertySchema` and `ParametersSchema`

This preserves extra fields like `default` through `model_validate()` → `model_dump()`, so they appear in the rendered chat template.

Patch file: `verl_rl/patches/verl_tool_schemas.py`
Target: `site-packages/verl/tools/schemas.py`

### openresearcher_tool_config.yaml — Tool definitions updated to match TOOL_CONTENT

- `browser.search`: Added `default: 10` for `topn`
- `browser.open`: Added `view_source` (boolean), `source` (string) params; changed `id` type to `["integer", "string"]`; added all `default` values (-1)
- `browser.find`: Added `default: -1` for `cursor`

### openresearcher_multiturn_grpo.yaml — Defaults updated

Updated to match run script: `max_prompt_length: 4096`, `max_response_length: 6144`.

## Verification

Ran a comparison script that renders the initial prompt through `tokenizer.apply_chat_template()` using both tool sources:

```
=== Token counts ===
Inference prompt (TOOL_CONTENT):     950 tokens
RL prompt (tool_config_path):        950 tokens
Difference:                          0 tokens

PROMPTS MATCH EXACTLY
```

After the fixes, the model sees **identical** tool definitions during RL and inference.

## Note on `enable_thinking`

The Nemotron chat template defaults to `enable_thinking=True` (line 12 of `chat_template.jinja`):
```jinja
{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}
```

This prepends `<think>\n` to every assistant generation prompt. The model generates reasoning content before acting. This is **correct behavior** — the SFT model was trained with thinking enabled (`deploy_agent.py` also uses the default). With the response budget now at 6144 tokens (3× the previous 2048), there is sufficient room for thinking + multi-turn research + answer submission.

If token budget becomes tight again (e.g., on smaller GPUs), `enable_thinking` can be disabled via:
```bash
data.apply_chat_template_kwargs.enable_thinking=False
```
This changes the generation prompt from `<think>\n` to `<think></think>` (empty thinking, model acts immediately).

## Files Changed (This Session)

| File | Change | Description |
|------|--------|-------------|
| `verl_rl/run_grpo_training.sh` | Modified | `max_prompt_length` 2048→4096, `max_response_length` 2048→6144, `max_model_len` 8192→12288 |
| `verl_rl/config/openresearcher_multiturn_grpo.yaml` | Modified | Synced `max_prompt_length` and `max_response_length` with run script |
| `verl_rl/config/tool_config/openresearcher_tool_config.yaml` | Modified | All tool schemas now match `TOOL_CONTENT` from `data_utils.py` exactly |
| `verl_rl/patches/verl_tool_schemas.py` | **Created** | New monkey patch for `verl/tools/schemas.py` — preserves extra fields and array types |
| `site-packages/verl/tools/schemas.py` | Patched | Applied the above patch to the installed verl 0.7.0 package |
| `../verl/verl/tools/schemas.py` | Patched | Applied same patch to the development verl source |
| `README.md` | Modified | Added patch #3 to the monkey patches section |
| `docs/02182026_training_pipeline_debug.md` | Modified | Added 02/22 fix section, updated monkey patches list and TODOs |

## Key Learnings

1. **`max_response_length` in multi-turn mode covers the entire conversation** — not just the model's final answer, but all thinking tokens, tool calls, tool responses (mask=0), and intermediate assistant messages across all turns. Setting this too low silently kills performance by forcing early termination.

2. **Pydantic schema strictness can silently break prompt alignment.** verl's `OpenAIFunctionPropertySchema` dropped `default` values and rejected array types without any warning. The resulting prompt looked almost right but was measurably different — the chat template's `render_extra_keys` macro rendered nothing where `<default>10</default>` should have been.

3. **Always verify prompt identity between SFT and RL.** A simple `apply_chat_template` comparison script caught the mismatch immediately. This should be a standard check in the training pipeline.

4. **`enable_thinking=True` is the Nemotron default** and was used during SFT. Disabling it for RL would create a distribution shift. The proper fix is to budget enough response tokens for thinking + action, not to disable thinking.
