# GAIA Trajectory Error Analysis

Automated pipeline for analyzing error modes in GAIA benchmark inference trajectories.

## Overview

The pipeline classifies incorrect agent trajectories into 9 error modes using Bedrock Claude, producing per-model summary tables and detailed per-trajectory classifications.

## Files

| File | Description |
|------|-------------|
| `error_analysis.py` | Main analysis script |
| `error_modes.md` | Error mode definitions and examples |

## Quick Start

```bash
source .venv/bin/activate

# Basic usage (requires evaluated.jsonl or evaluated_bedrock.jsonl in the input dir)
python error_analysis.py --input_dir results/gaia_grpo_qwen3_v0.8_0303

# If no evaluation file exists, the script runs correctness eval automatically
python error_analysis.py --input_dir results/gaia_openresearcher

# Use a stronger model for error classification
python error_analysis.py --input_dir results/gaia_grpo_qwen3_0.5_0.875 \
  --classification_model_id global.anthropic.claude-sonnet-4-20250514-v1:0
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | (required) | Directory containing trajectory JSONL files (`node_*_shard_*.jsonl`) |
| `--classification_model_id` | `global.anthropic.claude-sonnet-4-6` | Bedrock model for error classification |
| `--eval_model_id` | `global.anthropic.claude-sonnet-4-6` | Bedrock model for correctness evaluation (only used when no evaluation file exists) |
| `--region_name` | `us-east-1` | AWS region for Bedrock |
| `--qps` | `10.0` | Queries per second rate limit |
| `--max_workers` | `10` | Thread pool size for parallel API calls |
| `--max_tokens` | `2048` | Max tokens for classification response |

## Pipeline Steps

1. **Load trajectories** from `node_*_shard_*.jsonl` files in `--input_dir`
2. **Load correctness evaluations** from `evaluated.jsonl` or `evaluated_bedrock.jsonl`. If neither exists, runs Bedrock Claude correctness evaluation and saves `evaluated_bedrock.jsonl`
3. **Separate trajectories** into correct / incorrect / error (`status != "success"`)
4. **Classify incorrect trajectories** by sending compressed trajectories to Bedrock Claude, which assigns error mode labels. Error trajectories are automatically labeled E8
5. **Output results** to console (PrettyTable) and files

## Output Files

Both files are written to `--input_dir`:

### `error_analysis_detailed.jsonl`

One JSON object per failed trajectory with consistent key ordering:

```json
{
  "qid": 0,
  "question": "...",
  "correct_answer": "...",
  "agent_answer": "...",
  "error_modes": ["E1", "E2"],
  "primary_error": "E1",
  "search_quality": "poor",
  "explanation": "The agent performed only a single search...",
  "search_queries": ["nonnative species of clownfish USGS zip codes"],
  "num_tool_calls": 1
}
```

### `error_analysis_summary.json`

Aggregated results including error mode definitions and counts:

```json
{
  "input_dir": "results/gaia_grpo_qwen3_v0.8_0303",
  "classification_model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
  "total_questions": 103,
  "correct_count": 10,
  "incorrect_count": 38,
  "error_count": 55,
  "overall_accuracy": 0.0971,
  "error_mode_definitions": {"E1": "Insufficient Search Depth", "...": "..."},
  "error_mode_counts": {"E1": {"name": "Insufficient Search Depth", "count": 35}, "...": "..."},
  "primary_error_distribution": {"E1": {"name": "Insufficient Search Depth", "count": 19}, "...": "..."},
  "search_quality_distribution": {"poor": 38, "N/A": 55}
}
```

## Error Modes

| ID | Error Mode | Description |
|----|-----------|-------------|
| E1 | Insufficient Search Depth | Too few searches before answering; no query refinement |
| E2 | Hallucination from Incomplete Context | Facts in final answer not found in any tool result |
| E3 | Incorrect Search Query | Queries too vague, wrong terminology, or missing key terms |
| E4 | Misinterpretation of Search Results | Correct info in results but model picked wrong value |
| E5 | Reasoning / Logic Error | Arithmetic, logic, or multi-step reasoning mistakes |
| E6 | Unsupported Numerical Estimation | Numbers produced without any supporting search data |
| E7 | Failure to Navigate to Specific Documents | Cannot access required PDFs, GitHub issues, specific pages |
| E8 | Runtime Error / Timeout | Trajectory ended with `status != "success"` |
| E9 | Other / Unclear | Doesn't fit above categories |

Each trajectory can have **multiple** error modes; one is marked as the **primary** cause.

## Changelog

### 2026-03-14

- Created `error_modes.md` with 9 error mode definitions and manual analysis examples
- Created `error_analysis.py` with full pipeline:
  - Reuses `BedrockClaudeJudge` pattern from `eval_bedrock.py` (boto3, retries, ThreadPoolExecutor)
  - Reuses `ThreadRateLimiter` from `eval.py`
  - Reuses `GRADER_TEMPLATE` and `parse_judge_response()` from `eval.py`
  - Trajectory compression: truncates tool results (800 chars), reasoning (500 chars), assistant content (1500 chars)
  - Multi-label classification with primary error selection
  - Handles both `"correct": true/false` and `"judgement": "yes"/"no"` evaluation formats
  - Auto-runs correctness evaluation via Bedrock if no evaluation file exists
- Normalized key ordering in `error_analysis_detailed.jsonl` for consistency
- Added error mode names to `error_analysis_summary.json` alongside counts
- Created `ERROR_ANALYSIS_README.md` (this file)
