# Agent Output Analysis Script

This script analyzes deepresearch agent outputs to provide insights into:
1. **Token distribution** in model-generated content (excluding tool results)
2. **Tool call distribution** across all responses

## Features

- ✅ Extracts only assistant-generated content (content + reasoning_content)
- ✅ Excludes tool call results from token analysis
- ✅ Uses the model's actual tokenizer for accurate token counting
- ✅ Generates publication-quality bar plots
- ✅ Handles large JSONL files efficiently
- ✅ Provides detailed statistics

## Installation

Install required dependencies:

```bash
pip install -r requirements_analysis.txt
```

## Usage

### Quick Start (Using Shell Script)

```bash
# Use default settings
./run_analysis.sh

# Or specify custom paths
./run_analysis.sh <input_file> <model_name> <output_dir> <top_k>
```

**Example:**
```bash
./run_analysis.sh \
    /fsx-shared/juncheng/OpenResearcher/results/test/sample.jsonl \
    OpenResearcher/OpenResearcher-30B-A3B \
    /fsx-shared/juncheng/OpenResearcher/results/test/analysis \
    50
```

### Direct Python Usage

```bash
python analyze_agent_outputs.py \
    --input /fsx-shared/juncheng/OpenResearcher/results/test/sample.jsonl \
    --model OpenResearcher/OpenResearcher-30B-A3B \
    --top_k 50 \
    --output_dir ./results/analysis
```

### Arguments

- `--input`: Path to input JSONL file (required)
- `--model`: Model name or path for tokenizer (default: `OpenResearcher/OpenResearcher-30B-A3B`)
- `--top_k`: Number of top tokens to display in plot (default: 50)
- `--output_dir`: Output directory for plots (default: current directory)

## Output

The script generates:

1. **`token_distribution.png`**: Bar chart showing the top K most frequent tokens
   - Only includes assistant-generated content (not tool results)
   - Tokens are decoded using the model's tokenizer
   - Special characters (newlines, tabs, spaces) are displayed visibly

2. **`tool_distribution.png`**: Bar chart showing tool call frequency
   - Shows which tools the agent uses most frequently
   - Helps understand agent behavior patterns

3. **Console statistics**:
   - Total entries analyzed
   - Total tokens generated
   - Unique tokens used
   - Total tool calls
   - Top 10 tokens preview
   - Tool call breakdown

## How It Works

### Token Distribution Analysis

1. **Extracts assistant content**: Iterates through all messages and collects:
   - `content` field from assistant messages
   - `reasoning_content` field from assistant messages
   - **Excludes** tool results (messages with `role: "tool"`)

2. **Tokenizes using model tokenizer**: Uses the actual model's tokenizer to ensure accurate token counting

3. **Counts and visualizes**: Creates frequency distribution and generates bar plot

### Tool Call Distribution Analysis

1. **Extracts tool calls**: Parses `tool_calls` array from assistant messages
2. **Counts function names**: Tallies up which tool functions are called
3. **Visualizes distribution**: Creates bar chart showing tool usage patterns

## Example Output

```
Loading data from: /fsx-shared/juncheng/OpenResearcher/results/test/sample.jsonl
Loaded 10 entries

Loading tokenizer for model: OpenResearcher/OpenResearcher-30B-A3B
Tokenizer loaded successfully

Analyzing token distribution...
Processing entries: 100%|██████████| 10/10
Total tokens analyzed: 45,231

Top 10 tokens:
  '␣the': 2,143
  '␣to': 1,892
  '␣of': 1,567
  '.': 1,234
  '␣and': 1,123
  ...

Analyzing tool call distribution...
Processing entries: 100%|██████████| 10/10
Total tool calls: 156

Tool call distribution:
  browser.search: 89
  browser.open: 45
  browser.scroll: 22

Generating plots...
Token distribution plot saved to: analysis/token_distribution.png
Tool distribution plot saved to: analysis/tool_distribution.png

============================================================
Analysis complete!
============================================================
Token distribution plot: analysis/token_distribution.png
Tool distribution plot: analysis/tool_distribution.png
Total entries analyzed: 10
Total tokens: 45,231
Unique tokens: 8,742
Total tool calls: 156
Unique tools: 3
```

## Notes

- The script correctly distinguishes between model-generated content and tool results
- Special tokens and whitespace are handled properly
- Works with any HuggingFace-compatible tokenizer
- Efficient processing even for large JSONL files
- Plots are saved as high-resolution PNG files (300 DPI)
