#!/bin/bash
# Run analysis on agent outputs
# Supports both single file and directory with multiple shard files

# Set default values
INPUT_PATH="${1:-/fsx-shared/juncheng/OpenResearcher/results/test/sample.jsonl}"
MODEL="${2:-OpenResearcher/OpenResearcher-30B-A3B}"
OUTPUT_DIR="${3:-/fsx-shared/juncheng/OpenResearcher/results/test/analysis}"

echo "=========================================="
echo "Agent Output Analysis"
echo "=========================================="
echo "Input path: $INPUT_PATH"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run analysis
python /fsx-shared/juncheng/OpenResearcher/analyze_agent_outputs.py \
    --input "$INPUT_PATH" \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Analysis complete! Check outputs in:"
echo "$OUTPUT_DIR"
echo "=========================================="
