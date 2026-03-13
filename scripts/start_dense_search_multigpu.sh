#!/bin/bash
# Start dense search service on multiple GPUs
# Usage: ./scripts/start_dense_search_multigpu.sh [gpu_ids] [port]
#
# Examples:
#   ./scripts/start_dense_search_multigpu.sh 0,1,2,3 8000    # 4 GPUs
#   ./scripts/start_dense_search_multigpu.sh 4,5,6,7 8000    # 4 GPUs (different set)
#   ./scripts/start_dense_search_multigpu.sh 0 8000           # 1 GPU
#   ./scripts/start_dense_search_multigpu.sh                  # defaults: GPUs 0,1,2,3, port 8000

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

GPU_IDS_ARG="${1:-0,1,2,3}"
PORT="${2:-8000}"

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Build CUDA_VISIBLE_DEVICES and GPU_IDS
# CUDA_VISIBLE_DEVICES uses the physical GPU IDs
# GPU_IDS uses logical indices 0..N-1 since CUDA_VISIBLE_DEVICES remaps them
export CUDA_VISIBLE_DEVICES="${GPU_IDS_ARG}"

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS_ARG"
NUM_GPUS=${#GPU_ARRAY[@]}
LOGICAL_IDS=""
for (( i=0; i<NUM_GPUS; i++ )); do
    if [ -n "$LOGICAL_IDS" ]; then
        LOGICAL_IDS="${LOGICAL_IDS},$i"
    else
        LOGICAL_IDS="$i"
    fi
done
export GPU_IDS="${LOGICAL_IDS}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Dense Search Service (Multi-GPU)${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Physical GPUs (CUDA_VISIBLE_DEVICES): ${CUDA_VISIBLE_DEVICES}"
echo "Logical GPU IDs (GPU_IDS):            ${GPU_IDS}"
echo "Number of GPUs:                       ${NUM_GPUS}"
echo "Port:                                 ${PORT}"
echo ""

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Please run ./setup.sh first${NC}"
    exit 1
fi

echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate
echo "Using: $(python --version)"
echo ""

# Set environment variables
export SEARCHER_TYPE="dense"
export LUCENE_EXTRA_DIR="${PROJECT_ROOT}/tevatron"
export CORPUS_PARQUET_PATH="${PROJECT_ROOT}/Tevatron/browsecomp-plus-corpus/data/*.parquet"
export DENSE_INDEX_PATH="${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl"
export DENSE_MODEL_NAME="Qwen/Qwen3-Embedding-8B"

echo "CORPUS_PARQUET_PATH: ${CORPUS_PARQUET_PATH}"
echo "DENSE_INDEX_PATH:    ${DENSE_INDEX_PATH}"
echo "DENSE_MODEL_NAME:    ${DENSE_MODEL_NAME}"
echo ""

# Validate prerequisites
if [ ! -f "${LUCENE_EXTRA_DIR}/lucene-highlighter-9.9.1.jar" ]; then
    echo -e "${RED}Error: Lucene JARs not found in ${LUCENE_EXTRA_DIR}${NC}"
    exit 1
fi

CORPUS_COUNT=$(ls ${PROJECT_ROOT}/Tevatron/browsecomp-plus-corpus/data/*.parquet 2>/dev/null | wc -l)
if [ "$CORPUS_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: Corpus parquet files not found${NC}"
    exit 1
fi

INDEX_COUNT=$(ls ${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl 2>/dev/null | wc -l)
if [ "$INDEX_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: Dense index .pkl files not found${NC}"
    exit 1
fi

echo "Corpus files:  ${CORPUS_COUNT}"
echo "Index shards:  ${INDEX_COUNT}"
echo ""
echo -e "${GREEN}Starting uvicorn on port ${PORT}...${NC}"
echo "Press Ctrl+C to stop"
echo ""

uvicorn scripts.deploy_search_service:app --host 0.0.0.0 --port ${PORT}
