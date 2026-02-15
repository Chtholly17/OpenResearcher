#!/usr/bin/env bash
# Batch eval: 8 rollout dirs under results/OR_dataset, same QA.
# Usage:
#   ./run_eval_bedrock_batch.sh              # evaluate all samples
#   ./run_eval_bedrock_batch.sh 10           # evaluate first 10 samples (test)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCRIPT_DIR}/results/OR_dataset"
MAX_SAMPLES="${1:-}"

INPUT_DIRS=(
  "${BASE}/OpenResearcher_serper_0"
  "${BASE}/OpenResearcher_serper_1"
  "${BASE}/OpenResearcher_serper_2"
  "${BASE}/OpenResearcher_serper_3"
  "${BASE}/OpenResearcher_serper_4"
  "${BASE}/OpenResearcher_serper_5"
  "${BASE}/OpenResearcher_serper_6"
  "${BASE}/OpenResearcher_serper_7"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  python "${SCRIPT_DIR}/eval_bedrock_batch.py" \
    --input_dirs "${INPUT_DIRS[@]}" \
    --max_samples "$MAX_SAMPLES"
else
  python "${SCRIPT_DIR}/eval_bedrock_batch.py" \
    --input_dirs "${INPUT_DIRS[@]}"
fi
