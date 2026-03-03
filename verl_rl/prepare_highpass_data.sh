#!/bin/bash
# Prepare high-pass training data for Qwen3 GRPO (v0.8+)
#
# Downloads rejection-sampling evaluation results from HuggingFace, filters to
# questions where the Nemotron model had >= PASS_RATE_MIN pass rate, and writes
# verl-format parquet files with the Qwen3 system prompt embedded.
#
# Source dataset:  Chtholly17/OR_reject_sampling
#   - 6,102 questions, each evaluated by 8 binary judges
#   - pass_rate = fraction of judges that marked the Nemotron answer correct
#   - Pass rate distribution:
#       0.000: 2878 (47%)   ← too hard even for Nemotron
#       0.125:  331  (5%)
#       0.250:  229  (4%)
#       0.375:  203  (3%)
#       0.500:  145  (2%)   ← Goldilocks for Qwen3 mid-difficulty
#       0.625:  175  (3%)
#       0.750:  247  (4%)
#       0.875:  313  (5%)   ← high-pass starts here
#       1.000: 1581 (26%)   ← high-pass (most of these)
#
# Output:
#   data/qwen3_highpass/train.parquet  (default: all except 20 test examples)
#   data/qwen3_highpass/test.parquet   (20 held-out examples)
#
# Usage:
#   # Default: pass_rate >= 0.875 (1,894 examples)
#   bash verl_rl/prepare_highpass_data.sh
#
#   # Strict 100% only (1,581 examples)
#   PASS_RATE_MIN=1.0 bash verl_rl/prepare_highpass_data.sh
#
#   # Mid-difficulty: 0.5–0.8 (567 examples, used in v0.5–v0.7)
#   PASS_RATE_MIN=0.5 PASS_RATE_MAX=0.8 bash verl_rl/prepare_highpass_data.sh \
#     --output-dir data/qwen3_mid
#
#   # Custom output dir
#   OUT_DIR=data/qwen3_custom bash verl_rl/prepare_highpass_data.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/dlami/nvme/hqhd-miniconda3/envs/openresearcher/bin/python}"

# ── Parameters ────────────────────────────────────────────────────────────────
PASS_RATE_MIN="${PASS_RATE_MIN:-0.875}"   # minimum Nemotron pass rate (inclusive)
PASS_RATE_MAX="${PASS_RATE_MAX:-1.0}"     # maximum Nemotron pass rate (inclusive)
N_TEST="${N_TEST:-20}"                    # number of held-out test examples
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/data/qwen3_highpass}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Parse --output-dir flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "================================================"
echo "Preparing Qwen3 GRPO training data"
echo "================================================"
echo "Source:       Chtholly17/OR_reject_sampling"
echo "Pass rate:    [$PASS_RATE_MIN, $PASS_RATE_MAX]"
echo "Test split:   $N_TEST examples"
echo "Output dir:   $OUT_DIR"
echo "================================================"

"$PYTHON" - << PYEOF
import sys, os, random, json
sys.path.insert(0, '$PROJECT_DIR')
sys.path.insert(0, '$PROJECT_DIR/verl_rl')

os.environ['HF_HOME'] = '$HF_HOME'

from datasets import load_dataset, Dataset
from preprocess_openresearcher import make_verl_record

PASS_RATE_MIN = float('$PASS_RATE_MIN')
PASS_RATE_MAX = float('$PASS_RATE_MAX')
N_TEST = int('$N_TEST')
SEED = int('$SEED')
OUT_DIR = '$OUT_DIR'

random.seed(SEED)

# ── Download ──────────────────────────────────────────────────────────────────
print("Downloading Chtholly17/OR_reject_sampling ...")
ds = load_dataset('Chtholly17/OR_reject_sampling', split='train')
print(f"  Total examples: {len(ds)}")

# ── Filter ────────────────────────────────────────────────────────────────────
filtered = [r for r in ds if PASS_RATE_MIN <= r['pass_rate'] <= PASS_RATE_MAX]
print(f"\nFiltered to pass_rate [{PASS_RATE_MIN}, {PASS_RATE_MAX}]: {len(filtered)} examples")

# Show breakdown
from collections import Counter
breakdown = Counter(round(r['pass_rate'], 3) for r in filtered)
for rate in sorted(breakdown):
    print(f"  pass_rate={rate:.3f}: {breakdown[rate]:>5}")

# ── Split ─────────────────────────────────────────────────────────────────────
random.shuffle(filtered)
if len(filtered) < N_TEST + 1:
    raise ValueError(f"Not enough examples ({len(filtered)}) for {N_TEST} test + at least 1 train")

test_set  = filtered[:N_TEST]
train_set = filtered[N_TEST:]
print(f"\nSplit: train={len(train_set)}, test={N_TEST}")

# ── Convert to verl format ────────────────────────────────────────────────────
def to_records(rows, split):
    return [
        make_verl_record(
            qid=str(r['qid']),
            question=r['question'],
            answer=r['correct_answer'],
            split=split,
            idx=i,
            model_type='qwen3',
        )
        for i, r in enumerate(rows)
    ]

print("\nConverting to verl parquet format (model_type=qwen3) ...")
train_records = to_records(train_set, 'train')
test_records  = to_records(test_set,  'test')

# ── Verify system prompt ──────────────────────────────────────────────────────
sys_prompt = train_records[0]['prompt'][0]['content']
if 'MUST' not in sys_prompt:
    raise RuntimeError("System prompt missing 'MUST' keyword — update DEVELOPER_CONTENT_QWEN3 in data_utils.py")
print(f"System prompt check: 'MUST use browser.search' present ✓")
print(f"  Preview: {sys_prompt[:120].replace(chr(10), ' ')}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

train_path = os.path.join(OUT_DIR, 'train.parquet')
test_path  = os.path.join(OUT_DIR, 'test.parquet')

Dataset.from_list(train_records).to_parquet(train_path)
Dataset.from_list(test_records).to_parquet(test_path)

print(f"\nSaved:")
print(f"  {train_path}  ({len(train_records)} rows)")
print(f"  {test_path}  ({len(test_records)} rows)")

# ── Sample record ─────────────────────────────────────────────────────────────
sample = train_set[0]
print(f"\nSample record:")
print(f"  qid:        {sample['qid']}")
print(f"  pass_rate:  {sample['pass_rate']}")
print(f"  question:   {sample['question'][:100]}")
print(f"  answer:     {sample['correct_answer'][:80]}")

print("\nDone.")
PYEOF

echo ""
echo "Data ready at: $OUT_DIR"
echo "  Launch training with:"
echo "    TRAIN_DATA=$OUT_DIR/train.parquet \\"
echo "    VAL_DATA=$OUT_DIR/test.parquet \\"
echo "    bash verl_rl/run_grpo_training_qwen3_v0.8.sh"
