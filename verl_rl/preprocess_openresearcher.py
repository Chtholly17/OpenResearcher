"""
Preprocess OpenResearcher SFT dataset to verl parquet format for multi-turn GRPO training.

Usage:
    # From HuggingFace SFT dataset (96K trajectories) - extracts prompts + ground truth
    python verl_rl/preprocess_openresearcher.py \
        --hf_dataset OpenResearcher/OpenResearcher-Dataset \
        --hf_subset seed_42 \
        --local_save_dir ~/data/openresearcher

    # From evaluation benchmarks (for RL validation / curriculum)
    python verl_rl/preprocess_openresearcher.py \
        --eval_dataset browsecomp \
        --local_save_dir ~/data/openresearcher_eval

    # From local JSONL inference outputs (qid, question, answer fields)
    python verl_rl/preprocess_openresearcher.py \
        --jsonl_path outputs/*.jsonl \
        --local_save_dir ~/data/openresearcher

The output parquet files follow verl's required schema:
    - data_source: str
    - prompt: list[dict] (chat messages with system + user)
    - ability: str
    - reward_model: dict
    - extra_info: dict (with tools_kwargs and interaction_kwargs)
"""

import argparse
import glob
import json
import os
import sys

import datasets

# Add parent directory so we can import data_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_utils import DEVELOPER_CONTENT, load_dataset_unified

DATA_SOURCE = "OpenResearcher/OpenResearcher"


def make_verl_record(qid, question, answer, split, idx):
    """Convert a single (question, answer) pair into verl's required parquet format.

    Key design decisions:
    - System prompt: Use the SAME DEVELOPER_CONTENT that the SFT model was trained on,
      NOT a custom prompt. The RL policy starts from the SFT checkpoint and expects
      the same system prompt format.
    - No research_reward tool in tools_kwargs: The SFT model was never trained to call
      a "research_reward" tool. It uses <answer> tags natively. The trajectory-level
      reward is computed by verl's reward manager using our custom_reward_function
      (configured in the YAML), which extracts <answer> tags from the response.
    - Browser tools don't need per-instance create_kwargs since they connect to a
      shared stateful search service. The tools_kwargs are left empty for them.
    - interaction_kwargs enable the interaction handler to give feedback when the
      agent submits an answer (correct/incorrect), allowing multi-turn refinement.
    """
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {
                "role": "system",
                "content": DEVELOPER_CONTENT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "ability": "deep_research",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "qid": qid,
            "question": question,
            "answer": answer,
            "interaction_kwargs": {
                "name": "openresearcher",
                "query": question,
                "ground_truth": answer,
            },
        },
    }


def load_from_hf_sft(hf_dataset, hf_subset):
    """Load the OpenResearcher SFT dataset from HuggingFace.

    The SFT dataset has full multi-turn trajectories with messages, but for RL
    we only need the (question, answer) pairs as prompts. The RL agent will
    generate its own trajectories through tool interaction.
    """
    print(f"Loading SFT dataset: {hf_dataset} (subset: {hf_subset})")
    ds = datasets.load_dataset(hf_dataset, hf_subset, trust_remote_code=True)

    records = {"train": [], "test": []}
    for split_name in ds:
        split = ds[split_name]
        for item in split:
            records[split_name].append({
                "qid": item.get("qid", item.get("query_id", 0)),
                "question": item["question"],
                "answer": item["answer"],
            })

    print(f"Loaded {sum(len(v) for v in records.values())} records "
          f"({', '.join(f'{k}: {len(v)}' for k, v in records.items())})")
    return records


def load_from_eval_benchmark(dataset_name):
    """Load from OpenResearcher evaluation benchmarks (browsecomp, gaia, etc.)."""
    data = load_dataset_unified(dataset_name)
    # Use all eval data as test split
    records = {
        "test": [{"qid": d["qid"], "question": d["question"], "answer": d["answer"]} for d in data]
    }
    print(f"Loaded {len(records['test'])} records from {dataset_name} (as test split)")
    return records


def load_from_jsonl(jsonl_pattern):
    """Load from local JSONL files (from previous inference runs)."""
    files = sorted(glob.glob(jsonl_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {jsonl_pattern}")

    all_records = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                all_records.append({
                    "qid": item.get("qid", item.get("query_id", 0)),
                    "question": item["question"],
                    "answer": item.get("answer", item.get("correct_answer", "")),
                })

    # 90/10 train/test split
    n_test = max(1, len(all_records) // 10)
    records = {
        "train": all_records[:-n_test],
        "test": all_records[-n_test:],
    }
    print(f"Loaded {len(all_records)} records from {len(files)} JSONL files "
          f"(train: {len(records['train'])}, test: {len(records['test'])})")
    return records


def convert_and_save(records, local_save_dir):
    """Convert records to verl format and save as parquet."""
    os.makedirs(local_save_dir, exist_ok=True)

    verl_records = []
    for split_name, split_records in records.items():
        if not split_records:
            continue

        split_verl_records = []
        for idx, rec in enumerate(split_records):
            split_verl_records.append(
                make_verl_record(
                    qid=rec["qid"],
                    question=rec["question"],
                    answer=rec["answer"],
                    split=split_name,
                    idx=idx,
                )
            )

        # Convert to HuggingFace dataset and save as parquet
        ds = datasets.Dataset.from_list(split_verl_records)
        out_path = os.path.join(local_save_dir, f"{split_name}.parquet")
        ds.to_parquet(out_path)
        print(f"Saved {len(split_verl_records)} records to {out_path}")
        verl_records = split_verl_records  # keep last for sample display

    # Print a sample for verification
    if verl_records:
        sample = verl_records[0]
        print("\n--- Sample record ---")
        print(f"data_source: {sample['data_source']}")
        print(f"prompt[0]: {{role: {sample['prompt'][0]['role']}, content: <{len(sample['prompt'][0]['content'])} chars>}}")
        print(f"prompt[1]: {{role: {sample['prompt'][1]['role']}, content: {sample['prompt'][1]['content'][:100]}...}}")
        print(f"ability: {sample['ability']}")
        print(f"reward_model: {sample['reward_model']}")
        print(f"extra_info keys: {list(sample['extra_info'].keys())}")
        print(f"extra_info.interaction_kwargs: {sample['extra_info']['interaction_kwargs']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess OpenResearcher data to verl parquet format"
    )
    # Data source options (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--hf_dataset",
        type=str,
        help="HuggingFace dataset ID for SFT data (e.g., OpenResearcher/OpenResearcher-Dataset)",
    )
    source.add_argument(
        "--eval_dataset",
        type=str,
        help="Evaluation benchmark name (browsecomp, gaia, hle, seal, xbench, etc.)",
    )
    source.add_argument(
        "--jsonl_path",
        type=str,
        help="Glob pattern for local JSONL files",
    )

    parser.add_argument(
        "--hf_subset",
        type=str,
        default="seed_42",
        help="HuggingFace dataset subset/config (default: seed_42)",
    )
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default="~/data/openresearcher",
        help="Output directory for parquet files",
    )

    args = parser.parse_args()
    save_dir = os.path.expanduser(args.local_save_dir)

    if args.hf_dataset:
        records = load_from_hf_sft(args.hf_dataset, args.hf_subset)
    elif args.eval_dataset:
        records = load_from_eval_benchmark(args.eval_dataset)
    elif args.jsonl_path:
        records = load_from_jsonl(args.jsonl_path)

    convert_and_save(records, save_dir)
    print(f"\nDone. Files saved to: {save_dir}")
