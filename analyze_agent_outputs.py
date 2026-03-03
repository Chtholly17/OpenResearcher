#!/usr/bin/env python3
"""
Analyze deepresearch agent outputs:
1. Distribution of total number of tokens per question (model-generated only)
2. Distribution of total number of tokens per question (complete trajectory including tool results)
3. Distribution of total number of tool calls per question
4. Aggregate statistics across multiple rollouts and add to evaluation file

Questions are ranked/sorted by their values for better visualization.
Supports both single file and directory with multiple shard files (node_*_shard_*.jsonl)

Usage:
    # Single file
    python analyze_agent_outputs.py --input /path/to/sample.jsonl --model OpenResearcher/OpenResearcher-30B-A3B

    # Directory with multiple shards
    python analyze_agent_outputs.py --input /path/to/dir/ --model OpenResearcher/OpenResearcher-30B-A3B

    # Aggregate across rollouts
    python analyze_agent_outputs.py --aggregate_rollouts --rollout_base_dir /path/to/OR_dataset --output_file /path/to/evaluated_bedrock_batch.jsonl --model OpenResearcher/OpenResearcher-30B-A3B
"""

import json
import argparse
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import glob
import os
import tempfile
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial


def find_shard_files(input_path):
    """
    Find all shard files matching the pattern node_*_shard_*.jsonl

    Args:
        input_path: Path to directory or file

    Returns:
        list: List of file paths to process
    """
    path = Path(input_path)

    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        # Find all files matching the pattern
        pattern = os.path.join(str(path), "node_*_shard_*.jsonl")
        shard_files = sorted(glob.glob(pattern))

        if not shard_files:
            # Fallback: try to find any .jsonl files
            pattern = os.path.join(str(path), "*.jsonl")
            shard_files = sorted(glob.glob(pattern))

        return shard_files
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def load_jsonl(file_path):
    """Load JSONL file and return list of entries."""
    entries = []
    print(f"  Loading: {os.path.basename(file_path)}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    continue

    print(f"    Loaded {len(entries)} entries")
    return entries


def load_all_data(input_path):
    """
    Load data from single file or multiple shard files.

    Args:
        input_path: Path to file or directory

    Returns:
        list: All entries from all files
    """
    shard_files = find_shard_files(input_path)

    if not shard_files:
        raise ValueError(f"No shard files found in: {input_path}")

    print(f"\nFound {len(shard_files)} file(s) to process")

    all_entries = []
    for file_path in shard_files:
        entries = load_jsonl(file_path)
        all_entries.extend(entries)

    print(f"\nTotal entries loaded: {len(all_entries)}")
    return all_entries


def count_model_tokens_for_question(entry, tokenizer):
    """
    Count total tokens from model-generated content only (excluding tool results).

    Returns:
        int: Total token count for model-generated content
    """
    total_tokens = 0

    for msg in entry.get('messages', []):
        if msg.get('role') == 'assistant':
            # Combine content and reasoning_content
            content = msg.get('content', '') or ''
            reasoning = msg.get('reasoning_content', '') or ''

            # Combine both parts
            full_text = content + '\n' + reasoning
            full_text = full_text.strip()

            if full_text:
                # Tokenize and count
                token_ids = tokenizer.encode(full_text, add_special_tokens=False)
                total_tokens += len(token_ids)

    return total_tokens


def count_all_tokens_for_question(entry, tokenizer):
    """
    Count total tokens for entire answer trajectory (model + tool results).

    Returns:
        int: Total token count for complete trajectory
    """
    total_tokens = 0

    for msg in entry.get('messages', []):
        if msg.get('role') == 'assistant':
            # Model-generated content and reasoning
            content = msg.get('content', '') or ''
            reasoning = msg.get('reasoning_content', '') or ''
            full_text = content + '\n' + reasoning
            full_text = full_text.strip()

            if full_text:
                token_ids = tokenizer.encode(full_text, add_special_tokens=False)
                total_tokens += len(token_ids)

        elif msg.get('role') == 'tool':
            # Tool result content
            tool_content = msg.get('content', '') or ''
            tool_content = tool_content.strip()

            if tool_content:
                token_ids = tokenizer.encode(tool_content, add_special_tokens=False)
                total_tokens += len(token_ids)

    return total_tokens


def count_total_tool_calls_for_question(entry):
    """
    Count total number of tool calls across all assistant responses for one question.

    Returns:
        int: Total tool call count for this question
    """
    total_tool_calls = 0

    for msg in entry.get('messages', []):
        if msg.get('role') == 'assistant':
            # Count tool calls in this message
            tool_calls = msg.get('tool_calls', None)
            if tool_calls:
                total_tool_calls += len(tool_calls)

    return total_tool_calls


def process_entry_worker(entry, tokenizer_path):
    """
    Worker function to process a single entry and count tokens.
    Used for multiprocessing - each worker initializes its own tokenizer from local cache.

    Args:
        entry: JSONL entry containing question and messages
        tokenizer_path: Local path to tokenizer (already cached)

    Returns:
        tuple: (qid, model_tokens, trajectory_tokens)
    """
    # Initialize tokenizer in worker process from local cache
    # Use a global to cache tokenizer per worker process
    global _worker_tokenizer
    if '_worker_tokenizer' not in globals():
        try:
            _worker_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=True  # Only use local cache
            )
        except:
            _worker_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=False,
                trust_remote_code=True,
                local_files_only=True
            )

    qid = entry.get('qid')
    if qid is None:
        return None

    model_tokens = count_model_tokens_for_question(entry, _worker_tokenizer)
    trajectory_tokens = count_all_tokens_for_question(entry, _worker_tokenizer)

    return (qid, model_tokens, trajectory_tokens)


def process_rollout_batch(entries, tokenizer_path, rollout_idx):
    """
    Process a batch of entries from a single rollout using multiprocessing.

    Args:
        entries: List of JSONL entries
        tokenizer_path: Local path to cached tokenizer
        rollout_idx: Rollout index for progress display

    Returns:
        dict: Statistics for each qid
    """
    print(f"  Processing {len(entries)} entries with multiprocessing...")

    # Use partial to fix tokenizer_path argument
    worker_func = partial(process_entry_worker, tokenizer_path=tokenizer_path)

    # Use multiprocessing pool
    num_workers = min(cpu_count(), 32)  # Cap at 32 workers
    print(f"  Using {num_workers} workers")

    qid_stats = {}

    with Pool(num_workers) as pool:
        # Process entries in parallel with progress bar
        results = list(tqdm(
            pool.imap(worker_func, entries, chunksize=10),
            total=len(entries),
            desc=f"Rollout {rollout_idx}"
        ))

    # Collect results
    for result in results:
        if result is None:
            continue
        qid, model_tokens, trajectory_tokens = result

        if qid not in qid_stats:
            qid_stats[qid] = {
                'model_tokens': [],
                'trajectory_tokens': []
            }

        qid_stats[qid]['model_tokens'].append(model_tokens)
        qid_stats[qid]['trajectory_tokens'].append(trajectory_tokens)

    return qid_stats


def analyze_distributions(entries, tokenizer):
    """
    Analyze both token distributions (model-only and complete trajectory).

    Args:
        entries: List of JSONL entries (one per question)
        tokenizer: HuggingFace tokenizer

    Returns:
        tuple: (model_token_counts, all_token_counts, tool_call_counts)
    """
    print(f"\nAnalyzing token distributions per question...")
    model_token_counts = []
    all_token_counts = []
    tool_call_counts = []

    for entry in tqdm(entries, desc="Processing questions"):
        # Count model-generated tokens only
        model_tokens = count_model_tokens_for_question(entry, tokenizer)
        model_token_counts.append(model_tokens)

        # Count all tokens (model + tool results)
        all_tokens = count_all_tokens_for_question(entry, tokenizer)
        all_token_counts.append(all_tokens)

        # Count tool calls
        tool_calls = count_total_tool_calls_for_question(entry)
        tool_call_counts.append(tool_calls)

    print(f"\nTotal questions analyzed: {len(entries)}")

    print(f"\nModel-generated tokens statistics:")
    print(f"  Min: {min(model_token_counts):,}")
    print(f"  Max: {max(model_token_counts):,}")
    print(f"  Mean: {np.mean(model_token_counts):.2f}")
    print(f"  Median: {np.median(model_token_counts):.2f}")
    print(f"  Std Dev: {np.std(model_token_counts):.2f}")

    print(f"\nComplete trajectory tokens statistics (model + tool results):")
    print(f"  Min: {min(all_token_counts):,}")
    print(f"  Max: {max(all_token_counts):,}")
    print(f"  Mean: {np.mean(all_token_counts):.2f}")
    print(f"  Median: {np.median(all_token_counts):.2f}")
    print(f"  Std Dev: {np.std(all_token_counts):.2f}")

    print(f"\nTool call count statistics:")
    print(f"  Min: {min(tool_call_counts)}")
    print(f"  Max: {max(tool_call_counts)}")
    print(f"  Mean: {np.mean(tool_call_counts):.2f}")
    print(f"  Median: {np.median(tool_call_counts):.2f}")
    print(f"  Std Dev: {np.std(tool_call_counts):.2f}")

    return model_token_counts, all_token_counts, tool_call_counts


def plot_histogram(data, title, xlabel, output_path, color='skyblue'):
    """
    Create histogram for distribution data.
    """
    if not data:
        print(f"No data to plot for {title}!")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create histogram
    n, bins, patches = ax.hist(data, bins=50, color=color, edgecolor='navy', alpha=0.7)

    # Customize plot
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Questions)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text box
    stats_text = (
        f'Total Questions: {len(data)}\n'
        f'Mean: {np.mean(data):.1f}\n'
        f'Median: {np.median(data):.1f}\n'
        f'Std Dev: {np.std(data):.1f}\n'
        f'Min: {min(data):,}\n'
        f'Max: {max(data):,}'
    )
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_token_distributions_sorted(model_token_counts, all_token_counts,
                                   output_path='token_distribution_comparison.png'):
    """
    Create side-by-side scatter plots comparing model-only vs complete trajectory tokens.
    X-axis: Token bins (binned by 1000), Y-axis: Exact token counts for each question.
    """
    if not model_token_counts or not all_token_counts:
        print("No token counts to plot!")
        return

    bin_size = 1000

    # Assign bin labels to each data point
    model_bins = [int(x // bin_size) for x in model_token_counts]
    all_bins = [int(x // bin_size) for x in all_token_counts]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Model-generated tokens only
    ax1.scatter(model_bins, model_token_counts, color='skyblue',
                edgecolor='navy', alpha=0.6, s=30)
    ax1.set_xlabel('Token Bin (×1000)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Exact Token Count', fontsize=14, fontweight='bold')
    ax1.set_title('Token Distribution: Model-Generated Only',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='both', alpha=0.3, linestyle='--')

    # Add statistics text box
    stats_text1 = (
        f'Total Questions: {len(model_token_counts)}\n'
        f'Mean: {np.mean(model_token_counts):.1f}\n'
        f'Median: {np.median(model_token_counts):.1f}\n'
        f'Std Dev: {np.std(model_token_counts):.1f}\n'
        f'Min: {min(model_token_counts):,}\n'
        f'Max: {max(model_token_counts):,}'
    )
    ax1.text(0.97, 0.97, stats_text1,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Complete trajectory (model + tool results)
    ax2.scatter(all_bins, all_token_counts, color='lightcoral',
                edgecolor='darkred', alpha=0.6, s=30)
    ax2.set_xlabel('Token Bin (×1000)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Exact Token Count', fontsize=14, fontweight='bold')
    ax2.set_title('Token Distribution: Complete Trajectory (Model + Tool Results)',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(axis='both', alpha=0.3, linestyle='--')

    # Add statistics text box
    stats_text2 = (
        f'Total Questions: {len(all_token_counts)}\n'
        f'Mean: {np.mean(all_token_counts):.1f}\n'
        f'Median: {np.median(all_token_counts):.1f}\n'
        f'Std Dev: {np.std(all_token_counts):.1f}\n'
        f'Min: {min(all_token_counts):,}\n'
        f'Max: {max(all_token_counts):,}'
    )
    ax2.text(0.97, 0.97, stats_text2,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Token distribution comparison plot saved to: {output_path}")
    plt.close()


def plot_tool_distribution_sorted(tool_call_counts, output_path='tool_distribution.png'):
    """
    Create scatter plot for tool call count distribution.
    X-axis: Tool call bins (binned by 10), Y-axis: Exact tool call counts.
    """
    if not tool_call_counts:
        print("No tool call counts to plot!")
        return

    # Use smaller bin size for tool calls (e.g., 10)
    bin_size = 10
    tool_bins = [int(x // bin_size) for x in tool_call_counts]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create scatter plot
    ax.scatter(tool_bins, tool_call_counts, color='coral',
               edgecolor='darkred', alpha=0.6, s=30)

    # Customize plot
    ax.set_xlabel('Tool Call Bin (×10)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Exact Tool Call Count', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Total Tool Call Counts Per Question',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='both', alpha=0.3, linestyle='--')

    # Add statistics text box
    stats_text = (
        f'Total Questions: {len(tool_call_counts)}\n'
        f'Mean: {np.mean(tool_call_counts):.2f}\n'
        f'Median: {np.median(tool_call_counts):.1f}\n'
        f'Std Dev: {np.std(tool_call_counts):.2f}\n'
        f'Min: {min(tool_call_counts)}\n'
        f'Max: {max(tool_call_counts)}'
    )
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tool call distribution plot saved to: {output_path}")
    plt.close()


def aggregate_rollout_statistics(rollout_base_dir, output_file, model_name, num_rollouts=8):
    """
    Aggregate token statistics across multiple rollouts and add to evaluation file.

    Args:
        rollout_base_dir: Base directory containing OpenResearcher_serper_0 through OpenResearcher_serper_7
        output_file: Path to evaluated_bedrock_batch.jsonl file to update (in OpenResearcher_serper_0)
        model_name: Model name for tokenizer initialization
        num_rollouts: Number of rollouts (default: 8)
    """
    print(f"\nAggregating statistics across {num_rollouts} rollouts...")
    print(f"Using model: {model_name}")
    print(f"CPU count: {cpu_count()}")

    # Load and cache tokenizer locally to avoid rate limits
    print(f"\nLoading and caching tokenizer...")
    tokenizer_cache_dir = tempfile.mkdtemp(prefix="tokenizer_cache_")
    print(f"Tokenizer cache directory: {tokenizer_cache_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying with use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

    # Save tokenizer to cache directory
    print(f"Saving tokenizer to local cache...")
    tokenizer.save_pretrained(tokenizer_cache_dir)
    print(f"Tokenizer cached successfully")

    # Dictionary to store token counts by qid: {qid: {model_tokens: [], trajectory_tokens: []}}
    qid_stats = {}

    # Load data from all rollouts
    for rollout_idx in range(num_rollouts):
        rollout_dir = os.path.join(rollout_base_dir, f"OpenResearcher_serper_{rollout_idx}")
        print(f"\n{'='*60}")
        print(f"Processing rollout {rollout_idx}: {rollout_dir}")
        print(f"{'='*60}")

        # Load all shard files for this rollout
        entries = load_all_data(rollout_dir)

        # Process entries with multiprocessing
        rollout_stats = process_rollout_batch(entries, tokenizer_cache_dir, rollout_idx)

        # Merge results into qid_stats
        for qid, stats in rollout_stats.items():
            if qid not in qid_stats:
                qid_stats[qid] = {
                    'model_tokens': [],
                    'trajectory_tokens': []
                }
            qid_stats[qid]['model_tokens'].extend(stats['model_tokens'])
            qid_stats[qid]['trajectory_tokens'].extend(stats['trajectory_tokens'])

    print(f"\nProcessed {len(qid_stats)} unique questions across {num_rollouts} rollouts")

    # Calculate min, mean, max for each question
    print("\nCalculating statistics...")
    aggregated_stats = {}
    for qid, stats in qid_stats.items():
        model_tokens = stats['model_tokens']
        trajectory_tokens = stats['trajectory_tokens']

        aggregated_stats[qid] = {
            'model_response_len': {
                'min': int(min(model_tokens)) if model_tokens else 0,
                'mean': float(np.mean(model_tokens)) if model_tokens else 0.0,
                'max': int(max(model_tokens)) if model_tokens else 0
            },
            'trajectory_len': {
                'min': int(min(trajectory_tokens)) if trajectory_tokens else 0,
                'mean': float(np.mean(trajectory_tokens)) if trajectory_tokens else 0.0,
                'max': int(max(trajectory_tokens)) if trajectory_tokens else 0
            }
        }

    # Load the evaluation file
    print(f"\nLoading evaluation file: {output_file}")
    eval_entries = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    eval_entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue

    print(f"Loaded {len(eval_entries)} evaluation entries")

    # Update evaluation entries with aggregated statistics
    print("\nAdding aggregated statistics to evaluation entries...")
    updated_count = 0
    for entry in eval_entries:
        qid = entry.get('qid')
        if qid in aggregated_stats:
            entry['model_response_len'] = aggregated_stats[qid]['model_response_len']
            entry['trajectory_len'] = aggregated_stats[qid]['trajectory_len']
            updated_count += 1

    print(f"Updated {updated_count} evaluation entries")

    # Write updated evaluation file
    output_path = output_file.replace('.jsonl', '_with_stats.jsonl')
    print(f"\nWriting updated evaluation file: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in eval_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Successfully wrote updated file with {len(eval_entries)} entries")

    # Print some sample statistics
    print("\nSample statistics (first 3 questions):")
    for qid in sorted(aggregated_stats.keys())[:3]:
        stats = aggregated_stats[qid]
        print(f"  QID {qid}:")
        print(f"    Model response: min={stats['model_response_len']['min']}, "
              f"mean={stats['model_response_len']['mean']:.1f}, "
              f"max={stats['model_response_len']['max']}")
        print(f"    Trajectory:     min={stats['trajectory_len']['min']}, "
              f"mean={stats['trajectory_len']['mean']:.1f}, "
              f"max={stats['trajectory_len']['max']}")

    # Cleanup tokenizer cache
    print(f"\nCleaning up tokenizer cache: {tokenizer_cache_dir}")
    shutil.rmtree(tokenizer_cache_dir, ignore_errors=True)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Analyze deepresearch agent outputs per question (supports multiple shard files)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input JSONL file or directory containing node_*_shard_*.jsonl files'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='OpenResearcher/OpenResearcher-30B-A3B',
        help='Model name or path for tokenizer'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for plots (default: current directory)'
    )
    parser.add_argument(
        '--aggregate_rollouts',
        action='store_true',
        help='Aggregate statistics across multiple rollouts'
    )
    parser.add_argument(
        '--rollout_base_dir',
        type=str,
        help='Base directory containing OpenResearcher_serper_0 through OpenResearcher_serper_7 (for --aggregate_rollouts)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Path to evaluated_bedrock_batch.jsonl file to update (for --aggregate_rollouts)'
    )
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=8,
        help='Number of rollouts to aggregate (default: 8)'
    )

    args = parser.parse_args()

    # Handle aggregate mode
    if args.aggregate_rollouts:
        if not args.rollout_base_dir or not args.output_file:
            print("Error: --aggregate_rollouts requires --rollout_base_dir and --output_file")
            return

        output_path = aggregate_rollout_statistics(
            rollout_base_dir=args.rollout_base_dir,
            output_file=args.output_file,
            model_name=args.model,
            num_rollouts=args.num_rollouts
        )

        print("\n" + "="*60)
        print("Aggregation complete!")
        print("="*60)
        print(f"Updated file saved to: {output_path}")
        return

    # Load tokenizer for standard analysis mode
    print(f"\nLoading tokenizer for model: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print(f"Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying with use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    # Standard analysis mode
    if not args.input:
        print("Error: --input is required for standard analysis mode")
        return

    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from single file or multiple shards
    print(f"Loading data from: {args.input}")
    entries = load_all_data(args.input)

    if not entries:
        print("Error: No data loaded!")
        return

    # Analyze distributions
    model_token_counts, all_token_counts, tool_call_counts = analyze_distributions(entries, tokenizer)

    # Generate plots (sorted)
    print("\nGenerating plots...")
    token_comparison_path = output_dir / 'token_distribution_comparison.png'
    tool_plot_path = output_dir / 'tool_distribution.png'

    plot_token_distributions_sorted(model_token_counts, all_token_counts, token_comparison_path)
    plot_tool_distribution_sorted(tool_call_counts, tool_plot_path)

    # Calculate additional insights
    tool_result_tokens = [all_t - model_t for all_t, model_t in zip(all_token_counts, model_token_counts)]

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print(f"Token comparison plot: {token_comparison_path}")
    print(f"Tool call distribution plot: {tool_plot_path}")
    print(f"\nTotal questions analyzed: {len(entries)}")
    print(f"Total model-generated tokens: {sum(model_token_counts):,}")
    print(f"Total trajectory tokens (model + tools): {sum(all_token_counts):,}")
    print(f"Total tool result tokens: {sum(tool_result_tokens):,}")
    print(f"Total tool calls: {sum(tool_call_counts)}")
    print(f"\nAverage tokens from tool results per question: {np.mean(tool_result_tokens):.1f}")
    print(f"Tool result tokens as % of total: {(sum(tool_result_tokens)/sum(all_token_counts)*100):.1f}%")


if __name__ == '__main__':
    main()
