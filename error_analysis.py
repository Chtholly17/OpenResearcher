#!/usr/bin/env python3
"""
GAIA Trajectory Error Analysis Pipeline

Analyzes error modes in GAIA benchmark inference trajectories.
1. Loads trajectories and correctness evaluations
2. Classifies incorrect trajectories into error modes using Bedrock Claude
3. Produces per-model summary tables and detailed output files

Usage:
    # With existing evaluated.jsonl
    python error_analysis.py --input_dir results/gaia_grpo_qwen3_v0.8_0303

    # Without evaluated.jsonl (runs correctness eval first)
    python error_analysis.py --input_dir results/gaia_openresearcher

    # Use stronger model for classification
    python error_analysis.py --input_dir results/gaia_grpo_qwen3_0.5_0.875 \
      --classification_model_id global.anthropic.claude-sonnet-4-20250514-v1:0
"""

import argparse
import glob
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import dotenv
from botocore.exceptions import ClientError
from prettytable import PrettyTable
from tqdm import tqdm

from eval import GRADER_TEMPLATE, ThreadRateLimiter, parse_judge_response

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Error classification prompt
# ---------------------------------------------------------------------------

ERROR_MODES = {
    "E1": "Insufficient Search Depth",
    "E2": "Hallucination from Incomplete Context",
    "E3": "Incorrect Search Query",
    "E4": "Misinterpretation of Search Results",
    "E5": "Reasoning / Logic Error",
    "E6": "Unsupported Numerical Estimation",
    "E7": "Failure to Navigate to Specific Documents",
    "E8": "Runtime Error / Timeout",
    "E9": "Other / Unclear",
}

ERROR_CLASSIFICATION_PROMPT = """You are an expert analyst reviewing a research agent's trajectory on the GAIA benchmark. The agent uses browser tools (browser.search, browser.open, browser.find) to answer questions. This trajectory produced an INCORRECT answer.

## Error Modes

Classify the trajectory into one or more of these error modes:

- **E1: Insufficient Search Depth** - Too few searches (typically 1) before answering; no query refinement or follow-up. Look for: only 1-2 tool calls total, no iterative search refinement.
- **E2: Hallucination from Incomplete Context** - Specific facts/numbers in the final answer are NOT found in any tool result. Look for: claims in the answer that don't appear in search results.
- **E3: Incorrect Search Query** - Queries are too vague, use wrong terminology, or miss key terms. Look for: searches that don't target the specific information needed.
- **E4: Misinterpretation of Search Results** - Correct info was in search results but model picked wrong value/entity. Look for: the right answer appearing in results but being ignored or confused.
- **E5: Reasoning / Logic Error** - Arithmetic, logic, or multi-step reasoning mistakes despite correct facts. Look for: math errors, wrong logical deductions, incorrect counting.
- **E6: Unsupported Numerical Estimation** - Specific numbers produced without any supporting search data. Look for: numeric answers with no search results backing them.
- **E7: Failure to Navigate to Specific Documents** - Cannot access required PDFs, GitHub issues, specific pages. Look for: questions needing specific resources that the agent never accessed.
- **E9: Other / Unclear** - Doesn't fit above categories.

(Note: E8 is for runtime errors and is handled separately.)

## Task

Analyze the following trajectory and classify the error modes.

**Question:** {question}

**Correct Answer:** {correct_answer}

**Agent's Answer:** {agent_answer}

**Agent Trajectory:**
{trajectory}

## Response Format

You MUST respond in exactly this format:

error_modes: [comma-separated list of applicable error mode IDs, e.g. E1, E2]
primary_error: [single error mode ID that is the main cause of failure]
search_quality: [good/fair/poor - how well did the agent search for information]
explanation: [2-3 sentence explanation of what went wrong and why]
"""

# ---------------------------------------------------------------------------
# Trajectory compression
# ---------------------------------------------------------------------------

def compress_trajectory(messages, max_tool_result_chars=1600, max_reasoning_chars=1600, max_assistant_chars=1600):
    """Compress a trajectory for LLM analysis by truncating long content."""
    compressed = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            compressed.append(f"[SYSTEM] {msg.get('content', '')[:200]}...")
            continue

        if role == "user":
            compressed.append(f"[USER] {msg.get('content', '')}")
            continue

        if role == "assistant":
            parts = []
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                if len(reasoning) > max_reasoning_chars:
                    reasoning = reasoning[:max_reasoning_chars] + "... [truncated]"
                parts.append(f"  Thinking: {reasoning}")

            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if content:
                if len(content) > max_assistant_chars:
                    content = content[:max_assistant_chars] + "... [truncated]"
                parts.append(f"  Response: {content}")

            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    parts.append(f"  Tool Call: {name}({json.dumps(args) if isinstance(args, dict) else args})")

            compressed.append(f"[ASSISTANT]\n" + "\n".join(parts))
            continue

        if role == "tool":
            name = msg.get("name", "tool")
            content = msg.get("content", "") or ""
            if len(content) > max_tool_result_chars:
                content = content[:max_tool_result_chars] + "... [truncated]"
            compressed.append(f"[TOOL: {name}] {content}")
            continue

    return "\n\n".join(compressed)


def extract_search_queries(messages):
    """Extract all search queries from a trajectory."""
    queries = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                if "search" in fn.get("name", ""):
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(args, dict):
                        queries.append(args.get("query", str(args)))
                    else:
                        queries.append(str(args))
    return queries


def count_tool_calls(messages):
    """Count total tool calls in a trajectory."""
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            count += len(msg["tool_calls"])
    return count


def extract_agent_answer(messages):
    """Extract the final answer from the last assistant message."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if content.strip():
                return content.strip()
    return ""


# ---------------------------------------------------------------------------
# Bedrock Claude classifier (reuses BedrockClaudeJudge pattern)
# ---------------------------------------------------------------------------

class BedrockErrorClassifier:
    def __init__(
        self,
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1",
        qps=10,
        max_retries=5,
        max_workers=10,
        max_tokens=2048,
        temperature=0.0,
    ):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.rate_limiter = ThreadRateLimiter(qps)
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.temperature = temperature

    def classify_batch(self, items):
        """Classify a batch of incorrect trajectories."""
        output = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._classify, item): item for item in items}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Classifying errors"):
                output.append(f.result())
        return output

    @staticmethod
    def _extract_model_text(model_response):
        content_blocks = model_response.get("content", [])
        if not isinstance(content_blocks, list):
            return ""
        text_chunks = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text", "")
                if txt:
                    text_chunks.append(txt)
        return "\n".join(text_chunks).strip()

    def _classify(self, item):
        """Classify a single trajectory."""
        messages = item.get("messages", [])
        question = item["question"]
        correct_answer = item["correct_answer"]
        agent_answer = extract_agent_answer(messages)
        trajectory_text = compress_trajectory(messages)
        search_queries = extract_search_queries(messages)
        num_tools = count_tool_calls(messages)

        prompt = ERROR_CLASSIFICATION_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            agent_answer=agent_answer[:1500],
            trajectory=trajectory_text,
        )

        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.acquire()
            try:
                native_request = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                }
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(native_request),
                )
                model_response = json.loads(response["body"].read())
                result_text = self._extract_model_text(model_response)
                parsed = parse_classification_response(result_text)
                parsed["qid"] = item.get("qid")
                parsed["question"] = question
                parsed["correct_answer"] = correct_answer
                parsed["agent_answer"] = agent_answer[:500]
                parsed["search_queries"] = search_queries
                parsed["num_tool_calls"] = num_tools
                parsed["raw_classification"] = result_text
                return parsed
            except (ClientError, Exception) as e:
                if attempt == self.max_retries:
                    return {
                        "qid": item.get("qid"),
                        "question": question,
                        "correct_answer": correct_answer,
                        "agent_answer": agent_answer[:500],
                        "error_modes": ["E9"],
                        "primary_error": "E9",
                        "search_quality": "unknown",
                        "explanation": f"Classification failed: {str(e)}",
                        "search_queries": search_queries,
                        "num_tool_calls": num_tools,
                        "parse_error": True,
                    }
                backoff = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                time.sleep(backoff)


def parse_classification_response(text):
    """Parse the structured classification response."""
    result = {
        "error_modes": [],
        "primary_error": "E9",
        "search_quality": "unknown",
        "explanation": "",
        "parse_error": False,
    }

    if not text:
        result["parse_error"] = True
        return result

    # Extract error_modes
    modes_match = re.search(r"error_modes:\s*\[?\s*(.*?)\s*\]?\s*$", text, re.MULTILINE | re.IGNORECASE)
    if modes_match:
        modes_str = modes_match.group(1)
        modes = re.findall(r"E\d", modes_str)
        result["error_modes"] = sorted(set(modes))

    # Extract primary_error
    primary_match = re.search(r"primary_error:\s*(E\d)", text, re.IGNORECASE)
    if primary_match:
        result["primary_error"] = primary_match.group(1)

    # Extract search_quality
    quality_match = re.search(r"search_quality:\s*(good|fair|poor)", text, re.IGNORECASE)
    if quality_match:
        result["search_quality"] = quality_match.group(1).lower()

    # Extract explanation
    explanation_match = re.search(r"explanation:\s*(.*?)(?:\n\n|$)", text, re.DOTALL | re.IGNORECASE)
    if explanation_match:
        result["explanation"] = explanation_match.group(1).strip()

    if not result["error_modes"]:
        result["error_modes"] = ["E9"]
        result["parse_error"] = True

    # Ensure primary is in error_modes
    if result["primary_error"] not in result["error_modes"]:
        if result["error_modes"]:
            result["primary_error"] = result["error_modes"][0]

    return result


# ---------------------------------------------------------------------------
# Bedrock correctness judge (reused from eval_bedrock.py)
# ---------------------------------------------------------------------------

class BedrockCorrectnessJudge:
    """Simplified version of BedrockClaudeJudge for correctness evaluation."""

    def __init__(
        self,
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1",
        qps=20,
        max_retries=5,
        max_workers=20,
        max_tokens=512,
        temperature=0.0,
    ):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.rate_limiter = ThreadRateLimiter(qps)
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.temperature = temperature

    def judge(self, data):
        output = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._judge, d) for d in data]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluating correctness"):
                output.append(f.result())
        return output

    @staticmethod
    def _extract_gen_output(data):
        content = data["messages"][-1]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list) and len(content) > 0:
            first = content[0]
            if isinstance(first, dict):
                return first.get("text", "")
        return ""

    @staticmethod
    def _extract_model_text(model_response):
        content_blocks = model_response.get("content", [])
        if not isinstance(content_blocks, list):
            return ""
        text_chunks = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text", "")
                if txt:
                    text_chunks.append(txt)
        return "\n".join(text_chunks).strip()

    def _judge(self, data):
        question = data["question"]
        answer = data["answer"]
        gen_output = self._extract_gen_output(data)
        prompt = GRADER_TEMPLATE.format(
            question=question, response=gen_output, correct_answer=answer
        )

        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.acquire()
            try:
                native_request = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                }
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(native_request),
                )
                model_response = json.loads(response["body"].read())
                judge_text = self._extract_model_text(model_response)
                parsed = parse_judge_response(judge_text)
                parsed["qid"] = data.get("qid")
                parsed["question"] = question
                parsed["correct_answer"] = answer
                return parsed
            except (ClientError, Exception) as e:
                if attempt == self.max_retries:
                    return {
                        "qid": data.get("qid"),
                        "question": question,
                        "correct_answer": answer,
                        "correct": False,
                        "parse_error": True,
                        "error": str(e),
                    }
                backoff = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                time.sleep(backoff)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectories(input_dir):
    """Load trajectory data from shard files."""
    pattern = os.path.join(input_dir, "node_*_shard_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(input_dir, "*.jsonl")
        files = sorted(glob.glob(pattern))
        files = [f for f in files if not os.path.basename(f).startswith("evaluated")]
        files = [f for f in files if not os.path.basename(f).startswith("error_analysis")]

    data = []
    for fp in files:
        if os.path.basename(fp).startswith("evaluated") or os.path.basename(fp).startswith("error_analysis"):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    print(f"Loaded {len(data)} trajectories from {len(files)} file(s)")
    return data


def load_evaluations(input_dir):
    """Load existing correctness evaluations. Returns dict qid -> eval_entry or None."""
    # Try evaluated.jsonl first, then evaluated_bedrock.jsonl
    for fname in ["evaluated.jsonl", "evaluated_bedrock.jsonl"]:
        eval_path = os.path.join(input_dir, fname)
        if os.path.exists(eval_path):
            evals = {}
            with open(eval_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            evals[entry["qid"]] = entry
                        except (json.JSONDecodeError, KeyError):
                            continue
            print(f"Loaded {len(evals)} evaluations from {fname}")
            return evals
    return None


def is_correct(eval_entry):
    """Determine correctness from an evaluation entry, handling both formats."""
    # Format 1: "correct": true/false
    if "correct" in eval_entry:
        return eval_entry["correct"] is True
    # Format 2: "judgement": "yes"/"no"
    if "judgement" in eval_entry:
        return eval_entry["judgement"] == "yes"
    return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_correctness_eval(trajectories, input_dir, eval_model_id, region_name):
    """Run correctness evaluation on trajectories that completed successfully."""
    clean_data = [d for d in trajectories if d.get("status") == "success"]
    print(f"\nRunning correctness evaluation on {len(clean_data)} successful trajectories...")

    judge = BedrockCorrectnessJudge(
        model_id=eval_model_id, region_name=region_name
    )
    results = judge.judge(clean_data)

    # Save results
    eval_path = os.path.join(input_dir, "evaluated_bedrock.jsonl")
    evals = {}
    for item in sorted(results, key=lambda x: x.get("qid", 0)):
        correct_flag = item.get("correct")
        saved = {
            "qid": item.get("qid"),
            "question": item.get("question"),
            "correct_answer": item.get("correct_answer"),
            "correct": correct_flag,
        }
        evals[saved["qid"]] = saved
        with open(eval_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(saved, ensure_ascii=False) + "\n")

    print(f"Saved correctness evaluations to {eval_path}")
    return evals


def main():
    parser = argparse.ArgumentParser(description="GAIA Trajectory Error Analysis")
    parser.add_argument("--input_dir", required=True, help="Directory with trajectory JSONL files")
    parser.add_argument(
        "--classification_model_id",
        default="global.anthropic.claude-sonnet-4-6",
        help="Bedrock model ID for error classification",
    )
    parser.add_argument(
        "--eval_model_id",
        default="global.anthropic.claude-sonnet-4-6",
        help="Bedrock model ID for correctness evaluation (if needed)",
    )
    parser.add_argument("--region_name", default="us-east-1")
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    input_dir = args.input_dir.rstrip("/")

    # Step 1: Load trajectories
    print("=" * 60)
    print("Step 1: Loading trajectories")
    print("=" * 60)
    trajectories = load_trajectories(input_dir)
    qid_to_traj = {d["qid"]: d for d in trajectories}

    # Step 2: Load or run correctness evaluation
    print("\n" + "=" * 60)
    print("Step 2: Loading/running correctness evaluation")
    print("=" * 60)
    evals = load_evaluations(input_dir)
    if evals is None:
        print("No evaluation file found. Running correctness evaluation...")
        evals = run_correctness_eval(
            trajectories, input_dir, args.eval_model_id, args.region_name
        )

    # Step 3: Separate correct / incorrect / error
    print("\n" + "=" * 60)
    print("Step 3: Separating trajectories")
    print("=" * 60)

    correct_qids = []
    incorrect_qids = []
    error_qids = []  # status != success

    for traj in trajectories:
        qid = traj["qid"]
        if traj.get("status") != "success":
            error_qids.append(qid)
        elif qid in evals and is_correct(evals[qid]):
            correct_qids.append(qid)
        elif qid in evals:
            incorrect_qids.append(qid)
        else:
            # Not evaluated (shouldn't happen if eval covers all success)
            pass

    print(f"  Correct:   {len(correct_qids)}")
    print(f"  Incorrect: {len(incorrect_qids)}")
    print(f"  Error:     {len(error_qids)}")

    # Step 4: Classify incorrect trajectories
    print("\n" + "=" * 60)
    print("Step 4: Classifying error modes for incorrect trajectories")
    print("=" * 60)

    # Build items for classification
    items_to_classify = []
    for qid in incorrect_qids:
        traj = qid_to_traj[qid]
        eval_entry = evals[qid]
        items_to_classify.append({
            "qid": qid,
            "question": traj["question"],
            "correct_answer": eval_entry.get("correct_answer", traj.get("answer", "")),
            "messages": traj["messages"],
        })

    # Classify
    classified = []
    if items_to_classify:
        classifier = BedrockErrorClassifier(
            model_id=args.classification_model_id,
            region_name=args.region_name,
            qps=args.qps,
            max_workers=args.max_workers,
            max_tokens=args.max_tokens,
        )
        classified = classifier.classify_batch(items_to_classify)

    # Add E8 entries for error/timeout trajectories
    for qid in error_qids:
        traj = qid_to_traj[qid]
        classified.append({
            "qid": qid,
            "question": traj["question"],
            "correct_answer": traj.get("answer", ""),
            "agent_answer": "",
            "error_modes": ["E8"],
            "primary_error": "E8",
            "search_quality": "N/A",
            "explanation": f"Runtime error/timeout: status={traj.get('status', 'unknown')}, error={traj.get('error', 'unknown')}",
            "search_queries": extract_search_queries(traj.get("messages", [])),
            "num_tool_calls": count_tool_calls(traj.get("messages", [])),
        })

    # Step 5: Aggregate and display results
    print("\n" + "=" * 60)
    print("Step 5: Results")
    print("=" * 60)

    total = len(trajectories)
    correct_count = len(correct_qids)
    incorrect_count = len(incorrect_qids)
    error_count = len(error_qids)

    # Error mode counts (a trajectory can have multiple)
    error_mode_counts = Counter()
    primary_error_counts = Counter()
    search_quality_counts = Counter()

    for item in classified:
        for mode in item.get("error_modes", []):
            error_mode_counts[mode] += 1
        primary_error_counts[item.get("primary_error", "E9")] += 1
        search_quality_counts[item.get("search_quality", "unknown")] += 1

    # Summary table
    table = PrettyTable()
    table.title = f"Error Analysis Summary: {os.path.basename(input_dir)}"
    table.field_names = ["Metric", "Count", "Percentage"]
    table.align = "l"
    table.align["Count"] = "r"
    table.align["Percentage"] = "r"

    table.add_row(["Total Questions", total, "100.00%"])
    table.add_row(["  Correct", correct_count, f"{correct_count/total:.2%}" if total else "N/A"])
    table.add_row(["  Incorrect", incorrect_count, f"{incorrect_count/total:.2%}" if total else "N/A"])
    table.add_row(["  Error/Timeout", error_count, f"{error_count/total:.2%}" if total else "N/A"])
    table.add_row(["-" * 25, "-" * 8, "-" * 12], divider=True)
    overall_accuracy = correct_count / total if total else 0
    table.add_row(["Overall Accuracy", "", f"{overall_accuracy:.2%}"])

    print(table)

    # Error mode breakdown
    if classified:
        mode_table = PrettyTable()
        mode_table.title = "Error Mode Distribution (all occurrences)"
        mode_table.field_names = ["Error Mode", "ID", "Count", "% of Failures"]
        mode_table.align = "l"
        mode_table.align["Count"] = "r"
        mode_table.align["% of Failures"] = "r"

        failure_total = incorrect_count + error_count
        for eid in sorted(ERROR_MODES.keys()):
            cnt = error_mode_counts.get(eid, 0)
            if cnt > 0:
                mode_table.add_row([
                    ERROR_MODES[eid],
                    eid,
                    cnt,
                    f"{cnt/failure_total:.2%}" if failure_total else "N/A",
                ])

        print(f"\n{mode_table}")

        primary_table = PrettyTable()
        primary_table.title = "Primary Error Distribution"
        primary_table.field_names = ["Error Mode", "ID", "Count", "% of Failures"]
        primary_table.align = "l"
        primary_table.align["Count"] = "r"
        primary_table.align["% of Failures"] = "r"

        for eid in sorted(ERROR_MODES.keys()):
            cnt = primary_error_counts.get(eid, 0)
            if cnt > 0:
                primary_table.add_row([
                    ERROR_MODES[eid],
                    eid,
                    cnt,
                    f"{cnt/failure_total:.2%}" if failure_total else "N/A",
                ])

        print(f"\n{primary_table}")

        # Search quality breakdown
        print(f"\nSearch Quality: {dict(search_quality_counts)}")

    # Step 6: Save output files
    print("\n" + "=" * 60)
    print("Step 6: Saving output files")
    print("=" * 60)

    # Detailed output
    detailed_path = os.path.join(input_dir, "error_analysis_detailed.jsonl")
    classified_sorted = sorted(classified, key=lambda x: x.get("qid", 0))
    with open(detailed_path, "w", encoding="utf-8") as f:
        for item in classified_sorted:
            # Remove raw_classification to keep file size reasonable
            # Normalize key ordering for consistency across all entries
            save_item = {
                "qid": item.get("qid"),
                "question": item.get("question"),
                "correct_answer": item.get("correct_answer"),
                "agent_answer": item.get("agent_answer"),
                "error_modes": item.get("error_modes"),
                "primary_error": item.get("primary_error"),
                "search_quality": item.get("search_quality"),
                "explanation": item.get("explanation"),
                "search_queries": item.get("search_queries"),
                "num_tool_calls": item.get("num_tool_calls"),
            }
            f.write(json.dumps(save_item, ensure_ascii=False) + "\n")
    print(f"  Detailed results: {detailed_path}")

    # Summary output
    summary = {
        "input_dir": input_dir,
        "classification_model": args.classification_model_id,
        "total_questions": total,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "error_count": error_count,
        "overall_accuracy": round(overall_accuracy, 4),
        "error_mode_definitions": ERROR_MODES,
        "error_mode_counts": {
            eid: {"name": ERROR_MODES[eid], "count": error_mode_counts.get(eid, 0)}
            for eid in sorted(ERROR_MODES.keys())
            if error_mode_counts.get(eid, 0) > 0
        },
        "primary_error_distribution": {
            eid: {"name": ERROR_MODES[eid], "count": primary_error_counts.get(eid, 0)}
            for eid in sorted(ERROR_MODES.keys())
            if primary_error_counts.get(eid, 0) > 0
        },
        "search_quality_distribution": dict(search_quality_counts),
    }
    summary_path = os.path.join(input_dir, "error_analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary: {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
