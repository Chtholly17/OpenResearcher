"""Reward function for OpenResearcher RL training.

This module provides the reward computation used by verl's reward manager.
It is registered via the custom_reward_function config in the GRPO YAML:
    reward:
      custom_reward_function:
        path: verl_rl/reward/openresearcher_reward.py
        name: compute_score

The reward manager calls compute_score(data_source=..., solution_str=...,
ground_truth=..., extra_info=...) for each trajectory. We extract the answer
from the model's response using <answer> tags or other formats, then compare
against the ground truth using normalized string matching.
"""

import re
import os
import json
from typing import Optional

_DEBUG_LOG = os.environ.get("REWARD_DEBUG_LOG", "")
_debug_count = 0


def _debug_log(msg: str):
    global _debug_count
    if not _DEBUG_LOG:
        return
    _debug_count += 1
    if _debug_count <= 500:  # Log up to 500 samples
        with open(_DEBUG_LOG, "a") as f:
            f.write(msg + "\n")


def extract_answer(text: str) -> tuple[Optional[str], bool]:
    """Extract final answer from model output.

    Returns:
        (answer_text, is_explicit): The extracted answer and whether it was
        from an explicit submission (answer tags, submit_answer tool) vs
        a fallback extraction from thinking content.
    """
    if not text:
        return None, False

    # <answer>...</answer> (from submit_answer tool or explicit tags)
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), True

    # "Exact Answer:" format
    match = re.search(r"Exact Answer:\s*(.*?)(?:\n|Confidence:|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip(), True

    # "Final Answer:" format
    match = re.search(r"Final Answer:\s*(.*?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip(), True

    # --- Fallback: extract answer-like content from last <think> block ---
    # The SFT model was never trained to call submit_answer. But its <think>
    # blocks often contain reasoning about what the answer is. Extracting
    # this gives partial reward signal so RL has a gradient to optimize.
    think_blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_blocks:
        last_think = think_blocks[-1]
        # Look for "the answer is X" patterns in thinking
        for pattern in [
            r"(?:the answer (?:is|should be|would be|appears to be))\s*[:\"]?\s*(.+?)(?:\.|\"|\n|$)",
            r"(?:answer)\s*[:=]\s*(.+?)(?:\.|\"|\n|$)",
            r"(?:so|therefore|thus|hence),?\s+(?:the answer (?:is|should be))\s+(.+?)(?:\.|\"|\n|$)",
        ]:
            m = re.search(pattern, last_think, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip().strip('"').strip("'")
                if 3 < len(candidate) < 500:  # Sanity check length
                    return candidate, False

    return None, False


def strip_boxed(text: str) -> str:
    """Strip \\boxed{...} wrapper from ground truth answers."""
    match = re.match(r"^\\boxed\{(.*)\}$", text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def normalize_answer(text: str) -> str:
    """Normalize text for comparison."""
    text = strip_boxed(text)
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compute_efficiency(num_turns: int, resp_len: int) -> float:
    """Compute an efficiency factor in [0, 1] based on trajectory length.

    Uses both turn count and response length (in characters) to measure how
    efficiently the model reached its answer.  The two signals are combined
    with equal weight so that a model is rewarded for using fewer tool calls
    AND generating fewer tokens.

    Reference points (from SFT data analysis):
      - Median trajectory: ~67 rounds (~134 turns), ~34K response chars
      - p25 trajectory:   ~30 rounds (~60 turns),  ~15K response chars
      - p90 trajectory:  ~220 rounds (~440 turns), ~120K response chars

    The efficiency curve maps these to roughly:
      - p25 (fast)   → efficiency ≈ 0.85
      - p50 (median) → efficiency ≈ 0.55
      - p90 (slow)   → efficiency ≈ 0.05
    """
    # Turn efficiency: linear decay from 1.0 at 0 turns to 0.0 at ceiling
    TURN_CEILING = 1000  # max_assistant_turns(500) + max_user_turns(500)
    turn_eff = max(0.0, 1.0 - num_turns / TURN_CEILING)

    # Length efficiency: linear decay from 1.0 at 0 chars to 0.0 at ceiling
    LEN_CEILING = 250_000
    len_eff = max(0.0, 1.0 - resp_len / LEN_CEILING)

    # Equal-weight combination
    return 0.5 * turn_eff + 0.5 * len_eff


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Compute reward score for a research answer.

    Reward structure with efficiency scaling:

      Correct + efficient   → up to 1.2  (base 1.0 + efficiency bonus 0.2)
      Correct + inefficient → down to 0.8 (base 1.0 - efficiency penalty 0.2)
      Wrong + efficient     → 0.3         (format reward, fixed)
      Wrong + inefficient   → down to 0.1 (penalise wasted long trajectories)
      No answer             → 0.0

    The efficiency component creates a clear gradient:
      - For correct answers: small bonus/penalty (±0.2) — correctness dominates
      - For wrong answers: larger penalty (0.3→0.1) — long wrong trajectories
        are the worst outcome and receive the lowest non-zero reward
    """
    answer, is_explicit = extract_answer(solution_str)

    num_turns = int(extra_info.get("num_turns", 0)) if extra_info else 0
    resp_len = len(solution_str) if solution_str else 0
    efficiency = _compute_efficiency(num_turns, resp_len)

    if answer is None:
        _debug_log(json.dumps({
            "event": "no_answer",
            "ground_truth": ground_truth,
            "num_turns": num_turns,
            "resp_len": resp_len,
            "efficiency": round(efficiency, 3),
            "resp_tail_500": (solution_str[-500:] if solution_str else ""),
            "score": 0.0,
        }, ensure_ascii=False))
        return 0.0

    pred_norm = normalize_answer(answer)
    gt_norm = normalize_answer(ground_truth)

    is_correct = (
        bool(pred_norm and gt_norm)
        and (pred_norm == gt_norm or gt_norm in pred_norm or pred_norm in gt_norm)
    )

    if is_explicit:
        if is_correct:
            # Base 1.0, efficiency bonus/penalty in [-0.2, +0.2]
            score = 1.0 + 0.2 * (2 * efficiency - 1.0)  # maps eff 0→0.8, 0.5→1.0, 1→1.2
        else:
            # Wrong explicit answer: base 0.3, penalise long wrong trajectories
            # Maps eff 0→0.1, 0.5→0.2, 1→0.3
            score = 0.1 + 0.2 * efficiency
    else:
        # Fallback extraction (from <think> blocks)
        if is_correct:
            score = 0.8 + 0.1 * (2 * efficiency - 1.0)  # maps eff 0→0.7, 0.5→0.8, 1→0.9
        else:
            score = 0.05 + 0.1 * efficiency  # maps eff 0→0.05, 0.5→0.1, 1→0.15

    _debug_log(json.dumps({
        "event": "answer_found",
        "answer_extracted": answer[:200],
        "is_explicit": is_explicit,
        "is_correct": is_correct,
        "ground_truth": ground_truth,
        "pred_norm": pred_norm[:100],
        "gt_norm": gt_norm[:100],
        "num_turns": num_turns,
        "resp_len": resp_len,
        "efficiency": round(efficiency, 3),
        "score": round(score, 4),
    }, ensure_ascii=False))
    return score
