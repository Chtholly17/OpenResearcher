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
    # Use the LAST match: the interaction may tell the model "wrong, try again",
    # causing multiple submissions. The final attempt is the best answer.
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip(), True

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


def _count_search_calls(text: str) -> int:
    """Count browser.search tool calls in the trajectory."""
    return len(re.findall(r'"name"\s*:\s*"browser\.search"', text))


def _length_efficiency(num_turns: int) -> float:
    """Efficiency factor in [0, 1] based on number of turns.

    Linear decay: 1.0 at 0 turns → 0.0 at CEILING turns.
    Reference points for high-pass data (correct answers avg ~40 turns):
      10 turns  → 0.98  (very efficient)
      50 turns  → 0.90  (efficient)
      250 turns → 0.50  (midpoint, score=1.0)
      500 turns → 0.00  (at ceiling, score=0.8)
    """
    TURN_CEILING = 500
    return max(0.0, 1.0 - num_turns / TURN_CEILING)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Compute reward score for a research answer.

    v0.5 reward structure — three signals combined:

      1. Correctness:   did the model get the right answer?
      2. Search effort: did the model use browser.search (≥1 call)?
      3. Efficiency:    how quickly did it reach the correct answer?

    Score table:
      Correct + ≥1 searches  → 0.8 + 0.4 * eff   ∈ [0.8, 1.2]
                                 short (10 turns)  → ~1.19
                                 medium (50 turns) → ~1.16
                                 long (250 turns)  →  1.00
                                 very long (500+)  →  0.80
      Correct + 0 searches   → 0.3   (memory recall, no length bonus)
      Wrong   + ≥1 searches  → 0.1   (searched but wrong, fixed)
      Wrong   + 0 searches   → 0.0   (guessed without searching)
      No answer              → 0.0

    Key design decisions vs v0.4:
    - Length bonus ONLY on correct+searched: prevents the v0.1 problem where
      wrong answers clustered at similar efficiency-scaled values.
    - Memory recall penalty (0.3 vs 1.0): discourages answering from
      pretraining knowledge without engaging tools (75% of v0.4 correct
      answers had 0-1 searches, polluting the gradient signal).
    - Wrong+no-search = 0.0: removes incentive for pure guessing.
    """
    answer, is_explicit = extract_answer(solution_str)

    num_turns = int(extra_info.get("num_turns", 0)) if extra_info else 0
    resp_len = len(solution_str) if solution_str else 0
    n_searches = _count_search_calls(solution_str) if solution_str else 0

    if answer is None:
        _debug_log(json.dumps({
            "event": "no_answer",
            "ground_truth": ground_truth,
            "num_turns": num_turns,
            "n_searches": n_searches,
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

    searched = n_searches >= 1

    if is_explicit:
        if is_correct and searched:
            eff = _length_efficiency(num_turns)
            score = 0.8 + 0.4 * eff          # [0.8, 1.2]
        elif is_correct and not searched:
            score = 0.3                        # memory recall — partial credit
        elif not is_correct and searched:
            score = 0.1                        # searched but wrong
        else:
            score = 0.0                        # guessed without searching
    else:
        # Fallback extraction from <think> blocks — lower confidence
        score = 0.5 if is_correct else 0.05

    _debug_log(json.dumps({
        "event": "answer_found",
        "answer_extracted": answer[:200],
        "is_explicit": is_explicit,
        "is_correct": is_correct,
        "searched": searched,
        "ground_truth": ground_truth,
        "pred_norm": pred_norm[:100],
        "gt_norm": gt_norm[:100],
        "num_turns": num_turns,
        "n_searches": n_searches,
        "score": round(score, 4),
    }, ensure_ascii=False))
    return score
