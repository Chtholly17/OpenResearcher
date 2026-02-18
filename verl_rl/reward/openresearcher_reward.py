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
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from model output."""
    if not text:
        return None

    # <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # "Exact Answer:" format
    match = re.search(r"Exact Answer:\s*(.*?)(?:\n|Confidence:|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # "Final Answer:" format
    match = re.search(r"Final Answer:\s*(.*?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


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


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Compute reward score for a research answer.

    This follows verl's reward function calling convention. The NaiveRewardManager
    calls this as:
        score = compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

    Args:
        data_source: Dataset identifier (e.g., "OpenResearcher/OpenResearcher")
        solution_str: The full model output text (decoded response tokens)
        ground_truth: The correct answer string from reward_model.ground_truth
        extra_info: Additional info dict from the data record
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        Float reward:
          - 1.0 for correct answer
          - 0.1 for submitting any answer (format reward, encourages answer submission)
          - 0.0 for no answer submitted (model exhausted token budget on tool calls)
    """
    answer = extract_answer(solution_str)
    if answer is None:
        # No answer submitted — zero reward.
        # This is the cold-start problem: the model must learn to stop
        # researching and submit an answer within the token budget.
        return 0.0

    pred_norm = normalize_answer(answer)
    gt_norm = normalize_answer(ground_truth)

    if not pred_norm or not gt_norm:
        return 0.1  # Submitted but empty/unparseable — small format reward

    # Exact match
    if pred_norm == gt_norm:
        return 1.0

    # Containment match (common for short factual answers)
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return 1.0

    # Wrong answer — still give format reward for submitting
    return 0.1
